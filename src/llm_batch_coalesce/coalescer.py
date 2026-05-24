"""Core RequestCoalescer and AsyncRequestCoalescer implementations.

Single-flight: when many concurrent callers ask for the same key, only one
underlying `fn` invocation fires. All waiters receive the same result (or
the same exception).

The in-flight entry is removed as soon as the underlying call completes.
That means the next call with the same key fires a fresh underlying call.
For caching across completions, layer something like `cachebench` on top.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError as ThreadCancelledError
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CoalescerStats:
    """Counters snapshot. Returned by `.stats()`."""

    inflight_count: int
    coalesced_calls: int
    underlying_calls: int


class RequestCoalescer:
    """Thread-based single-flight coalescer.

    Use one instance per logical namespace (e.g. one per LLM provider /
    model). Concurrent threaded callers with the same `key` share one
    underlying `fn` invocation. Exceptions propagate to every waiter.

    The in-flight entry is removed after completion. The next call with
    the same key fires a fresh underlying call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[Any, Future[Any]] = {}
        # counters since last reset_stats()
        self._coalesced_calls = 0
        self._underlying_calls = 0

    def call(
        self,
        key: Any,
        fn: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run `fn(*args, **kwargs)` under coalescing `key`.

        If another caller is already running for the same key, this caller
        waits on that in-flight Future and returns its result. Otherwise
        this caller runs `fn` and shares the result with any future waiters
        that join during the call.
        """
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                self._coalesced_calls += 1
                fut: Future[T] = existing  # type: ignore[assignment]
                is_leader = False
            else:
                fut = Future()
                self._inflight[key] = fut
                self._underlying_calls += 1
                is_leader = True

        if is_leader:
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                self._finish_and_remove(key, fut, exc=exc)
                raise
            self._finish_and_remove(key, fut, result=result)
            return result

        # follower path: just wait. concurrent.futures.Future.result() will
        # re-raise whatever exception the leader saw, or return the result.
        return fut.result()

    def cancel(self, key: Any) -> bool:
        """Cancel the in-flight call for `key`, if any.

        Returns True if a call was found and cancellation was attempted.
        Waiters on the cancelled Future will see
        `concurrent.futures.CancelledError`. The leader thread that is
        actually running `fn` is not interrupted (Python has no safe way
        to stop an arbitrary worker thread), but its eventual result will
        be discarded as far as new waiters are concerned because the
        entry has already been removed.
        """
        with self._lock:
            fut = self._inflight.pop(key, None)
        if fut is None:
            return False
        # Future.cancel() only succeeds while PENDING. If the future is
        # already RUNNING (set_running_or_notify_cancel was called), we
        # fall back to setting a CancelledError so waiters wake up.
        if not fut.done() and not fut.cancel():
            with contextlib.suppress(Exception):
                fut.set_exception(ThreadCancelledError())
        return True

    def stats(self) -> CoalescerStats:
        with self._lock:
            return CoalescerStats(
                inflight_count=len(self._inflight),
                coalesced_calls=self._coalesced_calls,
                underlying_calls=self._underlying_calls,
            )

    def reset_stats(self) -> None:
        with self._lock:
            self._coalesced_calls = 0
            self._underlying_calls = 0

    def _finish_and_remove(
        self,
        key: Any,
        fut: Future[Any],
        /,
        *,
        result: Any = None,
        exc: BaseException | None = None,
    ) -> None:
        """Pop the entry and resolve waiters. The leader always removes its
        own entry, unless cancel(key) already swapped it out for a fresh
        run (in which case we leave the new entry alone)."""
        with self._lock:
            current = self._inflight.get(key)
            if current is fut:
                self._inflight.pop(key, None)
        if fut.done():
            # already cancelled or otherwise resolved by another path
            return
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)


class AsyncRequestCoalescer:
    """asyncio-based single-flight coalescer.

    Concurrent coroutines with the same `key` share one underlying `fn`
    invocation. `fn` must be an awaitable (or a coroutine function).
    Exceptions propagate to every waiter.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._inflight: dict[Any, asyncio.Future[Any]] = {}
        self._coalesced_calls = 0
        self._underlying_calls = 0

    async def call(
        self,
        key: Any,
        fn: Callable[..., Awaitable[T]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run `await fn(*args, **kwargs)` under coalescing `key`.

        If another coroutine is already running for the same key, this
        caller awaits the in-flight Future. Otherwise this caller fires
        `fn` and shares the result with any future waiters.
        """
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                self._coalesced_calls += 1
                fut: asyncio.Future[T] = existing  # type: ignore[assignment]
                is_leader = False
            else:
                # bind to the running loop so multiple loops in one process
                # do not cross wires.
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                self._inflight[key] = fut
                self._underlying_calls += 1
                is_leader = True

        if is_leader:
            try:
                result = await fn(*args, **kwargs)
            except BaseException as exc:
                await self._finish_and_remove(key, fut, exc=exc)
                raise
            await self._finish_and_remove(key, fut, result=result)
            return result

        # follower path
        return await fut

    async def cancel(self, key: Any) -> bool:
        """Cancel the in-flight call for `key`, if any.

        Waiters on the cancelled Future will see `asyncio.CancelledError`.
        The leader coroutine is not actively cancelled mid-await (that
        would require holding a Task reference), but its result will be
        discarded for new waiters because the entry is removed.
        """
        async with self._lock:
            fut = self._inflight.pop(key, None)
        if fut is None:
            return False
        if not fut.done():
            fut.cancel()
        return True

    def stats(self) -> CoalescerStats:
        return CoalescerStats(
            inflight_count=len(self._inflight),
            coalesced_calls=self._coalesced_calls,
            underlying_calls=self._underlying_calls,
        )

    def reset_stats(self) -> None:
        self._coalesced_calls = 0
        self._underlying_calls = 0

    async def _finish_and_remove(
        self,
        key: Any,
        fut: asyncio.Future[Any],
        /,
        *,
        result: Any = None,
        exc: BaseException | None = None,
    ) -> None:
        async with self._lock:
            current = self._inflight.get(key)
            if current is fut:
                self._inflight.pop(key, None)
        if fut.done():
            return
        if exc is not None:
            fut.set_exception(exc)
            # Mark the exception as retrieved if no follower is waiting on
            # this future. Otherwise asyncio logs "Future exception was
            # never retrieved" for the leader-only case (the leader raises
            # to its caller directly; the future is purely internal).
            # `_asyncio_future_blocking` doesn't help us here, so just
            # call .exception() to clear the flag. Followers that arrived
            # via `return await fut` already retrieved their copy by the
            # time they re-raise.
            fut.exception()
        else:
            fut.set_result(result)
