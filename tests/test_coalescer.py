import asyncio
import threading
import time
from concurrent.futures import CancelledError as ThreadCancelledError
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_batch_coalesce import (
    AsyncRequestCoalescer,
    CoalescerStats,
    RequestCoalescer,
)

# ============================================================
# RequestCoalescer (threaded)
# ============================================================


def test_single_call_returns_result():
    c = RequestCoalescer()
    calls = []

    def fn(x):
        calls.append(x)
        return x * 2

    assert c.call("k", fn, 21) == 42
    assert calls == [21]
    s = c.stats()
    assert s.underlying_calls == 1
    assert s.coalesced_calls == 0
    assert s.inflight_count == 0


def test_50_concurrent_callers_share_one_underlying_call():
    c = RequestCoalescer()
    invocations = 0
    invocations_lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()

    def fn():
        nonlocal invocations
        with invocations_lock:
            invocations += 1
        started.set()
        # block until all followers have arrived so coalescing has a chance
        release.wait(timeout=5)
        return "shared-result"

    n = 50
    with ThreadPoolExecutor(max_workers=n) as pool:
        # leader first so it has time to take the slot and start fn
        leader = pool.submit(c.call, "k", fn)
        # wait for fn to actually start before we send followers in
        assert started.wait(timeout=5)
        followers = [pool.submit(c.call, "k", fn) for _ in range(n - 1)]
        release.set()
        results = [leader.result(timeout=5)] + [f.result(timeout=5) for f in followers]

    assert invocations == 1
    assert results == ["shared-result"] * n
    s = c.stats()
    assert s.underlying_calls == 1
    assert s.coalesced_calls == n - 1
    assert s.inflight_count == 0


def test_different_keys_run_in_parallel():
    c = RequestCoalescer()
    barrier = threading.Barrier(3, timeout=5)

    def fn(name):
        barrier.wait()  # blocks unless 3 threads are inside
        return name

    with ThreadPoolExecutor(max_workers=3) as pool:
        fs = [pool.submit(c.call, k, fn, k) for k in ("a", "b", "c")]
        results = sorted(f.result(timeout=5) for f in fs)

    assert results == ["a", "b", "c"]
    s = c.stats()
    assert s.underlying_calls == 3
    assert s.coalesced_calls == 0


def test_exception_propagates_to_all_waiters():
    c = RequestCoalescer()
    started = threading.Event()
    release = threading.Event()

    class Boom(RuntimeError):
        pass

    def fn():
        started.set()
        release.wait(timeout=5)
        raise Boom("nope")

    with ThreadPoolExecutor(max_workers=10) as pool:
        leader = pool.submit(c.call, "k", fn)
        assert started.wait(timeout=5)
        followers = [pool.submit(c.call, "k", fn) for _ in range(9)]
        release.set()

        for fut in [leader, *followers]:
            with pytest.raises(Boom):
                fut.result(timeout=5)

    assert c.stats().inflight_count == 0


def test_after_exception_same_key_fires_fresh():
    c = RequestCoalescer()
    n = [0]

    def fn():
        n[0] += 1
        raise RuntimeError("first")

    with pytest.raises(RuntimeError):
        c.call("k", fn)
    with pytest.raises(RuntimeError):
        c.call("k", fn)

    assert n[0] == 2  # second call fired fresh, not deduped
    s = c.stats()
    assert s.underlying_calls == 2
    assert s.coalesced_calls == 0


def test_after_success_same_key_fires_fresh():
    c = RequestCoalescer()
    n = [0]

    def fn():
        n[0] += 1
        return n[0]

    assert c.call("k", fn) == 1
    assert c.call("k", fn) == 2
    assert n[0] == 2


def test_stats_and_reset():
    c = RequestCoalescer()
    c.call("a", lambda: 1)
    c.call("b", lambda: 2)
    assert c.stats().underlying_calls == 2
    c.reset_stats()
    s = c.stats()
    assert s.underlying_calls == 0
    assert s.coalesced_calls == 0


def test_args_and_kwargs_are_forwarded():
    c = RequestCoalescer()

    def fn(a, b, *, c_kw):
        return (a, b, c_kw)

    out = c.call("k", fn, 1, 2, c_kw=3)
    assert out == (1, 2, 3)


def test_cancel_on_unknown_key_returns_false():
    c = RequestCoalescer()
    assert c.cancel("missing") is False


def test_cancel_pending_call_wakes_waiters():
    c = RequestCoalescer()
    started = threading.Event()
    release = threading.Event()

    def fn():
        started.set()
        release.wait(timeout=5)
        return "should-be-discarded-for-followers"

    with ThreadPoolExecutor(max_workers=5) as pool:
        leader = pool.submit(c.call, "k", fn)
        assert started.wait(timeout=5)
        followers = [pool.submit(c.call, "k", fn) for _ in range(3)]
        # cancel before releasing fn
        assert c.cancel("k") is True

        # followers should see CancelledError
        for f in followers:
            with pytest.raises(ThreadCancelledError):
                f.result(timeout=5)

        # leader can still finish; let it
        release.set()
        # leader returns its own value, then the finalizer is a no-op for
        # the followers because the entry was swapped out by cancel.
        assert leader.result(timeout=5) == "should-be-discarded-for-followers"


def test_stats_returns_dataclass():
    c = RequestCoalescer()
    s = c.stats()
    assert isinstance(s, CoalescerStats)
    assert s.inflight_count == 0


def test_inflight_count_visible_during_call():
    c = RequestCoalescer()
    seen = []
    release = threading.Event()
    started = threading.Event()

    def fn():
        started.set()
        release.wait(timeout=5)
        return 1

    t = threading.Thread(target=lambda: c.call("k", fn))
    t.start()
    try:
        assert started.wait(timeout=5)
        seen.append(c.stats().inflight_count)
    finally:
        release.set()
        t.join(timeout=5)

    assert seen == [1]
    assert c.stats().inflight_count == 0


def test_after_cancel_same_key_fires_fresh():
    c = RequestCoalescer()
    started = threading.Event()
    release = threading.Event()
    invocations = [0]

    def fn():
        invocations[0] += 1
        started.set()
        release.wait(timeout=5)
        return "leader"

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(c.call, "k", fn)
        assert started.wait(timeout=5)
        # cancel removes the in-flight entry; the running leader is not stopped
        assert c.cancel("k") is True
        assert c.stats().inflight_count == 0
        release.set()
        assert leader.result(timeout=5) == "leader"

        # a fresh call with the same key must fire a new underlying call
        started.clear()
        release.set()  # second fn won't block since release is already set
        assert c.call("k", fn) == "leader"

    assert invocations[0] == 2


# ============================================================
# AsyncRequestCoalescer (asyncio)
# ============================================================


async def test_async_single_call_returns_result():
    c = AsyncRequestCoalescer()

    async def fn(x):
        return x + 1

    assert await c.call("k", fn, 41) == 42
    s = c.stats()
    assert s.underlying_calls == 1
    assert s.coalesced_calls == 0
    assert s.inflight_count == 0


async def test_async_100_callers_share_one_underlying_call():
    c = AsyncRequestCoalescer()
    invocations = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def fn():
        nonlocal invocations
        invocations += 1
        started.set()
        await release.wait()
        return "shared"

    # leader first
    leader_task = asyncio.create_task(c.call("k", fn))
    await started.wait()  # leader is now inside fn

    follower_tasks = [asyncio.create_task(c.call("k", fn)) for _ in range(99)]
    # give the followers a tick to enqueue
    await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(leader_task, *follower_tasks)
    assert invocations == 1
    assert results == ["shared"] * 100
    s = c.stats()
    assert s.underlying_calls == 1
    assert s.coalesced_calls == 99
    assert s.inflight_count == 0


async def test_async_different_keys_run_in_parallel():
    c = AsyncRequestCoalescer()
    order = []

    async def fn(name):
        order.append(("start", name))
        await asyncio.sleep(0.01)
        order.append(("end", name))
        return name

    results = await asyncio.gather(
        c.call("a", fn, "a"),
        c.call("b", fn, "b"),
        c.call("c", fn, "c"),
    )
    assert sorted(results) == ["a", "b", "c"]
    # All three should have started before any ended
    starts = [e for e in order if e[0] == "start"]
    ends = [e for e in order if e[0] == "end"]
    assert len(starts) == 3
    assert len(ends) == 3
    s = c.stats()
    assert s.underlying_calls == 3
    assert s.coalesced_calls == 0


async def test_async_exception_propagates_to_all_waiters():
    c = AsyncRequestCoalescer()
    started = asyncio.Event()
    release = asyncio.Event()

    class Boom(RuntimeError):
        pass

    async def fn():
        started.set()
        await release.wait()
        raise Boom("kaboom")

    leader = asyncio.create_task(c.call("k", fn))
    await started.wait()
    followers = [asyncio.create_task(c.call("k", fn)) for _ in range(5)]
    await asyncio.sleep(0)
    release.set()

    for t in [leader, *followers]:
        with pytest.raises(Boom):
            await t

    assert c.stats().inflight_count == 0


async def test_async_after_exception_same_key_fires_fresh():
    c = AsyncRequestCoalescer()
    n = 0

    async def fn():
        nonlocal n
        n += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await c.call("k", fn)
    with pytest.raises(RuntimeError):
        await c.call("k", fn)
    assert n == 2


async def test_async_args_and_kwargs_are_forwarded():
    c = AsyncRequestCoalescer()

    async def fn(a, b, *, c_kw):
        return (a, b, c_kw)

    assert await c.call("k", fn, 1, 2, c_kw=3) == (1, 2, 3)


async def test_async_cancel_unknown_key_returns_false():
    c = AsyncRequestCoalescer()
    assert await c.cancel("missing") is False


async def test_async_cancel_wakes_waiters():
    c = AsyncRequestCoalescer()
    started = asyncio.Event()
    release = asyncio.Event()

    async def fn():
        started.set()
        await release.wait()
        return "leader-finished"

    leader = asyncio.create_task(c.call("k", fn))
    await started.wait()
    followers = [asyncio.create_task(c.call("k", fn)) for _ in range(3)]
    await asyncio.sleep(0)

    assert await c.cancel("k") is True

    for f in followers:
        with pytest.raises(asyncio.CancelledError):
            await f

    # leader can still complete; its return value is not delivered to
    # followers because their future is gone.
    release.set()
    assert await leader == "leader-finished"
    assert c.stats().inflight_count == 0


async def test_async_reset_stats():
    c = AsyncRequestCoalescer()

    async def fn():
        return 1

    await c.call("a", fn)
    await c.call("b", fn)
    assert c.stats().underlying_calls == 2
    c.reset_stats()
    s = c.stats()
    assert s.underlying_calls == 0
    assert s.coalesced_calls == 0


async def test_async_after_cancel_same_key_fires_fresh():
    c = AsyncRequestCoalescer()
    started = asyncio.Event()
    release = asyncio.Event()
    invocations = 0

    async def fn():
        nonlocal invocations
        invocations += 1
        started.set()
        await release.wait()
        return "leader"

    leader = asyncio.create_task(c.call("k", fn))
    await started.wait()
    assert await c.cancel("k") is True
    assert c.stats().inflight_count == 0
    release.set()
    assert await leader == "leader"

    # a fresh call with the same key must fire a new underlying call
    started.clear()
    assert await c.call("k", fn) == "leader"
    assert invocations == 2


# ============================================================
# tiny smoke: leader's invocation time roughly equals fn time
# (no sequential serialization across followers)
# ============================================================


async def test_async_followers_do_not_serialize_underlying_call():
    c = AsyncRequestCoalescer()

    async def fn():
        await asyncio.sleep(0.05)
        return "ok"

    t0 = time.perf_counter()
    results = await asyncio.gather(*[c.call("k", fn) for _ in range(20)])
    elapsed = time.perf_counter() - t0

    assert results == ["ok"] * 20
    # 20 sequential 50ms calls would be ~1s. Coalesced is one ~50ms call.
    # Allow a healthy margin for scheduler overhead.
    assert elapsed < 0.5


# ============================================================
# packaging: the library is fully typed, so it must ship a
# PEP 561 py.typed marker or downstream type checkers ignore it.
# ============================================================


def test_package_ships_py_typed_marker():
    from importlib.resources import files

    assert (files("llm_batch_coalesce") / "py.typed").is_file()
