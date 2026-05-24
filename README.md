# llm-batch-coalesce

[![PyPI](https://img.shields.io/pypi/v/llm-batch-coalesce.svg)](https://pypi.org/project/llm-batch-coalesce/)
[![Python](https://img.shields.io/pypi/pyversions/llm-batch-coalesce.svg)](https://pypi.org/project/llm-batch-coalesce/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Request coalescing (single-flight) for LLM calls.** Many concurrent
callers ask for the same key, only one underlying call fires. Everyone
receives that one result.

This is the small, focused primitive behind any "thundering herd" fix
for LLM endpoints, embedding endpoints, or any expensive idempotent
function. asyncio and threading. Zero runtime deps.

## Install

```bash
pip install llm-batch-coalesce
```

## What it does

If 100 coroutines ask for the same key at the same time, you want one
network call, not 100. Layered with a cache, you also want the herd that
arrives during the first miss to wait on that one in-flight call rather
than each firing their own duplicate miss.

This library is just that wait coordination. Nothing else.

## Async example

```python
import asyncio
from llm_batch_coalesce import AsyncRequestCoalescer

coalescer = AsyncRequestCoalescer()

calls = 0

async def expensive_llm_call(prompt: str) -> str:
    global calls
    calls += 1
    await asyncio.sleep(0.5)
    return f"answer to: {prompt}"

async def main():
    # 100 coroutines, all asking for the same prompt at the same time.
    results = await asyncio.gather(*[
        coalescer.call("hello", expensive_llm_call, "hello")
        for _ in range(100)
    ])
    print(f"got {len(results)} answers from {calls} underlying call")
    # got 100 answers from 1 underlying call

asyncio.run(main())
```

## Sync example

```python
from concurrent.futures import ThreadPoolExecutor
from llm_batch_coalesce import RequestCoalescer

coalescer = RequestCoalescer()

def expensive_llm_call(prompt: str) -> str:
    # ... real HTTP call
    return f"answer to: {prompt}"

with ThreadPoolExecutor(max_workers=50) as pool:
    futs = [
        pool.submit(coalescer.call, "hello", expensive_llm_call, "hello")
        for _ in range(50)
    ]
    results = [f.result() for f in futs]
```

## Stats

```python
s = coalescer.stats()
s.inflight_count       # how many keys are in-flight right now
s.coalesced_calls      # callers that joined an in-flight call (saved work)
s.underlying_calls     # actual fn invocations that fired
coalescer.reset_stats()
```

`coalesced_calls / (coalesced_calls + underlying_calls)` is your
deduplication hit-ratio for the window.

## Cancellation

```python
await coalescer.cancel("hello")   # async
coalescer.cancel("hello")         # sync
```

Waiters wake with `asyncio.CancelledError` (async) or
`concurrent.futures.CancelledError` (sync). The leader coroutine or
thread that is actually running `fn` is not interrupted, because Python
has no safe way to stop an arbitrary worker mid-call. Its result is
discarded as far as new waiters are concerned (the entry is gone), so
the next caller with the same key starts a fresh call.

## What it does NOT do

- **No caching across completions.** As soon as the underlying call
  finishes (success or exception), the in-flight entry is dropped. The
  next call with the same key fires a fresh underlying call. If you
  want a cache, layer one on top.
  See [`cachebench`](https://pypi.org/project/cachebench/) for prompt-cache
  hit-ratio observability and miss-aware retry.
- **No key derivation.** You pass `key`. If you want a stable key derived
  from an LLM request structure (messages + model + tools + temperature,
  with provider-aware noise fields dropped), see
  [`llm-message-hash-py`](https://pypi.org/project/llm-message-hash-py/).
- **No retries.** Compose with your own retry layer.
- **No HTTP.** Doesn't talk to any LLM provider.

## License

MIT
