"""llm-batch-coalesce - request coalescing (single-flight) for LLM calls.

When many concurrent callers ask for the same key, only one underlying
call fires; every waiter receives that one result (or that one exception).

Threaded:

    from llm_batch_coalesce import RequestCoalescer

    coalescer = RequestCoalescer()
    result = coalescer.call("hello", expensive_llm_call, "hello")

asyncio:

    from llm_batch_coalesce import AsyncRequestCoalescer

    coalescer = AsyncRequestCoalescer()
    result = await coalescer.call("hello", expensive_llm_call, "hello")

The in-flight entry is removed as soon as the call completes, so the next
call with the same key fires a fresh underlying call. For caching across
completions, layer something like `cachebench` on top. For deriving keys
from request objects, see `llm-message-hash-py`.
"""

from llm_batch_coalesce.coalescer import (
    AsyncRequestCoalescer,
    CoalescerStats,
    RequestCoalescer,
)

__version__ = "0.1.0"

__all__ = [
    "AsyncRequestCoalescer",
    "CoalescerStats",
    "RequestCoalescer",
    "__version__",
]
