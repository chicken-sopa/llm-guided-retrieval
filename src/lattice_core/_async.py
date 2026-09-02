"""Running the traversal coroutines from synchronous callers.

Kept here so importing `lattice_core` never drags in the tree BUILDER
(`src/tree_construction/build_llm_bottom_up_tree.py`), which needs `datasets`
and `scikit-learn.cluster` that inference has no use for. The builder keeps its
own copy of `run_coro_sync`; this is the one the retriever uses.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any


def run_coro_sync(coro: Any) -> Any:
    """Run an async coroutine from either scripts or notebook cells.

    Jupyter already owns an event loop, so calling ``asyncio.run`` directly from
    a notebook raises ``RuntimeError``. In that case we run the coroutine in a
    short-lived background thread that gets its own event loop.

    NOTE for async callers: this gives each call a FRESH event loop, so an async
    host (an agent framework, a web service) should await `retrieve_async` /
    `retrieve_many_async` directly rather than paying for a thread hop and
    rebinding the LLM client to a new loop on every query.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _thread_main() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["value"] = exc

    thread = threading.Thread(target=_thread_main, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")
