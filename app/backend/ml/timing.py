"""Decorator for recording per-node wall-clock timing into RAGState."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any


def timed_node(node_name: str) -> Callable:
    """Wrap an async LangGraph node to record its execution time in ms.

    The elapsed time is merged into ``state["node_timings"]`` under *node_name*.
    If the wrapped function already sets ``node_timings`` in its return dict,
    the timing entry is added on top; otherwise a fresh dict entry is created.

    Args:
        node_name: Key used in the ``node_timings`` dict (e.g. "generator").

    Returns:
        Decorator that wraps the node coroutine.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            start = time.perf_counter()
            result: dict[str, Any] = await func(state)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            existing: dict[str, int] = {
                **(state.get("node_timings") or {}),
                **(result.get("node_timings") or {}),
            }
            existing[node_name] = elapsed_ms
            result["node_timings"] = existing
            return result

        return wrapper
    return decorator
