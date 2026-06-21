"""In-process background RAG task manager.

A single-user app has at most one active conversation, so we keep one task slot.
The task survives WebSocket disconnects (page refresh) and lets reconnecting clients
receive all buffered events and the final answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from backend.chat import service
from backend.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


@dataclass
class ConvTask:
    """State of a running (or recently finished) background RAG task."""

    events: list[str] = field(default_factory=list)
    done: bool = False
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def append_event(self, event_json: str) -> None:
        """Append an event and wake any waiting WebSocket reader."""
        async with self._condition:
            self.events.append(event_json)
            self._condition.notify_all()

    async def mark_done(self) -> None:
        """Signal that no more events will be added."""
        async with self._condition:
            self.done = True
            self._condition.notify_all()

    async def stream_from(self, start_index: int):
        """Async-iterate events starting at *start_index*, waiting for new ones."""
        idx = start_index
        while True:
            async with self._condition:
                while idx >= len(self.events) and not self.done:
                    await self._condition.wait()
                while idx < len(self.events):
                    yield self.events[idx]
                    idx += 1
                if self.done:
                    return


# One slot per conversation (single-user: effectively one global slot).
_active: dict[str, ConvTask] = {}


def get(conversation_id: str) -> ConvTask | None:
    """Return the active task for a conversation, or None."""
    task = _active.get(conversation_id)
    if task is not None and task.done:
        # Keep finished tasks available for reconnecting clients briefly,
        # but clean up after they've been read once.
        pass
    return task


def _clear(conversation_id: str) -> None:
    _active.pop(conversation_id, None)


async def run(conversation_id: str, user_content: str) -> ConvTask:
    """Start a background RAG task and return its ConvTask.

    Uses a dedicated database session independent of any WebSocket connection
    so the task survives client disconnects.
    """
    task = ConvTask()
    _active[conversation_id] = task

    async def _worker() -> None:
        try:
            async with AsyncSessionLocal() as db:
                async for event_json in service.process_message_stream(
                    db, conversation_id, user_content
                ):
                    await task.append_event(event_json)
        except Exception as exc:
            logger.exception("Background RAG task failed for conversation %s", conversation_id)
            await task.append_event(json.dumps({"type": "error", "text": str(exc)}))
        finally:
            done_json = json.dumps({"type": "done"})
            await task.append_event(done_json)
            await task.mark_done()
            # Keep the finished task in memory briefly so a reconnecting client
            # can read it, then remove it.
            await asyncio.sleep(30)
            _clear(conversation_id)

    asyncio.create_task(_worker())
    return task
