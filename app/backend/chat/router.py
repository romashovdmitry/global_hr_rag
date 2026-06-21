"""Chat API endpoints including WebSocket streaming.

Single-user mode: all endpoints operate on the singleton CONVERSATION_ID.
The concept of "sessions" is an internal implementation detail only.
"""

import json
import logging

from fastapi import (
    APIRouter,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

from backend.chat import service, background, repository as repo
from backend.chat.models.message import Message
from backend.chat.schemas import HistoryResponse, MessageRead
from backend.core.config import CONVERSATION_ID
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.get(
    "/messages",
    response_model=HistoryResponse,
    summary="Get conversation history",
    description="Returns all messages in the conversation, ordered by time.",
)
async def get_messages(db: AsyncSession = Depends(get_db)) -> HistoryResponse:
    """Return the full conversation history."""
    messages = await service.get_history(db, CONVERSATION_ID)
    return HistoryResponse(
        session_id=CONVERSATION_ID,
        messages=[MessageRead.model_validate(m) for m in messages],
    )


@router.delete(
    "/messages",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear conversation history",
    description="Deletes all messages. The conversation itself is preserved.",
)
async def clear_messages(db: AsyncSession = Depends(get_db)) -> None:
    """Delete all messages, keeping the singleton conversation row intact."""
    await db.execute(delete(Message).where(Message.session_id == CONVERSATION_ID))
    # Also reset rolling summary so the next query starts fresh.
    session = await repo.get_session(db, CONVERSATION_ID)
    if session is not None:
        session.summary = None
        session.summarized_through = 0
        session.cv_text = None
    await db.commit()


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time streaming of RAG responses.

    Processing runs as a background task independent of the WebSocket connection.
    If the client refreshes mid-processing, reconnecting will replay all buffered
    events and deliver the final answer once it is ready.

    Protocol:
        Client sends:  ``{"content": "user message text"}``
        Server sends:  ``{"type": "status"|"answer"|"error", "text": "..."}``
        Server sends:  ``{"type": "done"}`` after each complete response.
    """
    await websocket.accept()

    async def _stream_task(task: background.ConvTask, start: int = 0) -> None:
        """Forward all task events to this WebSocket, starting from *start*."""
        async for event_json in task.stream_from(start):
            await websocket.send_text(event_json)

    try:
        # On reconnect: only replay if a task is STILL RUNNING.
        # Completed tasks are already persisted in the DB and will be loaded
        # by the frontend's getMessages() call — replaying them here would
        # produce duplicate answer messages in the chat.
        ongoing = background.get(CONVERSATION_ID)
        if ongoing is not None and not ongoing.done:
            logger.info("Client reconnected — replaying %d buffered events.", len(ongoing.events))
            await websocket.send_text(
                json.dumps({"type": "status", "text": "Reconnected — resuming processing…"})
            )
            await _stream_task(ongoing, start=0)

        while True:
            data = await websocket.receive_json()
            user_content = data.get("content", "").strip()
            if not user_content:
                await websocket.send_json({"type": "error", "text": "Empty message."})
                continue

            # Start RAG in a background task; it will survive any future disconnect.
            task = await background.run(CONVERSATION_ID, user_content)
            await _stream_task(task, start=0)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — background task (if any) continues.")
    except Exception as exc:
        logger.exception("WebSocket error.")
        try:
            await websocket.send_text(json.dumps({"type": "error", "text": str(exc)}))
        except Exception:
            pass
