"""Chat orchestration service: runs the RAG pipeline and persists messages."""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.chat import repository
from backend.chat.models.message import Message
from backend.ml.graph import rag_graph
from backend.ml.state import RAGState
from backend.stats import repository as stats_repo
from backend.stats.models import QueryStat
from backend.utils.summarization import maybe_summarize

logger = logging.getLogger(__name__)


async def get_history(db: AsyncSession, session_id: str) -> list[Message]:
    """Return all messages for a session, ordered by creation time.

    Args:
        db: Active async database session.
        session_id: Target session UUID.

    Returns:
        List of Message ORM instances, or empty list if session not found.
    """
    session = await repository.get_session(db, session_id)
    if session is None:
        return []
    return list(session.messages)


async def process_message_stream(
    db: AsyncSession,
    session_id: str,
    user_content: str,
) -> AsyncGenerator[str, None]:
    """Run the RAG pipeline for a user message and stream progress + final answer.

    Persists both the user message and the final assistant answer to the database.
    Yields JSON-encoded events so the WebSocket handler can forward them verbatim.

    Event types:
        ``{"type": "status", "text": "..."}``   — intermediate progress update
        ``{"type": "answer", "text": "..."}``   — complete final answer
        ``{"type": "error",  "text": "..."}``   — unrecoverable error

    Args:
        db: Active async database session.
        session_id: Target session UUID.
        user_content: Raw user message text.

    Yields:
        JSON-encoded event strings.
    """
    session = await repository.get_session(db, session_id)
    if session is None:
        yield json.dumps({"type": "error", "text": "Session not found."})
        return

    await repository.add_message(db, session_id, "user", user_content)

    # Reload session so messages list includes the just-saved user message.
    session = await repository.get_session(db, session_id)

    # Build context: rolling summary (if any) + unsummarized recent messages.
    history: list[dict[str, Any]] = []
    if session.summary:
        history.append({
            "role": "system",
            "content": f"[Summary of earlier conversation]:\n{session.summary}",
        })
    history.extend(
        {"role": m.role, "content": m.content}
        for m in session.messages[session.summarized_through:]
    )

    pipeline_start = time.perf_counter()

    initial_state: RAGState = {
        "session_id": session_id,
        "user_query": user_content,
        "cv_text": session.cv_text,
        "chat_history": history,
        "intent": "find_matching",
        "current_role": None,
        "exclude_skills": [],
        "metadata_filters": {},
        "unsupported_criteria": [],
        "search_queries": [],
        "retrieved_vacancies": [],
        "graded_vacancies": [],
        "generated_answer": "",
        "is_grounded": False,
        "grounding_feedback": None,
        "rejected_answer": None,
        "answer_relevant": True,
        "retry_count": 0,
        "status_stream": [],
        "node_timings": {},
        "input_tokens": None,
        "output_tokens": None,
    }

    emitted_statuses: set[str] = set()
    final_answer = ""
    final_state: dict[str, Any] = {}

    try:
        async for event in rag_graph.astream(initial_state, stream_mode="updates"):
            for node_output in event.values():
                for status in node_output.get("status_stream", []):
                    if status not in emitted_statuses:
                        emitted_statuses.add(status)
                        yield json.dumps({"type": "status", "text": status})

                if "generated_answer" in node_output:
                    final_answer = node_output["generated_answer"]

                # Accumulate the last state values for stats collection.
                final_state.update(node_output)

        if final_answer:
            await repository.add_message(db, session_id, "assistant", final_answer)
            yield json.dumps({"type": "answer", "text": final_answer})
        else:
            fallback = "Unable to generate a response. Please try again."
            await repository.add_message(db, session_id, "assistant", fallback)
            yield json.dumps({"type": "answer", "text": fallback})

        # Persist per-query stats asynchronously (best-effort, never raises).
        try:
            total_ms = int((time.perf_counter() - pipeline_start) * 1000)
            stat = QueryStat(
                user_query=user_content,
                intent=final_state.get("intent", "find_matching"),
                total_latency_ms=total_ms,
                node_timings_ms=final_state.get("node_timings") or {},
                retrieved_count=len(final_state.get("retrieved_vacancies") or []),
                graded_count=len(final_state.get("graded_vacancies") or []),
                is_grounded=bool(final_state.get("is_grounded", True)),
                was_fallback=not bool(final_answer),
                was_off_topic=final_state.get("intent") == "off_topic",
                retry_count=int(final_state.get("retry_count") or 0),
                input_tokens=final_state.get("input_tokens"),
                output_tokens=final_state.get("output_tokens"),
            )
            await stats_repo.save(db, stat)
        except Exception:
            logger.warning("Failed to persist query stat — ignoring.", exc_info=True)

        # After each complete exchange, check whether old messages should be
        # compressed into the rolling summary.
        fresh_session = await repository.get_session(db, session_id)
        if fresh_session is not None:
            await maybe_summarize(db, fresh_session)

    except Exception as exc:
        logger.exception("RAG pipeline error for session %s", session_id)
        yield json.dumps({"type": "error", "text": f"Pipeline error: {exc}"})
