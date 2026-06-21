"""Rolling summarization for chat session history.

When the number of unsummarized messages in a session exceeds
``SUMMARIZE_TRIGGER``, the oldest batch is compressed into ``session.summary``
via an LLM call.  The summary accumulates across multiple passes so the full
conversation context is always available without bloating the LLM prompt.

Design:
- All messages remain in the database (full audit trail).
- ``session.summarized_through`` tracks how many messages (from index 0) are
  already captured in the current summary.
- At runtime the prompt context is: [summary] + messages[summarized_through:].
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from backend.chat import repository
from backend.chat.models.session import Session
from backend.ml.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# Compress when there are more than this many unsummarized messages.
SUMMARIZE_TRIGGER = 20
# Always keep the most recent messages outside the summary for fresh context.
KEEP_RECENT = 10

_SYSTEM = (
    "You are a conversation summarizer for an IT job-search assistant. "
    "Produce a concise but complete summary (max 300 words) in third person that preserves: "
    "the candidate's stated preferences (role type, salary range, work format, tech stack), "
    "vacancies already discussed and the candidate's reactions, "
    "any decisions or follow-up questions raised. "
    "If a previous summary is provided, merge it with the new messages into a single updated summary."
)


async def maybe_summarize(db: AsyncSession, session: Session) -> None:
    """Compress old messages into a rolling summary when the threshold is reached.

    Called after each complete assistant reply.  Does nothing when the session
    history is short enough to fit in context as-is.

    Args:
        db: Active async database session (used to persist the new summary).
        session: Session ORM instance with ``messages`` already loaded.
    """
    messages = list(session.messages)
    total = len(messages)
    unsummarized_count = total - session.summarized_through

    if unsummarized_count <= SUMMARIZE_TRIGGER:
        return

    # Summarize everything except the most recent KEEP_RECENT messages.
    end_idx = total - KEEP_RECENT
    to_summarize = messages[session.summarized_through:end_idx]

    if not to_summarize:
        return

    convo_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in to_summarize
    )

    user_content_parts: list[str] = []
    if session.summary:
        user_content_parts.append(f"Previous summary:\n{session.summary}\n")
    user_content_parts.append(f"New messages to incorporate:\n{convo_text}")

    llm = get_llm_service().get_llm()

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ])
        new_summary: str = response.content
    except Exception:
        logger.exception(
            "Summarization failed for session %s — keeping previous summary", session.id
        )
        return

    await repository.update_summary(
        db,
        session_id=session.id,
        new_summary=new_summary,
        summarized_through=end_idx,
    )

    logger.info(
        "Session %s: summarized messages 0–%d (%d total)",
        session.id, end_idx, total,
    )
