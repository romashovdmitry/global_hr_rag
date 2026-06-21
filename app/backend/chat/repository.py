"""Data-access layer for chat sessions and messages."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.chat.models.message import Message
from backend.chat.models.session import Session


async def create_session(db: AsyncSession, cv_text: str | None = None) -> Session:
    """Persist a new chat session and return it.

    Args:
        db: Active async database session.
        cv_text: Optional extracted CV text to attach to the session.

    Returns:
        The newly created Session ORM instance.
    """
    session = Session(cv_text=cv_text)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    """Fetch a session by its ID, eagerly loading messages.

    Args:
        db: Active async database session.
        session_id: UUID string of the target session.

    Returns:
        Session ORM instance or None if not found.
    """
    result = await db.execute(
        select(Session)
        .where(Session.id == session_id)
        .options(selectinload(Session.messages))
    )
    return result.scalar_one_or_none()


async def delete_session(db: AsyncSession, session_id: str) -> bool:
    """Delete a session and all its messages (cascade).

    Args:
        db: Active async database session.
        session_id: UUID string of the target session.

    Returns:
        True if the session existed and was deleted, False otherwise.
    """
    session = await get_session(db, session_id)
    if session is None:
        return False
    await db.delete(session)
    await db.commit()
    return True


async def add_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
) -> Message:
    """Append a message to an existing session.

    Args:
        db: Active async database session.
        session_id: Target session UUID.
        role: Either ``"user"`` or ``"assistant"``.
        content: Message text content.

    Returns:
        The persisted Message ORM instance.
    """
    message = Message(session_id=session_id, role=role, content=content)
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def update_summary(
    db: AsyncSession,
    session_id: str,
    new_summary: str,
    summarized_through: int,
) -> None:
    """Persist a new rolling summary and the message index it covers.

    Args:
        db: Active async database session.
        session_id: Target session UUID.
        new_summary: Updated summary text produced by the LLM.
        summarized_through: Number of messages (from index 0) now covered by the summary.
    """
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if session is not None:
        session.summary = new_summary
        session.summarized_through = summarized_through
        await db.commit()


async def update_session_cv(
    db: AsyncSession,
    session_id: str,
    cv_text: str,
) -> Session | None:
    """Attach or replace CV text on an existing session.

    Args:
        db: Active async database session.
        session_id: Target session UUID.
        cv_text: Extracted plain text from the uploaded CV.

    Returns:
        Updated Session or None if the session was not found.
    """
    session = await get_session(db, session_id)
    if session is None:
        return None
    session.cv_text = cv_text
    await db.commit()
    await db.refresh(session)
    return session
