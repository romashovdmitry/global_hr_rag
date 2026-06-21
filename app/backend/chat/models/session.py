"""SQLAlchemy model for a chat session."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class Session(Base):
    """Represents a single user chat session.

    A session holds optional CV text that scopes all RAG queries within it.
    Rolling summarization is used to keep the conversation context bounded:
    ``summary`` accumulates older exchanges; ``summarized_through`` tracks
    how many messages from the start of ``session.messages`` are already
    captured in the summary.
    """

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    cv_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    summarized_through: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    messages: Mapped[list["backend.chat.models.message.Message"]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
