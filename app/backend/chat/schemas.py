"""Pydantic schemas for chat sessions and messages."""

from datetime import datetime

from pydantic import BaseModel


class SessionCreate(BaseModel):
    """Request body for creating a new chat session."""

    cv_text: str | None = None


class SessionRead(BaseModel):
    """Serialised chat session returned by the API."""

    id: str
    cv_text: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageRead(BaseModel):
    """Serialised chat message returned by the API."""

    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    """Request body for sending a new user message."""

    content: str


class HistoryResponse(BaseModel):
    """Paginated message history for a session."""

    session_id: str
    messages: list[MessageRead]
