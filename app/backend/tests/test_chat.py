"""Tests for chat history endpoints and the chat repository (single-user mode).

The app runs in single-user mode: there is no per-request session creation.
All chat endpoints operate on the singleton ``CONVERSATION_ID``.
"""

import pytest

from backend.core.config import CONVERSATION_ID


class TestConversationHistory:
    """Session-free history endpoints operating on the singleton conversation."""

    @pytest.mark.asyncio
    async def test_get_history_empty(self, client):
        """GET /chat/messages returns an empty list when nothing is stored."""
        response = await client.get("/api/v1/chat/messages")
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == CONVERSATION_ID
        assert body["messages"] == []

    @pytest.mark.asyncio
    async def test_clear_history_is_idempotent(self, client):
        """DELETE /chat/messages returns 204 even with no messages stored."""
        response = await client.delete("/api/v1/chat/messages")
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_history_reflects_persisted_messages(self, client, db_session):
        """Messages persisted to the singleton conversation appear in history."""
        from backend.chat import repository
        from backend.chat.models.session import Session

        db_session.add(Session(id=CONVERSATION_ID))
        await db_session.commit()

        await repository.add_message(db_session, CONVERSATION_ID, "user", "Hi")
        await repository.add_message(db_session, CONVERSATION_ID, "assistant", "Hello!")

        response = await client.get("/api/v1/chat/messages")
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]


class TestMessageRepository:
    """Unit-level tests for the chat repository."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_messages(self, db_session):
        """Messages added to a session are retrievable in order."""
        from backend.chat import repository

        session = await repository.create_session(db_session)
        await repository.add_message(db_session, session.id, "user", "Hello!")
        await repository.add_message(db_session, session.id, "assistant", "Hi there!")

        fetched = await repository.get_session(db_session, session.id)
        assert len(fetched.messages) == 2
        assert fetched.messages[0].role == "user"
        assert fetched.messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_update_session_cv(self, db_session):
        """CV text can be attached to an existing session."""
        from backend.chat import repository

        session = await repository.create_session(db_session)
        updated = await repository.update_session_cv(db_session, session.id, "My CV text")
        assert updated is not None
        assert updated.cv_text == "My CV text"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session_returns_none(self, db_session):
        """Fetching a non-existent session returns None without raising."""
        from backend.chat import repository

        result = await repository.get_session(db_session, "does-not-exist")
        assert result is None
