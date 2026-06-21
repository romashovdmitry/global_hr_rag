"""Shared pytest fixtures for the backend test suite."""

import json
import os
import sys
import types as _types
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# Ragas 0.4.x imports langchain_community.chat_models.vertexai at the top
# level, but that submodule was removed in langchain-community 0.4.0.
# Inject a minimal stub so ragas imports succeed without Google Cloud deps.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = None  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

import pytest
import pytest_asyncio


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest marks to suppress unknown-mark warnings."""
    config.addinivalue_line(
        "markers",
        "eval: RAG metric evaluation tests — require live Qdrant + Ollama. "
        "Run with: RUN_EVAL_TESTS=1 pytest -m eval",
    )
    config.addinivalue_line(
        "markers",
        "behavioral: Fast behavioral pipeline tests — require live Qdrant + Ollama.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip eval-marked tests unless RUN_EVAL_TESTS=1 is set.

    This prevents eval tests from running (and failing on fixture setup)
    during normal ``pytest`` runs that only have mocked services.
    To run eval tests:
        RUN_EVAL_TESTS=1 pytest -m eval -v
    """
    if os.getenv("RUN_EVAL_TESTS") == "1":
        return
    skip_marker = pytest.mark.skip(
        reason="Live pipeline tests skipped — set RUN_EVAL_TESTS=1 to enable"
    )
    for item in items:
        if "eval" in item.keywords or "behavioral" in item.keywords:
            item.add_marker(skip_marker)
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.db.base import Base
from backend.db.session import get_db

# ---------------------------------------------------------------------------
# Use an in-memory SQLite database for tests — no real PostgreSQL required.
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create the in-memory test database engine and tables."""
    # Import all models so Base.metadata knows about every table before
    # create_all runs (otherwise per-test cleanup would target missing tables).
    import backend.chat.models  # noqa: F401
    import backend.stats.models  # noqa: F401

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a fresh database session with all tables cleaned at setup.

    Tests commit data (e.g. the singleton conversation uses a fixed id), so a
    plain rollback is not enough to isolate them — committed rows would leak
    across tests in the session-scoped in-memory database. We delete every row
    from all tables at setup (reusing the session's own connection to avoid the
    locking that a second connection to an in-memory SQLite DB would cause).
    """
    session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide a test HTTP client with database dependency overridden."""
    from backend.main import app

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def sample_vacancies() -> list[dict]:
    """Load the first 5 valid vacancies from vacancies.json as test data."""
    # In Docker the file is at /app/vacancies.json; on host it's 4 levels up
    vacancies_path = Path("/app/vacancies.json")
    if not vacancies_path.exists():
        vacancies_path = Path(__file__).parent.parent.parent.parent / "vacancies.json"
    all_vacancies = json.loads(vacancies_path.read_text())
    # Filter only valid records (with role and non-empty description)
    valid = [
        v for v in all_vacancies
        if v.get("role") and v.get("description_of_vacancy")
        and isinstance(v.get("status_of_role"), list)
    ]
    return valid[:5]


@pytest.fixture
def mock_qdrant_client():
    """Return a mocked async Qdrant client."""
    mock = AsyncMock()
    mock.collection_exists = AsyncMock(return_value=True)
    mock.get_collection = AsyncMock(return_value=MagicMock(points_count=100))
    mock.upsert = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_dense_embeddings():
    """Patch embed_dense to return zero vectors without running fastembed."""
    with patch(
        "backend.utils.qdrant.embed_dense",
        new_callable=AsyncMock,
        return_value=[[0.0] * 384],
    ) as m:
        yield m


@pytest.fixture
def mock_sparse_embeddings():
    """Patch embed_sparse_async to return trivial sparse vectors."""
    with patch(
        "backend.utils.qdrant.embed_sparse_async",
        new_callable=AsyncMock,
        return_value=[{"indices": [1, 2], "values": [0.5, 0.5]}],
    ) as m:
        yield m
