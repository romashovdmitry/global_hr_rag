"""FastAPI application entry point with startup lifespan."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel
from sqlalchemy import select

from backend.admin.router import router as admin_router
from backend.chat.models.session import Session
from backend.chat.router import router as chat_router
from backend.chat import repository
from backend.stats.router import router as stats_router
from backend.core.config import settings, CONVERSATION_ID
from backend.cv.router import router as cv_router
from backend.db.session import AsyncSessionLocal
from backend.utils.qdrant import ensure_collection, get_qdrant_client
from backend.vacancies.router import router as vacancies_router
from backend.vacancies.service import collection_is_empty, ingest_from_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: bootstrap Qdrant, ingest vacancies, and ensure singleton conversation."""
    client = get_qdrant_client()
    try:
        logger.info("Ensuring Qdrant collection '%s' exists…", settings.qdrant_collection)
        await ensure_collection(client)

        if await collection_is_empty(client):
            logger.info("Collection is empty — ingesting vacancies.json…")
            result = await ingest_from_file(client)
            logger.info(
                "Auto-ingestion complete: %d ingested, %d errors",
                result.ingested,
                len(result.errors),
            )
            if result.errors:
                for err in result.errors[:5]:
                    logger.warning("Ingestion error: %s", err)
        else:
            logger.info("Collection already populated — skipping auto-ingestion.")
    except Exception:
        logger.exception("Startup bootstrap failed — the app will still start.")
    finally:
        await client.close()

    # Ensure the singleton conversation row exists in PostgreSQL.
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Session).where(Session.id == CONVERSATION_ID)
            )
            if result.scalar_one_or_none() is None:
                db.add(Session(id=CONVERSATION_ID))
                await db.commit()
                logger.info("Singleton conversation created (id=%s).", CONVERSATION_ID)
            else:
                logger.info("Singleton conversation already exists.")
    except Exception:
        logger.exception("Failed to initialise singleton conversation.")

    yield


app = FastAPI(
    title="IT-Market Vacancy RAG System",
    description=(
        "RAG system for matching candidates to IT vacancies using hybrid "
        "dense+sparse search (Qdrant) and LangGraph orchestration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vacancies_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(cv_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check API liveness",
    description="Returns a simple status payload when the backend process is alive.",
)
async def health() -> HealthResponse:
    """Liveness probe endpoint."""
    return HealthResponse(status="ok")


# Mount the MCP server after all routes are registered so fastapi-mcp
# discovers every endpoint and exposes it as a callable tool in Cursor.
mcp = FastApiMCP(app)
mcp.mount_http()
