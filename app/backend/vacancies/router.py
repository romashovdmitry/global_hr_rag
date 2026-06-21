"""Vacancy API endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from backend.utils.qdrant import ensure_collection, get_qdrant_client
from backend.vacancies.schemas import IngestResponse
from backend.vacancies.service import ingest_from_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vacancies", tags=["Vacancies"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest vacancies from the pre-built JSON dataset",
    description=(
        "Reads vacancies.json, validates every record against the VacancyCreate schema, "
        "generates dense (OpenAI) and sparse (BM25) embeddings, and upserts valid points "
        "into the Qdrant collection. Invalid records are reported in the response errors list. "
        "The operation is idempotent — re-running it overwrites existing points."
    ),
)
async def ingest_vacancies() -> IngestResponse:
    """Trigger a full re-ingestion of vacancies.json into Qdrant."""
    client = get_qdrant_client()
    try:
        await ensure_collection(client)
        result = await ingest_from_file(client)
        logger.info(
            "Ingestion complete: %d ingested, %d errors",
            result.ingested,
            len(result.errors),
        )
        return result
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    finally:
        await client.close()
