"""Vacancy ingestion service — validates, embeds, and upserts into Qdrant."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from backend.core.config import settings
from backend.utils.qdrant import embed_dense, embed_sparse_async, vacancy_id
from backend.vacancies.schemas import IngestResponse, VacancyCreate

logger = logging.getLogger(__name__)

BATCH_SIZE = 50


async def ingest_from_file(
    client: AsyncQdrantClient,
    path: str | None = None,
) -> IngestResponse:
    """Load vacancies.json and upsert valid records into Qdrant.

    Idempotent: existing points are overwritten via upsert using deterministic IDs.
    Invalid records (validation errors or empty descriptions) are counted as errors.

    Args:
        client: Active async Qdrant client.
        path: Optional override path to the JSON file; defaults to settings value.

    Returns:
        Ingestion summary with counts of ingested, skipped, and errored records.
    """
    file_path = Path(path or settings.vacancies_json_path)
    raw_records: list[dict[str, Any]] = json.loads(file_path.read_text(encoding="utf-8"))

    valid: list[tuple[str, VacancyCreate]] = []
    errors: list[str] = []

    for idx, record in enumerate(raw_records):
        try:
            vacancy = VacancyCreate.model_validate(record)
            point_id = vacancy_id(record)
            valid.append((point_id, vacancy))
        except (ValidationError, ValueError) as exc:
            role = record.get("role", f"<record #{idx}>")
            errors.append(f"{role}: {exc}")

    ingested = 0
    for batch_start in range(0, len(valid), BATCH_SIZE):
        batch = valid[batch_start : batch_start + BATCH_SIZE]
        texts = [
            f"{v.role}\n{v.description_of_vacancy}" for _, v in batch
        ]

        dense_vecs = await embed_dense(texts)
        sparse_vecs = await embed_sparse_async(texts)

        points = [
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"],
                    ),
                },
                payload={
                    "role": v.role,
                    "description_of_vacancy": v.description_of_vacancy,
                    "salary": v.salary,
                    "company": v.company,
                    "contacts": v.contacts,
                    "status_of_role": [s.value for s in v.status_of_role],
                },
            )
            for (point_id, v), dense_vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs)
        ]

        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        ingested += len(points)
        logger.info("Ingested batch %d–%d", batch_start + 1, batch_start + len(batch))

    return IngestResponse(
        total=len(raw_records),
        ingested=ingested,
        skipped=len(raw_records) - len(valid) - len(errors),
        errors=errors,
    )


async def collection_is_empty(client: AsyncQdrantClient) -> bool:
    """Return True if the Qdrant collection has no points.

    Args:
        client: Active async Qdrant client.
    """
    info = await client.get_collection(settings.qdrant_collection)
    return (info.points_count or 0) == 0
