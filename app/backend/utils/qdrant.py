"""Qdrant collection bootstrap and shared embedding helpers.

All embeddings are computed locally via fastembed — no external API keys required.
Dense model: BAAI/bge-small-en-v1.5 (384 dims)
Sparse model: Qdrant/bm25
"""

import asyncio
import hashlib
import uuid
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from backend.core.config import settings

_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None


def get_dense_model() -> TextEmbedding:
    """Return a lazily-initialised fastembed dense embedding model (singleton)."""
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding(model_name=settings.dense_embedding_model)
    return _dense_model


def get_sparse_model() -> SparseTextEmbedding:
    """Return a lazily-initialised BM25 sparse embedding model (singleton)."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=settings.sparse_embedding_model)
    return _sparse_model


def get_qdrant_client() -> AsyncQdrantClient:
    """Create a new async Qdrant client from settings."""
    return AsyncQdrantClient(url=settings.qdrant_url)


async def ensure_collection(client: AsyncQdrantClient) -> None:
    """Create the vacancies collection with hybrid vector config if it does not exist.

    Idempotent: safe to call on every application startup.

    Args:
        client: An active async Qdrant client.
    """
    exists = await client.collection_exists(settings.qdrant_collection)
    if exists:
        return

    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": VectorParams(
                size=settings.dense_vector_size,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(on_disk=False),
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )

    await client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="status_of_role",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="salary",
        field_schema=PayloadSchemaType.INTEGER,
    )


def _embed_dense_sync(texts: list[str]) -> list[list[float]]:
    """Compute dense embeddings synchronously using fastembed.

    Args:
        texts: Input texts to embed.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    model = get_dense_model()
    return [emb.tolist() for emb in model.embed(texts)]


def _embed_sparse_sync(texts: list[str]) -> list[dict[str, Any]]:
    """Compute sparse BM25 embeddings synchronously using fastembed.

    Args:
        texts: Input texts to embed.

    Returns:
        List of dicts with ``indices`` and ``values`` keys.
    """
    model = get_sparse_model()
    return [
        {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
        for emb in model.embed(texts)
    ]


async def embed_dense(texts: list[str]) -> list[list[float]]:
    """Async wrapper: compute dense embeddings via fastembed (runs in thread pool).

    Args:
        texts: Input texts to embed.

    Returns:
        List of embedding vectors.
    """
    return await asyncio.to_thread(_embed_dense_sync, texts)


async def embed_sparse_async(texts: list[str]) -> list[dict[str, Any]]:
    """Async wrapper: compute sparse BM25 embeddings via fastembed (runs in thread pool).

    Args:
        texts: Input texts to embed.

    Returns:
        List of dicts with ``indices`` and ``values`` keys.
    """
    return await asyncio.to_thread(_embed_sparse_sync, texts)


def vacancy_id(vacancy: dict[str, Any]) -> str:
    """Derive a deterministic UUID from a vacancy's role and company fields.

    This ensures idempotent re-ingestion: the same vacancy always maps to the
    same Qdrant point ID so it will be upserted rather than duplicated.

    Args:
        vacancy: Raw vacancy dict from vacancies.json.

    Returns:
        UUID string.
    """
    key = f"{vacancy.get('role', '')}|{vacancy.get('company', '')}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return str(uuid.UUID(digest[:32]))
