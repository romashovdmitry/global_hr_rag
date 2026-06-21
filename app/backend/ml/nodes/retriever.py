"""LangGraph node: hybrid Qdrant search + RRF merge."""

import asyncio
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range, SparseVector

from backend.core.config import settings
from backend.ml.state import RAGState
from backend.utils.qdrant import embed_dense, embed_sparse_async, get_qdrant_client
from backend.utils.rrf import reciprocal_rank_fusion
from backend.ml.timing import timed_node


def _build_filter(metadata_filters: dict[str, Any]) -> Filter | None:
    """Translate metadata_filters dict into a Qdrant Filter object.

    Args:
        metadata_filters: Dict with optional keys ``status_of_role``,
            ``salary_min``, ``salary_max``.

    Returns:
        Qdrant Filter or None if no filters were specified.
    """
    conditions = []

    if statuses := metadata_filters.get("status_of_role"):
        conditions.append(
            FieldCondition(
                key="status_of_role",
                match=MatchAny(any=statuses),
            )
        )

    salary_range: dict[str, int] = {}
    if (salary_min := metadata_filters.get("salary_min")) is not None:
        salary_range["gte"] = salary_min
    if (salary_max := metadata_filters.get("salary_max")) is not None:
        salary_range["lte"] = salary_max
    if salary_range:
        conditions.append(
            FieldCondition(key="salary", range=Range(**salary_range))
        )

    if not conditions:
        return None
    return Filter(must=conditions)


async def _search_single(
    client: AsyncQdrantClient,
    query: str,
    qdrant_filter: Filter | None,
    top_k: int,
) -> list[str]:
    """Run dense + sparse hybrid search for one query and return merged point IDs.

    Args:
        client: Active async Qdrant client.
        query: Single search string.
        qdrant_filter: Optional hard-filter to apply.
        top_k: Number of results to retrieve per vector type.

    Returns:
        List of point ID strings ranked by RRF score.
    """
    dense_vec = (await embed_dense([query]))[0]
    sparse_vec = (await embed_sparse_async([query]))[0]

    dense_response, sparse_response = await asyncio.gather(
        client.query_points(
            collection_name=settings.qdrant_collection,
            query=dense_vec,
            using="dense",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ),
        client.query_points(
            collection_name=settings.qdrant_collection,
            query=SparseVector(
                indices=sparse_vec["indices"],
                values=sparse_vec["values"],
            ),
            using="sparse",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ),
    )

    dense_results = dense_response.points
    sparse_results = sparse_response.points

    dense_ids = [str(hit.id) for hit in dense_results]
    sparse_ids = [str(hit.id) for hit in sparse_results]
    merged_ids = reciprocal_rank_fusion([dense_ids, sparse_ids], k=settings.rrf_k)

    all_payloads = {
        str(hit.id): hit.payload
        for hit in dense_results + sparse_results
        if hit.payload
    }
    return merged_ids, all_payloads


@timed_node("retriever")
async def retriever_node(state: RAGState) -> dict:
    """Execute hybrid search for all translated queries and merge with RRF.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``retrieved_vacancies`` and a status entry.
    """
    queries = state["search_queries"]
    qdrant_filter = _build_filter(state.get("metadata_filters", {}))
    # Aggregate intent needs a wider net to synthesise across many vacancies.
    top_k = 20 if state.get("intent") == "aggregate" else settings.retrieval_top_k

    client = get_qdrant_client()
    try:
        per_query = await asyncio.gather(
            *[_search_single(client, q, qdrant_filter, top_k) for q in queries]
        )
    finally:
        await client.close()

    all_rankings: list[list[str]] = []
    merged_payloads: dict[str, Any] = {}
    for ids, payloads in per_query:
        all_rankings.append(ids)
        merged_payloads.update(payloads)

    fused_ids = reciprocal_rank_fusion(all_rankings, k=settings.rrf_k)

    retrieved = [
        {"id": doc_id, **merged_payloads[doc_id]}
        for doc_id in fused_ids
        if doc_id in merged_payloads
    ]

    if not retrieved:
        status_msg = "No vacancies matched the search criteria."
    else:
        status_msg = f"Retrieved {len(retrieved)} candidate vacancies…"

    return {
        "retrieved_vacancies": retrieved,
        "status_stream": state.get("status_stream", []) + [status_msg],
    }
