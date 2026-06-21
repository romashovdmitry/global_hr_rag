"""Reciprocal Rank Fusion algorithm for merging multiple ranked lists."""

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Merge multiple ranked lists of document IDs using RRF.

    Args:
        rankings: Each inner list is a ranked list of document IDs,
                  ordered from most to least relevant.
        k: Smoothing constant. Higher values reduce the impact of top ranks.

    Returns:
        Merged list of document IDs sorted by descending RRF score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)
