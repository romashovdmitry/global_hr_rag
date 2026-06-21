"""Tests for the Reciprocal Rank Fusion algorithm."""

import pytest

from backend.utils.rrf import reciprocal_rank_fusion


class TestRRF:
    """Unit tests covering RRF correctness and edge cases."""

    def test_single_ranking_preserved(self):
        """A single input ranking is returned in its original order."""
        ids = ["a", "b", "c"]
        result = reciprocal_rank_fusion([ids])
        assert result == ids

    def test_top_shared_document_scores_higher(self):
        """A document appearing at the top of multiple lists scores highest."""
        r1 = ["shared", "unique1", "unique2"]
        r2 = ["shared", "unique3", "unique4"]
        result = reciprocal_rank_fusion([r1, r2])
        assert result[0] == "shared"

    def test_empty_rankings_return_empty(self):
        """An empty input produces an empty output."""
        assert reciprocal_rank_fusion([]) == []

    def test_disjoint_rankings_merge(self):
        """Documents from disjoint rankings are all included in output."""
        r1 = ["a", "b"]
        r2 = ["c", "d"]
        result = reciprocal_rank_fusion([r1, r2])
        assert set(result) == {"a", "b", "c", "d"}

    def test_custom_k_affects_scores(self):
        """Higher k reduces the dominance of top-ranked items (smoke test)."""
        r = ["a", "b", "c"]
        result_low_k = reciprocal_rank_fusion([r], k=1)
        result_high_k = reciprocal_rank_fusion([r], k=1000)
        # Order should still be preserved since it's a single ranking
        assert result_low_k[0] == "a"
        assert result_high_k[0] == "a"
