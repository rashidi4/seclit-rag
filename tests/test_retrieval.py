"""Retrieval tests.

Fusion and tokenization are tested without models: both are pure functions, and
keeping them model-free means these run in milliseconds on every edit. Tests
that genuinely need the index are marked ``slow``.
"""

from __future__ import annotations

import pytest

from seclit.config import Settings
from seclit.models import RetrievedChunk
from seclit.retrieve.fuse import reciprocal_rank_fusion
from seclit.retrieve.sparse import tokenize


def chunk(chunk_id: str, *, dense: int | None = None, sparse: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=chunk_id.split("::")[0],
        text=f"text for {chunk_id}",
        page_start=1,
        page_end=1,
        section="",
        title=f"Paper {chunk_id}",
        authors="Author A; Author B",
        year=2024,
        dense_rank=dense,
        sparse_rank=sparse,
    )


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Zero Trust Architecture") == ["zero", "trust", "architecture"]

    def test_identifier_is_kept_whole_and_split(self):
        """A CVE query must match whether the user types the full ID or part."""
        tokens = tokenize("CVE-2021-44228")

        assert "cve-2021-44228" in tokens
        assert "cve" in tokens
        assert "44228" in tokens

    def test_dotted_versions_survive(self):
        tokens = tokenize("TLS 1.3 handshake")

        assert "1.3" in tokens
        assert "tls" in tokens

    def test_stopwords_removed(self):
        assert "the" not in tokenize("the firewall and the router")
        assert "firewall" in tokenize("the firewall and the router")

    def test_empty_input(self):
        assert tokenize("") == []

    def test_punctuation_only(self):
        assert tokenize("!!! ... ???") == []


class TestReciprocalRankFusion:
    def test_chunk_found_by_both_outranks_either_alone(self):
        """The central property: agreement between retrievers is the strongest
        signal available, so it must win."""
        dense = [chunk("a::1", dense=1), chunk("b::1", dense=2)]
        sparse = [chunk("b::1", sparse=1), chunk("c::1", sparse=2)]

        fused = reciprocal_rank_fusion(dense, sparse)

        assert fused[0].chunk_id == "b::1"
        assert fused[0].dense_rank == 2
        assert fused[0].sparse_rank == 1

    def test_deduplicates_by_chunk_id(self):
        dense = [chunk("a::1", dense=1)]
        sparse = [chunk("a::1", sparse=1)]

        fused = reciprocal_rank_fusion(dense, sparse)

        assert len(fused) == 1

    def test_scores_descend(self):
        dense = [chunk(f"d{i}::1", dense=i) for i in range(1, 6)]
        sparse = [chunk(f"s{i}::1", sparse=i) for i in range(1, 6)]

        fused = reciprocal_rank_fusion(dense, sparse)
        scores = [c.fused_score for c in fused]

        assert scores == sorted(scores, reverse=True)

    def test_handles_one_empty_list(self):
        dense = [chunk("a::1", dense=1), chunk("b::1", dense=2)]

        assert len(reciprocal_rank_fusion(dense, [])) == 2
        assert len(reciprocal_rank_fusion([], dense)) == 2

    def test_handles_both_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_rank_order_is_preserved_within_a_single_retriever(self):
        dense = [chunk(f"d{i}::1", dense=i) for i in range(1, 4)]

        fused = reciprocal_rank_fusion(dense, [])

        assert [c.chunk_id for c in fused] == ["d1::1", "d2::1", "d3::1"]

    def test_rrf_k_damps_top_rank_dominance(self):
        """A larger k flattens the curve, so rank 1 dominates less.

        Fresh chunks per call: fusion writes ``fused_score`` onto the objects it
        is given, so reusing one list across both calls would leave both result
        sets aliasing the same mutated objects.
        """

        def spread(config: Settings) -> float:
            fused = reciprocal_rank_fusion(
                [chunk("a::1", dense=1)],
                [chunk("b::1", sparse=2), chunk("c::1", sparse=3)],
                config,
            )
            return fused[0].fused_score - fused[-1].fused_score

        assert spread(Settings(rrf_k=1)) > spread(Settings(rrf_k=1000))


class TestRetrievedChunk:
    def test_single_page_label(self):
        assert chunk("a::1").page_label == "p. 1"

    def test_page_range_label(self):
        c = chunk("a::1")
        c.page_end = 4
        assert c.page_label == "pp. 1–4"

    def test_citation_includes_author_and_year(self):
        citation = chunk("a::1").citation
        assert "Author A" in citation
        assert "2024" in citation

    def test_from_chroma_tolerates_missing_metadata(self):
        c = RetrievedChunk.from_chroma("x::1", "body", {})

        assert c.chunk_id == "x::1"
        assert c.year == 0
        assert c.title == "Untitled"


@pytest.mark.slow
class TestAgainstRealIndex:
    """Exercises the built index. Skipped automatically when it is absent."""

    @pytest.fixture(autouse=True)
    def _require_index(self):
        from seclit.ingest.store import ChunkStore

        store = ChunkStore()
        if store.count() == 0:
            pytest.skip("no index built")
        self.store = store

    def test_sparse_finds_rare_literal(self):
        from seclit.retrieve.sparse import search_sparse

        results = search_sparse("Kubernetes", self.store)
        assert results, "BM25 should match a literal term present in the corpus"

    def test_modes_return_results(self):
        from seclit.retrieve import Retriever

        retriever = Retriever(self.store)
        for mode in ("dense", "sparse", "hybrid"):
            assert retriever.search("intrusion detection", mode=mode, top_k=5)
