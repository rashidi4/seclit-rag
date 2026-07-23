"""Reciprocal Rank Fusion of dense and sparse result lists.

RRF scores each document as ``sum(1 / (k + rank))`` over the lists it appears
in. It is used here rather than a weighted blend of raw scores because cosine
similarities and BM25 scores are not comparable quantities: BM25 is unbounded
and corpus-dependent, cosine sits in [-1, 1]. Normalising them onto a shared
scale requires tuning constants that drift as the corpus grows. Ranks are
already scale-free, so RRF needs no tuning and cannot be destabilised by one
retriever producing unusually large scores.

The constant ``k`` (default 60, the value from the original RRF paper) damps
the influence of top ranks so a single retriever cannot dominate the fused
ordering on its own.
"""

from __future__ import annotations

from seclit.config import Settings, settings
from seclit.models import RetrievedChunk


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    config: Settings | None = None,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one, deduplicated by chunk ID."""
    config = config or settings
    k = config.rrf_k

    merged: dict[str, RetrievedChunk] = {}

    for chunk in dense:
        merged[chunk.chunk_id] = chunk
        chunk.fused_score = 1.0 / (k + (chunk.dense_rank or len(dense) + 1))

    for chunk in sparse:
        existing = merged.get(chunk.chunk_id)
        contribution = 1.0 / (k + (chunk.sparse_rank or len(sparse) + 1))
        if existing is None:
            chunk.fused_score = contribution
            merged[chunk.chunk_id] = chunk
        else:
            # Found by both retrievers — the strongest signal available.
            existing.sparse_rank = chunk.sparse_rank
            existing.fused_score += contribution

    return sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)
