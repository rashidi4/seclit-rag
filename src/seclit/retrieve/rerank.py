"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

Bi-encoders (the embedding model) encode query and document *separately*, so
they can never model term-level interaction between them. A cross-encoder reads
the pair jointly and scores relevance directly. It is far too slow to run over a
whole corpus, and is used only as a final pass over a small fused pool.

Measured value is smaller than the technique's reputation suggests: on this
corpus's gold set it adds +0.010 MRR over hybrid fusion alone, at ~137x the
latency. It stays enabled by default because generation dominates a turn, but
`SECLIT_RETRIEVAL_MODE=hybrid` is a reasonable choice. See eval/results.md.

This stage is also where a **score floor** is applied. Retrieval always returns
*something* — ask an unrelated question and you still get the k least-unrelated
chunks. Passing those to the model invites it to synthesise an answer from
irrelevant context, which is a primary hallucination path. Dropping candidates
below the floor lets the pipeline return nothing and say so.
"""

from __future__ import annotations

import threading

from seclit.config import Settings, settings
from seclit.models import RetrievedChunk

_reranker_lock = threading.Lock()
_reranker_cache: dict[tuple[str, str], object] = {}


def get_reranker(config: Settings | None = None):
    """Load and cache the cross-encoder. Lazy — import must stay cheap."""
    config = config or settings
    device = config.resolve_device()
    key = (config.reranker_model, device)

    if key not in _reranker_cache:
        with _reranker_lock:
            if key not in _reranker_cache:
                from sentence_transformers import CrossEncoder

                # max_length matters a lot here. bge-reranker-v2-m3 defaults to
                # an 8192-token window, but relevance is decided well before
                # that: capping at 512 cut latency ~36% on this corpus with no
                # change to the top-ranked papers.
                _reranker_cache[key] = CrossEncoder(
                    config.reranker_model,
                    device=device,
                    max_length=config.rerank_max_length,
                )
    return _reranker_cache[key]


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    config: Settings | None = None,
    *,
    top_k: int | None = None,
    apply_floor: bool = True,
) -> list[RetrievedChunk]:
    """Rescore candidates against the query and return the strongest.

    Returns an empty list when nothing clears the relevance floor — that is a
    meaningful answer ("the corpus does not cover this"), not a failure.
    """
    config = config or settings
    if not candidates:
        return []

    pool = candidates[: config.rerank_candidates]
    model = get_reranker(config)
    scores = model.predict(
        [(query, chunk.text) for chunk in pool],
        show_progress_bar=False,
    )

    for chunk, score in zip(pool, scores, strict=False):
        chunk.rerank_score = float(score)

    ordered = sorted(pool, key=lambda c: c.rerank_score or float("-inf"), reverse=True)

    if apply_floor:
        ordered = [c for c in ordered if (c.rerank_score or 0.0) >= config.rerank_score_floor]

    return ordered[: top_k or config.final_top_k]
