"""Retrieval pipeline: dense + sparse -> RRF fusion -> cross-encoder rerank.

Each stage exists for a specific failure it prevents:

* **Dense** handles paraphrase — "how do I stop lateral movement" finding a
  chunk that never uses those words.
* **Sparse (BM25)** handles rare literals — CVE identifiers, protocol versions,
  tool names — which embeddings blur together.
* **RRF** merges the two without needing their incomparable scores normalised.
* **Rerank** applies term-level query/document interaction that neither
  retriever can, and enforces a relevance floor so an out-of-corpus question
  returns nothing instead of plausible-looking noise. On the gold set its
  ranking benefit is modest (+0.010 MRR over hybrid alone); the relevance
  floor is arguably the more valuable half of this stage.

``mode`` exists so the evaluation harness can measure each configuration and
show what the added machinery is worth. See ``eval/results.md``.
"""

from __future__ import annotations

from typing import Literal

from seclit.config import Settings, settings
from seclit.ingest.store import ChunkStore
from seclit.models import RetrievedChunk
from seclit.retrieve.dense import search_dense
from seclit.retrieve.fuse import reciprocal_rank_fusion
from seclit.retrieve.rerank import rerank
from seclit.retrieve.sparse import search_sparse

Mode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]

__all__ = ["Mode", "Retriever", "rerank", "search_dense", "search_sparse"]


class Retriever:
    def __init__(self, store: ChunkStore | None = None, config: Settings | None = None) -> None:
        self.config = config or settings
        self.store = store or ChunkStore(self.config)

    def search(
        self,
        query: str,
        *,
        mode: Mode | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.config.final_top_k
        mode = mode or self.config.retrieval_mode  # type: ignore[assignment]

        if mode == "dense":
            return search_dense(query, self.store, self.config)[:top_k]

        if mode == "sparse":
            return search_sparse(query, self.store, self.config)[:top_k]

        dense = search_dense(query, self.store, self.config)
        sparse = search_sparse(query, self.store, self.config)
        fused = reciprocal_rank_fusion(dense, sparse, self.config)

        if mode == "hybrid":
            return fused[:top_k]

        return rerank(query, fused, self.config, top_k=top_k)

    def with_markers(
        self,
        query: str,
        *,
        mode: Mode | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Search and assign per-turn citation markers (``c1``, ``c2``, ...).

        Markers are assigned here, at the boundary between retrieval and
        generation, so the exact set shown to the model is the exact set the
        citation validator checks against.
        """
        results = self.search(query, mode=mode, top_k=top_k)
        for position, chunk in enumerate(results, start=1):
            chunk.marker = f"c{position}"
        return results
