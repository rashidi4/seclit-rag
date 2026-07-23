"""Dense vector retrieval over Chroma."""

from __future__ import annotations

from seclit.config import Settings, settings
from seclit.ingest.embed import embed_query
from seclit.ingest.store import ChunkStore
from seclit.models import RetrievedChunk


def search_dense(
    query: str,
    store: ChunkStore,
    config: Settings | None = None,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return the nearest chunks, rank-annotated.

    Ranks (not raw distances) are what feed fusion — see ``fuse.py`` for why.
    """
    config = config or settings
    if store.count() == 0:
        return []

    embedding = embed_query(query, config)
    raw = store.query(embedding, top_k=top_k or config.dense_top_k)

    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]

    results: list[RetrievedChunk] = []
    for rank, (chunk_id, document, metadata) in enumerate(
        zip(ids, documents, metadatas, strict=False), start=1
    ):
        chunk = RetrievedChunk.from_chroma(chunk_id, document, metadata or {})
        chunk.dense_rank = rank
        results.append(chunk)
    return results
