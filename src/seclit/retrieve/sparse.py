"""BM25 lexical retrieval.

Dense embeddings are strong on paraphrase and weak on rare literal strings.
Security literature is full of exactly those: ``CVE-2021-44228``, ``TLS 1.3``,
``SHA-256``, ``x86-64``, ``BGP``. A user asking about Log4Shell by CVE number
needs the chunk containing that exact identifier, and an embedding model will
happily return semantically "nearby" chunks about unrelated vulnerabilities
instead. BM25 covers that gap, which is why retrieval here is hybrid rather
than purely vector-based.

Tokenization emits compound identifiers **both whole and split**, so
``CVE-2021-44228`` indexes as ``cve-2021-44228`` plus ``cve``, ``2021``, and
``44228``. A query for either the full identifier or just ``CVE`` then matches.

The index is held in memory and rebuilt when the collection size changes, which
keeps it correct after incremental additions without a persistence format to
maintain. At corpus scale (a few thousand chunks) a rebuild is well under a
second.
"""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from seclit.config import Settings, settings
from seclit.ingest.store import ChunkStore
from seclit.models import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_SPLIT_RE = re.compile(r"[._-]")

# Deliberately minimal. BM25's IDF already discounts ubiquitous terms; an
# aggressive stopword list mainly risks discarding meaningful query words.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "then there these this to was were which with we our".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, extract identifiers, and emit compound parts alongside wholes."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        if _SPLIT_RE.search(token):
            tokens.extend(p for p in _SPLIT_RE.split(token) if p and p not in _STOPWORDS)
    return tokens


class SparseIndex:
    """In-memory BM25 index over every chunk in the store."""

    def __init__(self, store: ChunkStore) -> None:
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._built_at_count = -1
        self._lock = threading.Lock()

    def _ensure_built(self) -> None:
        count = self.store.count()
        if self._bm25 is not None and self._built_at_count == count:
            return

        with self._lock:
            if self._bm25 is not None and self._built_at_count == count:
                return

            payload = self.store.all_chunks()
            self._ids = list(payload.get("ids") or [])
            self._documents = list(payload.get("documents") or [])
            self._metadatas = list(payload.get("metadatas") or [])

            corpus = [tokenize(doc) for doc in self._documents]
            # BM25Okapi rejects an empty corpus, so leave the index unbuilt.
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._built_at_count = count

    def search(self, query: str, top_k: int = 30) -> list[RetrievedChunk]:
        self._ensure_built()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[RetrievedChunk] = []
        for rank, position in enumerate(ranked[:top_k], start=1):
            if scores[position] <= 0:
                break  # no lexical overlap at all
            chunk = RetrievedChunk.from_chroma(
                chunk_id=self._ids[position],
                document=self._documents[position],
                metadata=self._metadatas[position],
            )
            chunk.sparse_rank = rank
            results.append(chunk)
        return results


_index_cache: dict[int, SparseIndex] = {}


def get_sparse_index(store: ChunkStore, config: Settings | None = None) -> SparseIndex:
    """Cache one index per store instance so it is not rebuilt per query."""
    del config  # reserved for future per-config tuning
    key = id(store)
    if key not in _index_cache:
        _index_cache[key] = SparseIndex(store)
    return _index_cache[key]


def search_sparse(
    query: str,
    store: ChunkStore,
    config: Settings | None = None,
) -> list[RetrievedChunk]:
    config = config or settings
    return get_sparse_index(store, config).search(query, top_k=config.sparse_top_k)
