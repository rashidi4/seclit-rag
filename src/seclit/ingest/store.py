"""Persistence: Chroma for vectors, SQLite for the ingest ledger.

Two stores rather than one, because they answer different questions:

* **Chroma** answers "which chunks are similar to this query".
* **SQLite catalog** answers "have I already ingested this paper, and which
  chunk IDs belong to it".

The catalog is what makes incremental ingest work. Without a ledger keyed by
content hash, the only way to know whether a paper is already indexed is to
scan the vector store, and the only safe way to add a paper is to rebuild. The
brief explicitly requires adding papers *without* rebuilding, so the ledger is
load-bearing, not bookkeeping.

Embeddings are computed by this application and passed to Chroma explicitly.
Chroma's built-in embedding functions are bypassed so the model choice stays
visible and swappable in one place.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from seclit.config import Settings, settings
from seclit.models import Chunk, PaperMeta

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id     TEXT PRIMARY KEY,
    sha256       TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    authors      TEXT NOT NULL DEFAULT '',
    year         INTEGER,
    subtopic     TEXT,
    pdf_url      TEXT,
    source_path  TEXT,
    n_pages      INTEGER NOT NULL DEFAULT 0,
    n_chunks     INTEGER NOT NULL DEFAULT 0,
    ingested_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_sha ON papers(sha256);
CREATE INDEX IF NOT EXISTS idx_papers_subtopic ON papers(subtopic);
"""


@dataclass
class PaperRecord:
    paper_id: str
    sha256: str
    title: str
    n_chunks: int
    subtopic: str | None = None


class Catalog:
    """SQLite ledger of ingested papers."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def has_hash(self, sha256: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM papers WHERE sha256 = ?", (sha256,)).fetchone()
        return row is not None

    def get(self, paper_id: str) -> PaperRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT paper_id, sha256, title, n_chunks, subtopic FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        return PaperRecord(**dict(row)) if row else None

    def record(
        self,
        meta: PaperMeta,
        *,
        source_path: str,
        n_pages: int,
        n_chunks: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (paper_id, sha256, title, authors, year, subtopic,
                                    pdf_url, source_path, n_pages, n_chunks, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    sha256=excluded.sha256, title=excluded.title, authors=excluded.authors,
                    year=excluded.year, subtopic=excluded.subtopic, pdf_url=excluded.pdf_url,
                    source_path=excluded.source_path, n_pages=excluded.n_pages,
                    n_chunks=excluded.n_chunks, ingested_at=excluded.ingested_at
                """,
                (
                    meta.paper_id,
                    meta.sha256,
                    meta.title,
                    "; ".join(meta.authors),
                    meta.year,
                    meta.subtopic,
                    meta.pdf_url,
                    source_path,
                    n_pages,
                    n_chunks,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def remove(self, paper_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))

    def all_papers(self) -> list[PaperRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT paper_id, sha256, title, n_chunks, subtopic FROM papers ORDER BY paper_id"
            ).fetchall()
        return [PaperRecord(**dict(r)) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            totals = conn.execute(
                "SELECT COUNT(*) AS papers, COALESCE(SUM(n_chunks), 0) AS chunks,"
                " COALESCE(SUM(n_pages), 0) AS pages FROM papers"
            ).fetchone()
            by_topic = conn.execute(
                "SELECT COALESCE(subtopic, 'unclassified') AS subtopic, COUNT(*) AS n"
                " FROM papers GROUP BY subtopic ORDER BY n DESC"
            ).fetchall()
        return {
            "papers": totals["papers"],
            "chunks": totals["chunks"],
            "pages": totals["pages"],
            "by_subtopic": {r["subtopic"]: r["n"] for r in by_topic},
        }


class ChunkStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, config: Settings | None = None) -> None:
        import chromadb

        self.config = config or settings
        self.config.ensure_dirs()
        self._client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def add(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        meta: PaperMeta,
    ) -> None:
        """Upsert chunks. Idempotent — re-adding the same IDs overwrites in place
        rather than duplicating, so a re-run cannot inflate the index."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunk/embedding length mismatch: {len(chunks)} vs {len(embeddings)}")

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[c.to_chroma_metadata(meta) for c in chunks],
        )

    def delete_paper(self, paper_id: str) -> None:
        self._collection.delete(where={"paper_id": paper_id})

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 30,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, max(1, self.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def all_chunks(self) -> dict[str, Any]:
        """Fetch every chunk — used to build the in-memory BM25 index."""
        if self.count() == 0:
            return {"ids": [], "documents": [], "metadatas": []}
        return self._collection.get(include=["documents", "metadatas"])

    def chunk_ids(self) -> set[str]:
        if self.count() == 0:
            return set()
        return set(self._collection.get(include=[])["ids"])
