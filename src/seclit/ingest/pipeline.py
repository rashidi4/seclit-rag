"""Ingestion orchestration: PDF -> pages -> chunks -> vectors -> store.

Two properties this pipeline guarantees, both required by the brief:

**Idempotent.** Running it twice over the same corpus leaves the index
byte-identical. Deduplication is by SHA-256 of file content, so the same paper
under a different filename is ingested once, and chunk IDs are deterministic
(``{paper_id}::{index:04d}``) so re-ingesting overwrites in place instead of
appending duplicates.

**Incremental.** Adding a paper touches only that paper's chunks. Nothing is
rebuilt, and existing chunk IDs are never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from seclit.config import Settings, settings
from seclit.ingest.chunk import Chunker
from seclit.ingest.embed import embed_texts, token_counter
from seclit.ingest.extract import extract_document, sha256_file
from seclit.ingest.store import Catalog, ChunkStore
from seclit.models import PaperMeta

Status = Literal["ingested", "skipped", "failed"]


@dataclass
class IngestResult:
    paper_id: str
    status: Status
    n_chunks: int = 0
    n_pages: int = 0
    error: str = ""


@dataclass
class IngestSummary:
    results: list[IngestResult]

    @property
    def ingested(self) -> int:
        return sum(1 for r in self.results if r.status == "ingested")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def failed(self) -> list[IngestResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def total_chunks(self) -> int:
        return sum(r.n_chunks for r in self.results)

    def summary(self) -> str:
        lines = [
            f"ingested {self.ingested}, skipped {self.skipped}, failed {len(self.failed)}",
            f"chunks written: {self.total_chunks}",
        ]
        for result in self.failed[:10]:
            lines.append(f"  ! {result.paper_id}: {result.error}")
        return "\n".join(lines)


class IngestPipeline:
    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or settings
        self.config.ensure_dirs()
        self.store = ChunkStore(self.config)
        self.catalog = Catalog(self.config.catalog_path)
        self._chunker: Chunker | None = None

    @property
    def chunker(self) -> Chunker:
        """Built lazily so the real tokenizer is only loaded when ingesting."""
        if self._chunker is None:
            self._chunker = Chunker(
                target_tokens=self.config.chunk_target_tokens,
                overlap_ratio=self.config.chunk_overlap_ratio,
                min_tokens=self.config.chunk_min_tokens,
                token_counter=token_counter(self.config),
            )
        return self._chunker

    def ingest_pdf(
        self,
        pdf_path: Path | str,
        meta: PaperMeta | None = None,
        *,
        force: bool = False,
    ) -> IngestResult:
        pdf_path = Path(pdf_path)
        paper_id = meta.paper_id if meta else pdf_path.stem

        try:
            digest = sha256_file(pdf_path)

            if not force and self.catalog.has_hash(digest):
                return IngestResult(paper_id=paper_id, status="skipped")

            if meta is None:
                meta = PaperMeta(paper_id=paper_id, title=pdf_path.stem)
            meta.sha256 = digest

            doc = extract_document(pdf_path, meta)
            chunks = self.chunker.chunk(doc)
            if not chunks:
                return IngestResult(paper_id=paper_id, status="failed", error="no chunks produced")

            # Re-ingesting a changed file: clear stale chunks first, since the
            # new version may produce fewer chunks and upsert alone would leave
            # orphans behind.
            if force:
                self.store.delete_paper(paper_id)

            embeddings = embed_texts([c.text for c in chunks], self.config)
            self.store.add(chunks, embeddings, meta)
            self.catalog.record(
                meta,
                source_path=str(pdf_path),
                n_pages=doc.n_pages,
                n_chunks=len(chunks),
            )

            return IngestResult(
                paper_id=paper_id,
                status="ingested",
                n_chunks=len(chunks),
                n_pages=doc.n_pages,
            )

        except Exception as exc:  # noqa: BLE001 - one bad PDF must not halt a batch
            return IngestResult(paper_id=paper_id, status="failed", error=str(exc))

    def ingest_corpus(
        self,
        pdf_dir: Path | None = None,
        manifest: list[PaperMeta] | None = None,
        *,
        force: bool = False,
        verbose: bool = True,
    ) -> IngestSummary:
        """Ingest every PDF in a directory, enriched by manifest metadata."""
        pdf_dir = Path(pdf_dir or self.config.pdf_dir)
        by_id = {m.paper_id: m for m in (manifest or [])}

        pdfs = sorted(pdf_dir.glob("*.pdf"))
        results: list[IngestResult] = []

        for pos, pdf in enumerate(pdfs, start=1):
            meta = by_id.get(pdf.stem)
            result = self.ingest_pdf(pdf, meta, force=force)
            results.append(result)

            if verbose:
                mark = {"ingested": "+", "skipped": "=", "failed": "!"}[result.status]
                detail = (
                    f"{result.n_chunks:3d} chunks"
                    if result.status == "ingested"
                    else result.error[:60] or result.status
                )
                title = (meta.title if meta else pdf.stem)[:50]
                print(f"  {mark} [{pos:3d}/{len(pdfs)}] {pdf.stem:14s} {detail:16s} {title}")

        return IngestSummary(results=results)

    def stats(self) -> dict:
        catalog_stats = self.catalog.stats()
        catalog_stats["vectors"] = self.store.count()
        return catalog_stats
