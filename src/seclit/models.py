"""Core data model.

The invariant that matters: a ``Chunk`` always knows which pages of which paper
it came from. Page provenance is established at extraction time and carried
through chunking, embedding, retrieval, and generation without ever being
reconstructed or guessed. That is what makes the citations trustworthy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    """One page of extracted text. ``number`` is 1-indexed to match what a
    reader sees in a PDF viewer."""

    number: int
    text: str


@dataclass
class PaperMeta:
    """Bibliographic record. Mirrors one line of ``corpus_manifest.jsonl``."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    subtopic: str | None = None
    abstract: str = ""
    pdf_url: str = ""
    sha256: str = ""

    @property
    def author_str(self) -> str:
        if not self.authors:
            return "Unknown"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al."

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> PaperMeta:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Document:
    """A paper after text extraction, before chunking."""

    meta: PaperMeta
    pages: list[Page]
    source_path: str = ""

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


@dataclass
class Chunk:
    """A retrievable unit of text with intact page provenance."""

    chunk_id: str
    paper_id: str
    index: int
    text: str
    page_start: int
    page_end: int
    section: str | None = None

    def to_chroma_metadata(self, meta: PaperMeta) -> dict[str, Any]:
        """Chroma metadata must be flat scalars, so lists get joined here.

        Everything the UI needs to render a citation lives in this dict, which
        means rendering a source never requires a second lookup.
        """
        return {
            "paper_id": self.paper_id,
            "index": self.index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section or "",
            "title": meta.title,
            "authors": "; ".join(meta.authors),
            "year": meta.year or 0,
            "subtopic": meta.subtopic or "",
            "pdf_url": meta.pdf_url,
        }


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, plus scoring provenance.

    ``marker`` is the citation label (``c1``, ``c2``, ...) assigned per turn and
    injected into the prompt. The citation validator checks the model's output
    against exactly this set.
    """

    chunk_id: str
    paper_id: str
    text: str
    page_start: int
    page_end: int
    section: str
    title: str
    authors: str
    year: int
    pdf_url: str = ""
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    marker: str = ""

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}–{self.page_end}"

    @property
    def citation(self) -> str:
        author = self.authors.split(";")[0].strip() if self.authors else "Unknown"
        if self.authors and len(self.authors.split(";")) > 1:
            author += " et al."
        year = self.year or "n.d."
        return f"{author} ({year}), {self.page_label}"

    @classmethod
    def from_chroma(
        cls,
        chunk_id: str,
        document: str,
        metadata: dict[str, Any],
    ) -> RetrievedChunk:
        return cls(
            chunk_id=chunk_id,
            paper_id=str(metadata.get("paper_id", "")),
            text=document,
            page_start=int(metadata.get("page_start", 0)),
            page_end=int(metadata.get("page_end", 0)),
            section=str(metadata.get("section", "")),
            title=str(metadata.get("title", "Untitled")),
            authors=str(metadata.get("authors", "")),
            year=int(metadata.get("year", 0) or 0),
            pdf_url=str(metadata.get("pdf_url", "")),
        )
