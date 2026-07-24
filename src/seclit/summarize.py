"""Whole-document summarisation.

Summarising a paper differs from answering a question: there is no query to
retrieve against, so chunks are selected by *position* rather than relevance.

Chunks are sampled evenly across the document with the opening always included.
The opening carries the abstract and problem statement; even spacing prevents
the summary from over-representing whichever section happens to be longest.
Taking the first N chunks instead would summarise the introduction and miss the
findings entirely.

Citations still apply. A summary is a set of factual claims about the paper, so
each one is marked and validated exactly as in question answering.
"""

from __future__ import annotations

from dataclasses import dataclass

from seclit.config import Settings, settings
from seclit.generate.cite import CitationReport, validate_citations
from seclit.generate.prompt import build_summary_prompt, build_system_prompt
from seclit.generate.providers import LLMProvider, get_provider
from seclit.ingest.store import ChunkStore
from seclit.models import RetrievedChunk

MAX_SUMMARY_CHUNKS = 12


@dataclass
class Summary:
    paper_id: str
    title: str
    text: str
    chunks: list[RetrievedChunk]
    report: CitationReport | None = None


def load_paper_chunks(paper_id: str, store: ChunkStore) -> list[RetrievedChunk]:
    """Fetch every chunk of one paper, in document order."""
    payload = store.collection.get(where={"paper_id": paper_id}, include=["documents", "metadatas"])
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []

    chunks = [
        RetrievedChunk.from_chroma(cid, doc, meta or {})
        for cid, doc, meta in zip(ids, documents, metadatas, strict=False)
    ]
    # Chroma does not guarantee ordering; chunk index is authoritative.
    chunks.sort(key=lambda c: int(c.chunk_id.rsplit("::", 1)[-1]))
    return chunks


def select_representative(
    chunks: list[RetrievedChunk], limit: int = MAX_SUMMARY_CHUNKS
) -> list[RetrievedChunk]:
    """Evenly spaced sample across the document, always keeping the opening."""
    if len(chunks) <= limit:
        return chunks

    step = len(chunks) / limit
    picked = {0}
    picked.update(int(i * step) for i in range(1, limit))
    return [chunks[i] for i in sorted(picked) if i < len(chunks)]


def summarize_paper(
    paper_id: str,
    store: ChunkStore | None = None,
    provider: LLMProvider | None = None,
    config: Settings | None = None,
) -> Summary:
    config = config or settings
    store = store or ChunkStore(config)
    provider = provider or get_provider(config=config)

    chunks = load_paper_chunks(paper_id, store)
    if not chunks:
        return Summary(
            paper_id=paper_id,
            title=paper_id,
            text=f"No indexed content found for paper '{paper_id}'.",
            chunks=[],
        )

    selected = select_representative(chunks)
    for position, chunk in enumerate(selected, start=1):
        chunk.marker = f"c{position}"

    title = selected[0].title or paper_id
    # The system prompt carries the citation contract. Omitting it here meant
    # the summary path silently bypassed the guarantee the rest of the system
    # is built on — summaries came back with no markers at all.
    raw = provider.complete(
        build_summary_prompt(title, selected),
        system=build_system_prompt(selected),
    )
    report = validate_citations(raw, {c.marker for c in selected})

    return Summary(
        paper_id=paper_id,
        title=title,
        text=report.text,
        chunks=selected,
        report=report,
    )
