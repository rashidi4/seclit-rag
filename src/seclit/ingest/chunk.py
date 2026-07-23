"""Section-aware chunking that preserves page provenance.

The mechanism: concatenate pages into one character stream while recording a
``(start, end, page_number)`` span for each page. Chunk boundaries are then
computed against that stream, and every chunk's character span is mapped back
through the index to recover the exact pages it covers.

This is why fixed-size chunking was rejected. Splitting on a raw character count
loses the page index the moment a chunk straddles a page break, and page numbers
then have to be approximated. Approximate page numbers in a citation are worse
than none — they look authoritative and are wrong.

Chunks never cross section boundaries. A chunk spanning the end of "Threat
Model" and the start of "Evaluation" retrieves poorly for both.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from seclit.models import Chunk, Document, Page

# Sentence splitter that avoids breaking on common academic abbreviations and
# decimal numbers ("Fig. 3", "et al.", "CVE-2021-44228", "0.5").
_SENTENCE_RE = re.compile(
    r"(?<![A-Z][a-z]\.)(?<!\bet al\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bTab\.)"
    r"(?<!\bi\.e\.)(?<!\be\.g\.)(?<!\bcf\.)(?<!\bvs\.)(?<!\d\.)"
    r"(?<=[.!?])\s+(?=[A-Z(\[])"
)

_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def approx_token_count(text: str) -> int:
    """Fast token estimate: English averages ~1.3 subword tokens per word.

    Good enough for chunk sizing and keeps chunking free of a model dependency,
    which matters for fast unit tests. The pipeline injects the real tokenizer.
    """
    return int(len(text.split()) * 1.3) + 1


class PageIndex:
    """Maps character offsets in the concatenated stream back to page numbers."""

    SEPARATOR = "\n\n"

    def __init__(self, pages: list[Page]) -> None:
        self.spans: list[tuple[int, int, int]] = []
        parts: list[str] = []
        cursor = 0
        for page in pages:
            start = cursor
            parts.append(page.text)
            cursor += len(page.text)
            self.spans.append((start, cursor, page.number))
            parts.append(self.SEPARATOR)
            cursor += len(self.SEPARATOR)
        self.text = "".join(parts)

    def pages_for(self, start: int, end: int) -> tuple[int, int]:
        """Page range overlapping the character span ``[start, end)``."""
        touched = [
            page_no
            for span_start, span_end, page_no in self.spans
            if span_start < end and start < span_end
        ]
        if not touched:
            # Span landed entirely in a separator; fall back to nearest page.
            nearest = min(
                self.spans,
                key=lambda s: min(abs(s[0] - start), abs(s[1] - start)),
                default=(0, 0, 1),
            )
            return nearest[2], nearest[2]
        return min(touched), max(touched)


class Chunker:
    """Splits a document into overlapping, section-aligned chunks."""

    def __init__(
        self,
        target_tokens: int = 800,
        overlap_ratio: float = 0.15,
        min_tokens: int = 120,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")
        self.target_tokens = target_tokens
        self.overlap_tokens = int(target_tokens * overlap_ratio)
        self.min_tokens = min_tokens
        self.count = token_counter or approx_token_count

    # -- section segmentation ------------------------------------------------

    def _sections(self, index: PageIndex) -> list[tuple[int, int, str | None]]:
        """Split the stream into ``(start, end, heading)`` regions."""
        from seclit.ingest.extract import detect_sections

        headings = detect_sections(index.text)
        if not headings:
            return [(0, len(index.text), None)]

        regions: list[tuple[int, int, str | None]] = []
        if headings[0][0] > 0:
            regions.append((0, headings[0][0], None))

        for pos, (offset, heading) in enumerate(headings):
            end = headings[pos + 1][0] if pos + 1 < len(headings) else len(index.text)
            regions.append((offset, end, heading))
        return regions

    # -- splitting -----------------------------------------------------------

    def _split_units(self, text: str) -> list[str]:
        """Break text into paragraphs, subdividing any that exceed the target."""
        units: list[str] = []
        for para in _PARAGRAPH_RE.split(text):
            para = para.strip()
            if not para:
                continue
            if self.count(para) <= self.target_tokens:
                units.append(para)
                continue

            # Oversized paragraph — fall back to sentences.
            buffer: list[str] = []
            for sentence in _SENTENCE_RE.split(para):
                sentence = sentence.strip()
                if not sentence:
                    continue
                candidate = " ".join([*buffer, sentence])
                if buffer and self.count(candidate) > self.target_tokens:
                    units.append(" ".join(buffer))
                    buffer = [sentence]
                else:
                    buffer.append(sentence)
            if buffer:
                units.append(" ".join(buffer))
        return units

    def _overlap_tail(self, units: list[str]) -> list[str]:
        """Trailing units totalling roughly ``overlap_tokens``, for continuity."""
        if not self.overlap_tokens:
            return []
        tail: list[str] = []
        total = 0
        for unit in reversed(units):
            unit_tokens = self.count(unit)
            if total + unit_tokens > self.overlap_tokens and tail:
                break
            tail.insert(0, unit)
            total += unit_tokens
        return tail

    # -- public API ----------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        index = PageIndex(doc.pages)
        chunks: list[Chunk] = []
        counter = 0
        search_from = 0

        for region_start, region_end, heading in self._sections(index):
            region_text = index.text[region_start:region_end]
            if not region_text.strip():
                continue

            groups: list[list[str]] = []
            current: list[str] = []
            current_tokens = 0

            for unit in self._split_units(region_text):
                unit_tokens = self.count(unit)
                if current and current_tokens + unit_tokens > self.target_tokens:
                    groups.append(current)
                    current = [*self._overlap_tail(current), unit]
                    current_tokens = sum(self.count(u) for u in current)
                else:
                    current.append(unit)
                    current_tokens += unit_tokens
            if current:
                groups.append(current)

            for group in groups:
                text = "\n\n".join(group)
                if self.count(text) < self.min_tokens and chunks:
                    continue  # too small to stand alone; overlap already covers it

                # Locate this chunk in the stream to recover its page span.
                # Searching forward from the last match keeps repeated text
                # (common in overlap regions) mapped to the right occurrence.
                anchor = group[0][:120]
                found = index.text.find(anchor, search_from)
                if found == -1:
                    found = index.text.find(anchor, region_start)
                start = found if found != -1 else region_start
                end = min(start + len(text), len(index.text))
                search_from = start + max(1, len(anchor) // 2)

                page_start, page_end = index.pages_for(start, end)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.meta.paper_id}::{counter:04d}",
                        paper_id=doc.meta.paper_id,
                        index=counter,
                        text=text,
                        page_start=page_start,
                        page_end=page_end,
                        section=heading,
                    )
                )
                counter += 1

        return chunks
