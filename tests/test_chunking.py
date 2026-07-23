"""Chunking tests.

The contract under test: every chunk reports a page range that actually
contains its text. Citations are only trustworthy if this holds, so these are
the highest-value tests in the suite.
"""

from __future__ import annotations

import pytest

from seclit.ingest.chunk import Chunker, PageIndex, approx_token_count
from seclit.models import Document, Page, PaperMeta


def make_doc(pages: list[str], paper_id: str = "test001") -> Document:
    return Document(
        meta=PaperMeta(paper_id=paper_id, title="Test Paper"),
        pages=[Page(number=i, text=t) for i, t in enumerate(pages, start=1)],
    )


def sentences(word: str, n: int) -> str:
    """n distinct sentences, so text is long but not degenerate."""
    return " ".join(f"The {word} number {i} explains a security property." for i in range(n))


class TestPageIndex:
    def test_maps_offsets_to_correct_pages(self):
        index = PageIndex([Page(1, "alpha"), Page(2, "beta"), Page(3, "gamma")])

        assert index.pages_for(0, 5) == (1, 1)
        beta_start = index.text.index("beta")
        assert index.pages_for(beta_start, beta_start + 4) == (2, 2)

    def test_span_crossing_pages_reports_full_range(self):
        index = PageIndex([Page(1, "alpha"), Page(2, "beta"), Page(3, "gamma")])
        gamma_end = index.text.index("gamma") + len("gamma")

        assert index.pages_for(0, gamma_end) == (1, 3)

    def test_page_numbers_are_preserved_not_reindexed(self):
        """Pages keep their original numbers even when earlier pages were
        dropped as empty during extraction."""
        index = PageIndex([Page(7, "seven"), Page(8, "eight")])

        assert index.pages_for(0, 5) == (7, 7)


class TestChunker:
    def test_short_document_yields_one_chunk(self):
        doc = make_doc(["Brief note about firewalls."])
        chunks = Chunker().chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1

    def test_every_chunk_page_range_contains_its_text(self):
        """The core invariant. For each chunk, the pages it claims must
        actually contain the opening of its text."""
        pages = [sentences(w, 60) for w in ("alpha", "bravo", "charlie", "delta")]
        doc = make_doc(pages)
        chunks = Chunker(target_tokens=200, overlap_ratio=0.1).chunk(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            claimed = " ".join(pages[p - 1] for p in range(chunk.page_start, chunk.page_end + 1))
            opening = chunk.text.split(".")[0].strip()
            assert opening in claimed, (
                f"{chunk.chunk_id} claims pp.{chunk.page_start}-{chunk.page_end} "
                f"but its text is not on those pages"
            )

    def test_page_ranges_are_ordered_and_in_bounds(self):
        doc = make_doc([sentences(w, 50) for w in ("one", "two", "three")])
        chunks = Chunker(target_tokens=150).chunk(doc)

        for chunk in chunks:
            assert 1 <= chunk.page_start <= chunk.page_end <= 3

    def test_chunk_ids_are_unique_and_stable(self):
        doc = make_doc([sentences("stable", 80)])
        first = Chunker(target_tokens=200).chunk(doc)
        second = Chunker(target_tokens=200).chunk(doc)

        ids = [c.chunk_id for c in first]
        assert len(ids) == len(set(ids))
        assert ids == [c.chunk_id for c in second]
        assert all(c.chunk_id.startswith("test001::") for c in first)

    def test_indices_are_sequential(self):
        doc = make_doc([sentences("seq", 90)])
        chunks = Chunker(target_tokens=150).chunk(doc)

        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_sections_are_detected_and_attached(self):
        text = (
            "1 Introduction\n\n"
            + sentences("intro", 40)
            + "\n\n2 Threat Model\n\n"
            + sentences("threat", 40)
        )
        chunks = Chunker(target_tokens=200).chunk(make_doc([text]))

        found = {c.section for c in chunks if c.section}
        assert any("Introduction" in s for s in found)
        assert any("Threat Model" in s for s in found)

    def test_chunks_do_not_span_sections(self):
        text = (
            "1 Introduction\n\n"
            + sentences("intro", 30)
            + "\n\n2 Evaluation\n\n"
            + sentences("eval", 30)
        )
        chunks = Chunker(target_tokens=2000).chunk(make_doc([text]))

        # Target is large enough to hold everything, so if sections were ignored
        # this would collapse to a single chunk.
        assert len(chunks) >= 2

    def test_overlap_shares_content_between_neighbours(self):
        doc = make_doc([sentences("overlap", 120)])
        chunks = Chunker(target_tokens=200, overlap_ratio=0.25).chunk(doc)

        assert len(chunks) > 1
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        assert first_words & second_words, "expected overlap between adjacent chunks"

    def test_zero_overlap_is_supported(self):
        doc = make_doc([sentences("nooverlap", 100)])
        chunks = Chunker(target_tokens=200, overlap_ratio=0.0).chunk(doc)

        assert len(chunks) > 1

    def test_rejects_invalid_overlap(self):
        with pytest.raises(ValueError):
            Chunker(overlap_ratio=1.0)

    def test_respects_custom_token_counter(self):
        """Chunk sizing must follow the injected tokenizer, not the default."""
        doc = make_doc([sentences("custom", 60)])
        greedy = Chunker(target_tokens=100, token_counter=lambda t: len(t.split()) * 10)
        lenient = Chunker(target_tokens=100, token_counter=lambda t: len(t.split()))

        assert len(greedy.chunk(doc)) > len(lenient.chunk(doc))


def test_approx_token_count_is_proportional():
    assert approx_token_count("") >= 0
    short = approx_token_count("one two three")
    long = approx_token_count(" ".join(["word"] * 100))
    assert long > short
