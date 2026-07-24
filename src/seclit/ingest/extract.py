"""PDF text extraction with page-level provenance.

Extraction happens page by page and never gets flattened into one blob. That is
deliberate: the brief requires citations to show page numbers, and the only way
to guarantee a page number is correct is to know it at extraction time. Any
approach that concatenates first and reconstructs pages later is guessing.

Text is extracted from layout *blocks* rather than raw lines. PDFs break text at
visual line ends, so a naive read yields one word per line for many papers.
Blocks correspond to paragraph-ish units, so lines can be reflowed within a
block while genuine paragraph and heading boundaries survive.

Four cleanups are applied because academic PDFs consistently need them:

1. **Reflow.** Lines inside a block are joined with spaces, undoing visual
   line wrapping.
2. **De-hyphenation.** PDF line wrapping splits ``vulner-\\nability``. Left
   alone, BM25 never matches "vulnerability" and dense retrieval degrades.
3. **Running header/footer removal.** Conference PDFs repeat a venue banner on
   every page. Repeated across 30 pages it becomes the most "central" text in
   the document and pollutes retrieval.
4. **Bibliography truncation.** Reference lists are dense with title keywords
   but contain no claims. They generate confident-looking retrievals that
   support nothing, which is a direct hallucination risk.

Finally, a quality gate rejects documents whose text came out garbled. Older
LaTeX PDFs with broken font encodings extract as ``Inje tor`` for "Injector" and
``mo dule`` for "module" — characters are silently dropped. Measured across a
sample corpus, healthy papers average 5.2–5.6 characters per word with under 8%
short-token fragments, while garbled ones fall to ~3–4 characters with 19–32%
fragments. The gap is wide enough to separate reliably. Indexing garbled text is
worse than skipping it: the chunks are unreadable but still retrievable, so they
surface as confident nonsense.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from seclit.models import Document, Page, PaperMeta

# Matches "References", "REFERENCES", "Bibliography", "7 References", etc.,
# alone on a line — the standard shapes for a bibliography heading.
_REFERENCES_RE = re.compile(
    r"^\s*(?:\d+\.?\s+|[IVXL]+\.?\s+)?(references|bibliography|works\s+cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Numbered or lettered section headings: "3 Threat Model", "4.2. Evaluation",
# "II. RELATED WORK". Also bare well-known headings like "Abstract".
_SECTION_RE = re.compile(
    r"^\s*(?:(?P<num>\d+(?:\.\d+)*\.?|[IVXL]+\.)\s+(?P<titled>[A-Z][^\n]{2,70})"
    r"|(?P<bare>Abstract|Introduction|Related Work|Background|Methodology|"
    r"Evaluation|Discussion|Conclusion|Threat Model|Related Works?)\s*)$",
    re.MULTILINE,
)

# A word split across a line break by a hyphen.
_HYPHEN_SPLIT_RE = re.compile(r"(\w+)-[ \t]*\n[ \t]*(\w+)")
# A genuine compound hyphen occurring within a single line.
_INLINE_HYPHEN_RE = re.compile(r"[A-Za-z]{2,}-[A-Za-z]{2,}")
_PAGE_NUM_LINE_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
# Inline bibliography references: "[11]", "[1, 2]", "[11]-[13]", "[4]–[6]".
# Purely numeric brackets only, so "[CVE-2021-44228]" and "[see Table 2]" survive.
# Matches a whole run, so "[4]-[6]" is consumed together rather than leaving a
# dangling separator behind.
_INLINE_REF_RE = re.compile(r"(?:\[\s*\d+(?:\s*[,;–—-]\s*\d+)*\s*\]\s*[-–—]?\s*)+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_WORD_RE = re.compile(r"[A-Za-z]+")

# Compounds a document may only ever use at a line break, leaving nothing
# inline for the self-disambiguation pass to learn from. Measured misses on the
# corpus were concentrated here: "denial-of-service", "well-known" and
# "on-the-fly" were being welded into "denialof", "wellknown", "thefly".
#
# Kept deliberately short. This is a backstop for the tail, not a dictionary —
# a general wordlist is the wrong tool for text whose important terms
# ("cluster-based", "vulnerability-triggering", "hex-rays") no wordlist
# contains. The document remains the primary oracle.
_COMMON_COMPOUNDS = frozenset(
    """
    well-known well-defined well-formed well-suited so-called state-of-the-art
    on-the-fly out-of-band out-of-scope end-to-end peer-to-peer man-in-the-middle
    denial-of-service proof-of-concept proof-of-work point-to-point
    real-time run-time compile-time third-party open-source closed-source
    cross-site cross-domain cross-platform side-channel zero-day zero-trust
    fine-grained coarse-grained large-scale small-scale high-level low-level
    read-only write-only single-sign-on multi-factor two-factor
    """.split()
)

# Genuine one- and two-letter English words, so they aren't counted as
# extraction damage.
_REAL_SHORT_WORDS = frozenset(
    "a i o is in it we be to of on as at by or an if no so do us my me he up "
    "vs et al re id ip os pc db ml ai".split()
)

# Thresholds calibrated against a sample of the target corpus (see module
# docstring). Deliberately loose — the goal is to catch clearly broken font
# encodings, not to police style.
GARBLED_SHORT_TOKEN_RATIO = 0.15
GARBLED_MEAN_WORD_LENGTH = 4.3
QUALITY_SAMPLE_CHARS = 20_000


def sha256_file(path: Path) -> str:
    """Content hash — the identity used for incremental-ingest deduplication.

    Hashing content rather than filename means the same paper saved under two
    names is ingested once.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_running_lines(page_texts: list[str], threshold: float = 0.4) -> set[str]:
    """Find short lines repeated across most pages — headers and footers.

    Only the first and last three lines of each page are considered, so a
    genuinely recurring phrase in body text is not stripped.
    """
    if len(page_texts) < 4:
        return set()

    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines[:3] + lines[-3:]:
            if 3 < len(line) < 120:
                counts[line] += 1

    cutoff = max(3, int(len(page_texts) * threshold))
    return {line for line, count in counts.items() if count >= cutoff}


def collect_hyphenated(text: str) -> set[str]:
    """Compound words that appear hyphenated *within* a line somewhere in the doc.

    Used to disambiguate line-break hyphens. ``vulner-\\nability`` should join
    into one word, but ``cluster-\\nbased`` should keep its hyphen — and the
    document itself usually says which is which, because the same compound
    almost always appears mid-line elsewhere.
    """
    return {m.group(0).lower() for m in _INLINE_HYPHEN_RE.finditer(text)}


def is_compound(pair: str, compounds: set[str]) -> bool:
    """Decide whether ``left-right`` is a real compound rather than a wrapped word.

    Three sources of evidence, cheapest first:

    1. The document uses it hyphenated inline. Strongest signal — it is the
       paper's own usage.
    2. It is the head of a longer compound the document uses, so a break inside
       ``denial-of-service`` at ``denial-|of`` is still recognised.
    3. It is a common compound that documents often only ever wrap on.
    """
    pair = pair.lower()
    if pair in compounds or pair in _COMMON_COMPOUNDS:
        return True
    prefix = f"{pair}-"
    if any(known.startswith(prefix) for known in compounds):
        return True
    return any(known.startswith(prefix) for known in _COMMON_COMPOUNDS)


def _dehyphenate(text: str, compounds: set[str]) -> str:
    """Resolve hyphens at line breaks, preferring the document's own usage."""

    def resolve(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if is_compound(f"{left}-{right}", compounds):
            return f"{left}-{right}"
        return f"{left}{right}"

    return _HYPHEN_SPLIT_RE.sub(resolve, text)


def _reflow_block(text: str, compounds: set[str] | None = None) -> str:
    """Undo visual line wrapping inside one layout block.

    De-hyphenation runs before the join so a wrapped word becomes one token
    rather than ``vulner- ability``.
    """
    text = _dehyphenate(text, compounds or set())
    lines = [ln.strip() for ln in text.splitlines()]
    return " ".join(ln for ln in lines if ln)


def _clean_page(blocks: list[str], running: set[str], compounds: set[str] | None = None) -> str:
    """Reflow and filter a page's blocks into paragraph-separated text."""
    kept: list[str] = []
    for block in blocks:
        flowed = _reflow_block(block, compounds)
        if not flowed:
            continue
        if flowed in running:
            continue
        if _PAGE_NUM_LINE_RE.match(flowed):
            continue
        kept.append(flowed)

    text = "\n\n".join(kept)
    # Drop the paper's own bibliography markers. They point into a reference
    # list that gets truncated anyway, they add meaningless integers to the BM25
    # index, and — observed in practice — a generating model reads "[11]" in an
    # excerpt and emits "[^c11]" as a citation, fabricating a source number that
    # was never retrieved.
    text = _INLINE_REF_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def extraction_quality(text: str) -> tuple[float, float]:
    """Return ``(short_token_ratio, mean_word_length)`` for a text sample.

    Both degrade together when a PDF's font encoding drops characters, which is
    what makes the pair a reliable garbling signal.
    """
    words = _WORD_RE.findall(text[:QUALITY_SAMPLE_CHARS])
    if len(words) < 200:
        # Too little text to judge; treat as healthy and let other checks decide.
        return 0.0, 99.0

    fragments = sum(1 for w in words if len(w) <= 2 and w.lower() not in _REAL_SHORT_WORDS)
    mean_length = sum(len(w) for w in words) / len(words)
    return fragments / len(words), mean_length


def is_garbled(text: str) -> bool:
    """Both signals must fire.

    Requiring both matters: text made of legitimately short words (a table of
    values, a repetitive list) drags the mean word length down while leaving the
    fragment ratio near zero. Font damage moves both together — measured on real
    examples, broken papers hit 0.19–0.32 fragments *and* 3.3–4.0 mean length,
    while healthy ones stay under 0.11 fragments. Treating either signal alone
    as sufficient rejects valid documents.
    """
    ratio, mean_length = extraction_quality(text)
    return ratio > GARBLED_SHORT_TOKEN_RATIO and mean_length < GARBLED_MEAN_WORD_LENGTH


def _truncate_at_references(pages: list[Page]) -> list[Page]:
    """Drop the bibliography, keeping everything before the heading.

    Only fires in the last 40% of the document so a forward-reference like
    "see References" in an introduction cannot truncate the whole paper.
    """
    if not pages:
        return pages

    earliest = int(len(pages) * 0.6)
    for idx in range(earliest, len(pages)):
        match = _REFERENCES_RE.search(pages[idx].text)
        if not match:
            continue

        head = pages[idx].text[: match.start()].strip()
        kept = list(pages[:idx])
        if len(head) > 200:  # meaningful content precedes the heading
            kept.append(Page(number=pages[idx].number, text=head))
        return kept or pages[: idx + 1]

    return pages


def detect_sections(text: str) -> list[tuple[int, str]]:
    """Return ``(char_offset, heading)`` pairs for detected section headings."""
    found: list[tuple[int, str]] = []
    for match in _SECTION_RE.finditer(text):
        if match.group("bare"):
            heading = match.group("bare").strip()
        else:
            num = (match.group("num") or "").strip()
            heading = f"{num} {match.group('titled').strip()}".strip()
        found.append((match.start(), heading))
    return found


def extract_document(
    pdf_path: Path,
    meta: PaperMeta | None = None,
    *,
    drop_references: bool = True,
    check_quality: bool = True,
) -> Document:
    """Extract a PDF into per-page text with cleanup applied.

    Raises ``ValueError`` when the PDF yields no usable text (scanned images
    needing OCR) or when extraction is garbled (broken font encoding). Both
    surface as explicit failures rather than silently indexing unusable text.
    """
    pdf_path = Path(pdf_path)
    with fitz.open(pdf_path) as doc:
        # sort=True yields blocks in reading order rather than internal PDF order.
        page_blocks = [
            [b[4] for b in page.get_text("blocks", sort=True) if b[4].strip()] for page in doc
        ]
        pdf_title = (doc.metadata or {}).get("title", "") or ""

    # Learn the document's own compound-word usage before reflowing, so
    # line-break hyphens can be resolved against it.
    compounds = collect_hyphenated("\n".join("\n".join(b) for b in page_blocks))

    running = _find_running_lines([_reflow_block("\n".join(b), compounds) for b in page_blocks])

    pages = []
    for page_no, blocks in enumerate(page_blocks, start=1):
        cleaned = _clean_page(blocks, running, compounds)
        if cleaned:
            pages.append(Page(number=page_no, text=cleaned))

    if not pages:
        raise ValueError(
            f"No extractable text in {pdf_path.name} — likely a scanned PDF needing OCR."
        )

    if check_quality:
        combined = "\n".join(p.text for p in pages)
        if is_garbled(combined):
            ratio, mean_length = extraction_quality(combined)
            raise ValueError(
                f"Garbled text extraction in {pdf_path.name} "
                f"(fragment ratio {ratio:.2f}, mean word length {mean_length:.1f}) — "
                "the PDF likely has a broken font encoding and needs OCR."
            )

    if drop_references:
        pages = _truncate_at_references(pages)

    if meta is None:
        meta = PaperMeta(
            paper_id=pdf_path.stem,
            title=pdf_title.strip() or pdf_path.stem,
        )
    if not meta.sha256:
        meta.sha256 = sha256_file(pdf_path)

    return Document(meta=meta, pages=pages, source_path=str(pdf_path))
