"""Citation extraction and validation.

The premise: asking a model to "cite your sources" is unenforced and unverified.
Models invent plausible-looking references, and nothing downstream notices.

Here every retrieved chunk gets a per-turn marker (``c1``, ``c2``, ...), the
model is required to tag claims with those markers, and **this module checks
every marker the model emitted against the set it was actually shown**. Markers
that don't resolve are stripped from the output and counted. The resulting
``validity_rate`` is a measurable property of the system rather than a promise.

Marker parsing is deliberately permissive about *form* while strict about
*identity*. Local models are inconsistent formatters — ``[^c1]``, ``[c1]``,
``(c1)``, ``[^c1, ^c3]`` all appear in practice — so all of those are accepted
and normalised. What is never relaxed is whether ``c1`` corresponds to a chunk
that was in this turn's context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Accepts [^c1], [c1], (c1), [^c1, c2], [c1][c2], with optional whitespace.
_MARKER_GROUP_RE = re.compile(r"[\[(]\s*\^?\s*(c\d+(?:\s*[,;]\s*\^?\s*c\d+)*)\s*[\])]", re.I)
_MARKER_ID_RE = re.compile(r"c(\d+)", re.I)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<![A-Z][a-z]\.)(?<!\bet al\.)(?<!\bFig\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\d\.)"
    r"(?<=[.!?])\s+"
)

# Sentences that make no factual claim don't need a citation.
_NON_CLAIM_PREFIXES = (
    "based on the provided",
    "the retrieved",
    "i could not find",
    "the provided sources",
    "in summary",
    "to summarise",
    "to summarize",
    "however,",
    "note that",
)


@dataclass
class CitationReport:
    """Outcome of validating one generated answer."""

    text: str
    valid_markers: list[str] = field(default_factory=list)
    invalid_markers: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    total_sentences: int = 0

    @property
    def total_markers(self) -> int:
        return len(self.valid_markers) + len(self.invalid_markers)

    @property
    def validity_rate(self) -> float:
        """Share of emitted markers that resolve to a real retrieved chunk.

        Returns 1.0 when the model emitted no markers at all — vacuously true.
        Pair it with ``grounded_rate`` before drawing conclusions.
        """
        if self.total_markers == 0:
            return 1.0
        return len(self.valid_markers) / self.total_markers

    @property
    def grounded_rate(self) -> float:
        """Share of claim-bearing sentences carrying at least one citation."""
        if self.total_sentences == 0:
            return 1.0
        return 1.0 - (len(self.uncited_claims) / self.total_sentences)

    @property
    def is_clean(self) -> bool:
        return not self.invalid_markers and not self.uncited_claims


def extract_markers(text: str) -> list[str]:
    """Return every marker id referenced in ``text``, in order, with repeats.

    Repeats are preserved because the validity rate is over *citations made*,
    not distinct sources.
    """
    found: list[str] = []
    for group in _MARKER_GROUP_RE.finditer(text):
        found.extend(f"c{m.group(1)}" for m in _MARKER_ID_RE.finditer(group.group(1)))
    return found


def _tidy(text: str) -> str:
    """Repair punctuation left behind when a marker is removed.

    Models often wrap markers in their own parentheses — ``([^c11], p. 3)``.
    Stripping the marker leaves ``(, p. 3)``, which reads as a formatting bug to
    the user even though the removal was correct. The prompt discourages that
    shape; this handles the cases where it happens anyway.
    """
    # "(, p. 3)" -> "(p. 3)"; "(, )" and "()" -> removed entirely.
    text = re.sub(r"\(\s*[,;]\s*", "(", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return text.strip()


def _is_claim(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(stripped) < 25:
        return False
    lowered = stripped.lower()
    if lowered.startswith(_NON_CLAIM_PREFIXES):
        return False
    # Headings and list scaffolding carry no claim of their own.
    return not (stripped.startswith("#") or stripped.rstrip().endswith(":"))


def validate_citations(
    text: str,
    allowed: set[str] | list[str],
    *,
    strip_invalid: bool = True,
    check_grounding: bool = True,
) -> CitationReport:
    """Check every marker in ``text`` against ``allowed``.

    Invalid markers are removed from the returned text by default: showing a
    user a citation that points nowhere is worse than showing none, because it
    still looks like evidence.
    """
    allowed_set = {m.lower() for m in allowed}

    valid: list[str] = []
    invalid: list[str] = []

    def replace_group(match: re.Match[str]) -> str:
        ids = [f"c{m.group(1)}" for m in _MARKER_ID_RE.finditer(match.group(1))]
        kept: list[str] = []
        for marker in ids:
            if marker.lower() in allowed_set:
                valid.append(marker)
                kept.append(marker)
            else:
                invalid.append(marker)
        if not kept:
            return ""
        return "[^" + ", ".join(kept) + "]"

    rendered = _MARKER_GROUP_RE.sub(replace_group, text)

    if not strip_invalid:
        rendered = text

    rendered = _tidy(rendered)

    uncited: list[str] = []
    sentences: list[str] = []
    if check_grounding:
        for block in rendered.split("\n"):
            for sentence in _SENTENCE_SPLIT_RE.split(block):
                if not _is_claim(sentence):
                    continue
                sentences.append(sentence)
                if not extract_markers(sentence):
                    uncited.append(sentence.strip())

    return CitationReport(
        text=rendered,
        valid_markers=valid,
        invalid_markers=invalid,
        uncited_claims=uncited,
        total_sentences=len(sentences),
    )


def render_sources(chunks, cited_only: bool = True) -> str:
    """Render a numbered source list for the markers actually used."""
    if not chunks:
        return ""
    lines = ["**Sources**"]
    for chunk in chunks:
        if cited_only and not chunk.marker:
            continue
        section = f" — {chunk.section}" if chunk.section else ""
        lines.append(
            f"- `[^{chunk.marker}]` {chunk.title} ({chunk.year}), {chunk.page_label}{section}"
        )
    return "\n".join(lines) if len(lines) > 1 else ""
