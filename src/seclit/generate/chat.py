"""Chat orchestration: rewrite -> retrieve -> generate -> validate.

Two details worth calling out.

**Query rewriting.** Follow-up questions ("how does that compare to the second
approach?") are meaningless to a retriever on their own. Each turn is rewritten
into a standalone query against recent history before retrieval runs. Without
this, multi-turn conversation degrades to independent one-shot lookups that
silently retrieve the wrong thing.

**Validation is terminal, not advisory.** Generation output is never returned
directly. It always passes through the citation validator first, so an
unresolvable marker cannot reach the user. When streaming, the raw text is shown
live for responsiveness and then replaced by the validated text once the stream
completes — the only correct order, since validation needs the whole answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from seclit.config import Settings, settings
from seclit.generate.cite import CitationReport, validate_citations
from seclit.generate.prompt import (
    NO_CONTEXT_ANSWER,
    build_question_prompt,
    build_rewrite_prompt,
    build_system_prompt,
)
from seclit.generate.providers import LLMProvider, get_provider
from seclit.models import RetrievedChunk
from seclit.retrieve import Mode, Retriever


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Answer:
    text: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    report: CitationReport | None = None
    search_query: str = ""
    rewritten: bool = False

    @property
    def cited_chunks(self) -> list[RetrievedChunk]:
        """Only the sources the answer actually cites.

        The UI shows these rather than everything retrieved: listing eight
        sources for an answer that used two implies support that isn't there.
        """
        if self.report is None:
            return self.chunks
        used = {m.lower() for m in self.report.valid_markers}
        return [c for c in self.chunks if c.marker.lower() in used]


class ChatEngine:
    def __init__(
        self,
        retriever: Retriever | None = None,
        provider: LLMProvider | None = None,
        config: Settings | None = None,
    ) -> None:
        self.config = config or settings
        self.retriever = retriever or Retriever(config=self.config)
        self.provider = provider or get_provider(config=self.config)

    # -- query preparation ---------------------------------------------------

    def rewrite_query(self, question: str, history: list[Turn]) -> tuple[str, bool]:
        """Resolve a follow-up into a standalone query. Returns ``(query, changed)``.

        Falls back to the original question on any failure — a degraded query is
        better than a failed turn.
        """
        if not history:
            return question, False

        recent = history[-self.config.history_window :]
        prompt = build_rewrite_prompt(
            question, [{"role": t.role, "content": t.content} for t in recent]
        )
        try:
            rewritten = self.provider.complete(prompt, max_tokens=120).strip()
        except Exception:  # noqa: BLE001 - never fail a turn on the rewrite
            return question, False

        rewritten = rewritten.strip().strip('"').split("\n")[0].strip()
        # Guard against a model that returns commentary instead of a query.
        if not rewritten or len(rewritten) > 400:
            return question, False
        return rewritten, rewritten.lower() != question.strip().lower()

    def retrieve(self, query: str, mode: Mode | None = None) -> list[RetrievedChunk]:
        return self.retriever.with_markers(query, mode=mode)

    # -- generation ----------------------------------------------------------

    def ask(
        self,
        question: str,
        history: list[Turn] | None = None,
        *,
        mode: Mode | None = None,
    ) -> Answer:
        history = history or []
        search_query, rewritten = self.rewrite_query(question, history)
        chunks = self.retrieve(search_query, mode=mode or self.config.retrieval_mode)

        if not chunks:
            return Answer(
                text=NO_CONTEXT_ANSWER, chunks=[], search_query=search_query, rewritten=rewritten
            )

        prompt = build_question_prompt(question, chunks)
        raw = self.provider.complete(prompt, system=build_system_prompt(chunks))
        report = validate_citations(raw, {c.marker for c in chunks})

        return Answer(
            text=report.text,
            chunks=chunks,
            report=report,
            search_query=search_query,
            rewritten=rewritten,
        )

    def ask_stream(
        self,
        question: str,
        history: list[Turn] | None = None,
        *,
        mode: Mode | None = None,
    ) -> tuple[list[RetrievedChunk], Iterator[str], _StreamResult]:
        """Stream an answer.

        Returns the retrieved chunks (so the UI can render sources immediately),
        a text iterator, and a handle that holds the validated result once the
        iterator is exhausted.
        """
        history = history or []
        search_query, rewritten = self.rewrite_query(question, history)
        chunks = self.retrieve(search_query, mode=mode or self.config.retrieval_mode)
        handle = _StreamResult(search_query=search_query, rewritten=rewritten)

        if not chunks:
            handle.finish(NO_CONTEXT_ANSWER, [], NO_CONTEXT_ANSWER)
            return [], iter([NO_CONTEXT_ANSWER]), handle

        prompt = build_question_prompt(question, chunks)

        def generate() -> Iterator[str]:
            pieces: list[str] = []
            for piece in self.provider.stream(prompt, system=build_system_prompt(chunks)):
                pieces.append(piece)
                yield piece
            handle.finish("".join(pieces), chunks, None)

        return chunks, generate(), handle


@dataclass
class _StreamResult:
    """Holds the validated answer produced after a stream completes."""

    search_query: str = ""
    rewritten: bool = False
    answer: Answer | None = None

    def finish(
        self,
        raw: str,
        chunks: list[RetrievedChunk],
        override: str | None,
    ) -> None:
        if override is not None:
            self.answer = Answer(
                text=override,
                chunks=chunks,
                search_query=self.search_query,
                rewritten=self.rewritten,
            )
            return

        report = validate_citations(raw, {c.marker for c in chunks})
        self.answer = Answer(
            text=report.text,
            chunks=chunks,
            report=report,
            search_query=self.search_query,
            rewritten=self.rewritten,
        )
