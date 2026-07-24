"""Prompt construction: the citation contract, query rewriting, summarisation.

The system prompt is written to be *checkable*. Every instruction it gives is
one the validator in ``cite.py`` can verify after the fact — marker syntax,
grounding in provided context, explicit refusal when coverage is missing. An
instruction that cannot be checked ("be accurate") is not worth the tokens.

Refusal is stated as a positive instruction rather than a prohibition. Told only
"do not use outside knowledge", models still answer from parametric memory
because refusing feels unhelpful. Given an explicit, approved way to fail, they
take it.
"""

from __future__ import annotations

from seclit.models import RetrievedChunk

SYSTEM_PROMPT_TEMPLATE = """You are a research assistant answering questions about a library of \
cybersecurity research papers.

You will be given numbered source excerpts. Follow these rules exactly.

1. GROUNDING. Answer only from the excerpts provided. Do not add facts from your own \
knowledge, even when you are confident they are correct.

2. CITATION. End every sentence that states a fact with a marker naming the excerpt \
that supports it, written exactly as [^c1]. Use [^c1, c2] when several excerpts \
support one sentence.

You have been given exactly {n} excerpts, numbered {first} to {last}. Only these \
markers exist. Never write a marker outside this range — a citation to an excerpt \
you were not given is worse than no citation at all.

Write the marker on its own, with nothing inside the brackets but the excerpt \
numbers. Do not add page numbers or titles to the marker: [^c1] is correct, \
([^c1], p. 3) is not. Page numbers are attached automatically from the source \
record.

3. INSUFFICIENT COVERAGE. If the excerpts do not answer the question, say so plainly: \
"The provided sources do not cover this." You may then state what related material the \
excerpts do contain. Reporting a gap is a correct and useful answer; guessing is not.

4. DISAGREEMENT. If excerpts conflict, report both positions with their citations \
rather than silently choosing one.

5. STYLE. Be direct and specific. Prefer the paper's own terminology. Do not open with \
a summary of the question or close with an offer of further help."""


CONTEXT_HEADER = "Here are the source excerpts:"

# A worked example is the single most effective lever on format adherence.
# Smaller local models follow a demonstrated shape far more reliably than a
# described one — in testing, gemma3:12b intermittently produced answers with
# no markers at all until an example was added.
#
# The example MUST be content-free. An earlier version used realistic prose
# about zero trust, including an invented "40% reduction" statistic. gemma3:12b
# reproduced it almost verbatim as a real answer, citing valid markers — so the
# validator scored it 100% valid and 100% grounded while the headline claim came
# from the prompt, not the corpus. A plausible example is an instruction to
# plagiarise it. Placeholders cannot be mistaken for findings.
FORMAT_EXAMPLE = """Example of the required answer format (placeholder text — \
never reuse these words or invent statistics like them):

  <a claim taken from the first excerpt> [^c1]. <a claim supported by two \
excerpts together> [^c1, c3]. <a further claim from another excerpt> [^c3].

Every sentence ends with a marker, and each marker contains nothing but excerpt \
numbers. Replace the angle-bracket placeholders with real content drawn from \
the excerpts above."""

QUESTION_TEMPLATE = """{header}

{context}

---

{example}

---

Question: {question}

Answer using only the excerpts above. Every factual sentence must end with a \
[^c...] marker."""


REWRITE_PROMPT = """Rewrite the user's latest message into a standalone search query.

Resolve pronouns and references against the conversation so the query makes sense \
on its own. Keep the user's technical terms exactly as written. Output only the \
rewritten query, with no preamble or explanation.

Conversation:
{history}

Latest message: {question}

Standalone search query:"""


SUMMARY_PROMPT = """Summarise the following excerpts from the paper "{title}".

Cover, where the excerpts support it: the problem addressed, the approach taken, the \
main findings, and any stated limitations.

{context}

---

{example}

---

Write the summary now. Every point must end with a [^c...] marker naming the excerpt \
it came from. Start with the substance — no preamble such as "Here is a summary"."""


def build_system_prompt(chunks: list[RetrievedChunk]) -> str:
    """Naming the valid marker range measurably reduces fabricated citations.

    Local models drift toward citation numbering they saw in training —
    inventing ``[^c11]`` and ``[^c17]`` when only eight excerpts were supplied.
    Stating the range explicitly gives the model the actual bound instead of a
    guessed one. The validator still enforces it; this just means it has less
    to catch.
    """
    markers = [c.marker for c in chunks if c.marker]
    return SYSTEM_PROMPT_TEMPLATE.format(
        n=len(markers),
        first=markers[0] if markers else "c1",
        last=markers[-1] if markers else "c1",
    )


def format_chunk(chunk: RetrievedChunk) -> str:
    """Render one excerpt with the provenance the model needs to cite it."""
    section = f", {chunk.section}" if chunk.section else ""
    header = (
        f'[^{chunk.marker}] "{chunk.title}" ({chunk.year or "n.d."}), {chunk.page_label}{section}'
    )
    return f"{header}\n{chunk.text}"


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(format_chunk(chunk) for chunk in chunks)


def build_question_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return QUESTION_TEMPLATE.format(
        header=CONTEXT_HEADER,
        context=build_context(chunks),
        example=FORMAT_EXAMPLE,
        question=question.strip(),
    )


def build_rewrite_prompt(question: str, history: list[dict[str, str]]) -> str:
    rendered = "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in history)
    return REWRITE_PROMPT.format(history=rendered or "(no prior turns)", question=question)


def build_summary_prompt(title: str, chunks: list[RetrievedChunk]) -> str:
    return SUMMARY_PROMPT.format(title=title, context=build_context(chunks), example=FORMAT_EXAMPLE)


NO_CONTEXT_ANSWER = (
    "The provided sources do not cover this. No excerpt in the indexed corpus was "
    "relevant enough to this question to support an answer."
)
