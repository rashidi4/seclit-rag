"""Streamlit interface for the seclit research assistant.

Design intent: make the retrieval and citation machinery *visible*. A chat box
that emits prose is easy; what makes this trustworthy for research use is that
the user can see which excerpt supports each claim, which pages it came from,
and whether any citation was rejected as invalid. Those are surfaced in the UI
rather than buried in logs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Support `streamlit run app/streamlit_app.py` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seclit.config import settings  # noqa: E402
from seclit.generate.chat import ChatEngine, Turn  # noqa: E402
from seclit.ingest.pipeline import IngestPipeline  # noqa: E402

st.set_page_config(
    page_title="seclit — security literature assistant", page_icon="🔐", layout="wide"
)


@st.cache_resource(show_spinner="Loading models and index…")
def load_engine() -> ChatEngine:
    """Cached across reruns — models load once per process, not per message."""
    return ChatEngine()


@st.cache_data(ttl=30)
def load_stats() -> dict:
    return IngestPipeline().stats()


def render_sources(chunks, key_prefix: str) -> None:
    """Render the cited sources with page numbers and the underlying excerpt."""
    if not chunks:
        return
    st.caption(f"{len(chunks)} source{'s' if len(chunks) != 1 else ''} cited")
    for chunk in chunks:
        section = f" · {chunk.section}" if chunk.section else ""
        score = f" · relevance {chunk.rerank_score:.2f}" if chunk.rerank_score is not None else ""
        label = f"[{chunk.marker}] {chunk.title} — {chunk.page_label}{section}"
        with st.expander(label, expanded=False):
            st.markdown(f"**{chunk.title}**")
            st.caption(
                f"{chunk.authors or 'Unknown authors'} · {chunk.year or 'n.d.'} · "
                f"{chunk.page_label}{section}{score}"
            )
            st.text(chunk.text)
            if chunk.pdf_url:
                st.markdown(f"[Open PDF]({chunk.pdf_url})")


def render_diagnostics(answer) -> None:
    """Show the citation audit. This is the anti-hallucination guarantee made
    visible rather than asserted."""
    report = answer.report
    if report is None:
        return

    cols = st.columns(3)
    cols[0].metric("Citation validity", f"{report.validity_rate:.0%}")
    cols[1].metric("Grounded sentences", f"{report.grounded_rate:.0%}")
    cols[2].metric("Citations made", report.total_markers)

    if report.invalid_markers:
        st.warning(
            f"Removed {len(report.invalid_markers)} citation(s) pointing to excerpts that "
            f"were never retrieved: {', '.join(sorted(set(report.invalid_markers)))}"
        )
    if report.uncited_claims:
        with st.expander(f"{len(report.uncited_claims)} sentence(s) without a citation"):
            for claim in report.uncited_claims:
                st.markdown(f"- {claim}")


# ---------------------------------------------------------------- sidebar ---

with st.sidebar:
    st.title("🔐 seclit")
    st.caption("Local RAG over cybersecurity literature")

    try:
        stats = load_stats()
    except Exception as exc:  # noqa: BLE001
        stats = {}
        st.error(f"Index unavailable: {exc}")

    if stats:
        col_a, col_b = st.columns(2)
        col_a.metric("Papers", stats.get("papers", 0))
        col_b.metric("Chunks", stats.get("chunks", 0))

        if stats.get("by_subtopic"):
            st.caption("Coverage by area")
            for topic, count in sorted(stats["by_subtopic"].items(), key=lambda kv: -kv[1]):
                st.progress(
                    count / max(stats["papers"], 1),
                    text=f"{topic.replace('_', ' ')} · {count}",
                )

    st.divider()

    mode = st.selectbox(
        "Retrieval mode",
        ["hybrid_rerank", "hybrid", "dense", "sparse"],
        index=0,
        help=(
            "hybrid_rerank is the production path: dense + BM25, fused by "
            "reciprocal rank, then reranked by a cross-encoder. The others are "
            "exposed so the difference is observable."
        ),
    )

    engine = load_engine()
    ok, detail = engine.provider.available()
    if ok:
        st.success(f"{engine.provider.name} · {engine.provider.model}")
    else:
        st.error(detail)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption(
        f"Embeddings `{settings.embedding_model}`  \n"
        f"Reranker `{settings.reranker_model}`  \n"
        f"Device `{settings.resolve_device()}`"
    )


# ------------------------------------------------------------------- chat ---

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Security literature assistant")
st.caption(
    "Answers are generated only from the indexed papers. Every factual sentence is "
    "cited to a specific excerpt and page, and citations that do not resolve to a "
    "retrieved excerpt are stripped before display."
)

for i, entry in enumerate(st.session_state.history):
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("answer") is not None:
            answer = entry["answer"]
            if answer.rewritten:
                st.caption(f"Searched for: _{answer.search_query}_")
            render_sources(answer.cited_chunks, key_prefix=f"h{i}")
            render_diagnostics(answer)

if question := st.chat_input("Ask about the indexed papers…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        ok, detail = engine.provider.available()
        if not ok:
            st.error(detail)
            st.stop()

        turns = [Turn(role=e["role"], content=e["content"]) for e in st.session_state.history[:-1]]

        with st.spinner("Retrieving…"):
            chunks, stream, handle = engine.ask_stream(question, turns, mode=mode)

        if chunks:
            st.caption(f"Reading {len(chunks)} excerpts…")

        placeholder = st.empty()
        buffer = ""
        for piece in stream:
            buffer += piece
            placeholder.markdown(buffer + "▌")

        # Swap the raw stream for the validated text. Validation needs the whole
        # answer, so it can only run once streaming completes.
        answer = handle.answer
        final_text = answer.text if answer else buffer
        placeholder.markdown(final_text)

        if answer:
            if answer.rewritten:
                st.caption(f"Searched for: _{answer.search_query}_")
            render_sources(answer.cited_chunks, key_prefix="live")
            render_diagnostics(answer)

        st.session_state.history.append(
            {"role": "assistant", "content": final_text, "answer": answer}
        )
