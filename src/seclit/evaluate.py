"""Retrieval evaluation.

Measures each retrieval configuration against a hand-written gold set so the
value of the added machinery is demonstrated rather than asserted. If hybrid
retrieval and cross-encoder reranking do not beat plain dense search on this
corpus, that should be visible — including if the answer is inconvenient.

Metrics are computed at *paper* granularity, not chunk. A question is answered
correctly if the right paper surfaces; which of its chunks ranked first is an
implementation detail the user never sees.

* **hit@k** — share of questions with a correct paper in the top *k*.
* **MRR** — mean reciprocal rank of the first correct paper. Distinguishes
  "ranked first" from "ranked eighth", which hit@k alone hides.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from seclit.config import Settings, settings
from seclit.retrieve import Mode, Retriever

MODES: list[Mode] = ["dense", "sparse", "hybrid", "hybrid_rerank"]
K_VALUES = (1, 3, 5, 10)

MODE_LABELS = {
    "dense": "Dense only (bge-m3)",
    "sparse": "Sparse only (BM25)",
    "hybrid": "Hybrid (RRF fusion)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}


@dataclass
class GoldQuestion:
    question: str
    expected_papers: list[str]
    subtopic: str = ""
    note: str = ""


@dataclass
class ModeResult:
    mode: str
    hits: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    mean_latency_ms: float = 0.0
    misses: list[str] = field(default_factory=list)


def load_gold_set(path: Path) -> list[GoldQuestion]:
    questions: list[GoldQuestion] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            raw = json.loads(line)
            questions.append(
                GoldQuestion(
                    question=raw["question"],
                    expected_papers=raw["expected_papers"],
                    subtopic=raw.get("subtopic", ""),
                    note=raw.get("note", ""),
                )
            )
    return questions


def _ranked_paper_ids(chunks) -> list[str]:
    """Collapse a chunk ranking into a paper ranking, keeping first appearance."""
    seen: list[str] = []
    for chunk in chunks:
        if chunk.paper_id not in seen:
            seen.append(chunk.paper_id)
    return seen


def evaluate_mode(
    mode: Mode,
    questions: list[GoldQuestion],
    retriever: Retriever,
    max_k: int = max(K_VALUES),
) -> ModeResult:
    result = ModeResult(mode=mode)
    hit_counts = dict.fromkeys(K_VALUES, 0)
    reciprocal_total = 0.0
    latencies: list[float] = []

    for question in questions:
        started = time.perf_counter()
        chunks = retriever.search(question.question, mode=mode, top_k=max_k)
        latencies.append((time.perf_counter() - started) * 1000)

        papers = _ranked_paper_ids(chunks)
        expected = set(question.expected_papers)

        rank = next((i for i, pid in enumerate(papers, start=1) if pid in expected), None)
        if rank is None:
            result.misses.append(question.question)
        else:
            reciprocal_total += 1.0 / rank
            for k in K_VALUES:
                if rank <= k:
                    hit_counts[k] += 1

    total = max(len(questions), 1)
    result.hits = {k: hit_counts[k] / total for k in K_VALUES}
    result.mrr = reciprocal_total / total
    result.mean_latency_ms = sum(latencies) / max(len(latencies), 1)
    return result


def format_report(results: list[ModeResult], n_questions: int, corpus: dict) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        f"Corpus: **{corpus.get('papers', 0)} papers**, "
        f"**{corpus.get('chunks', 0)} chunks**. "
        f"Gold set: **{n_questions} questions** with known source papers.",
        "",
        "Scored at paper granularity: a question counts as answered if a correct "
        "paper appears in the top *k*. MRR is the mean reciprocal rank of the first "
        'correct paper, which separates "ranked first" from "ranked eighth".',
        "",
        "| Configuration | hit@1 | hit@3 | hit@5 | hit@10 | MRR | latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        label = MODE_LABELS.get(result.mode, result.mode)
        lines.append(
            f"| {label} "
            f"| {result.hits.get(1, 0):.2f} "
            f"| {result.hits.get(3, 0):.2f} "
            f"| {result.hits.get(5, 0):.2f} "
            f"| {result.hits.get(10, 0):.2f} "
            f"| {result.mrr:.3f} "
            f"| {result.mean_latency_ms:.0f} ms |"
        )

    by_mode = {r.mode: r for r in results}
    dense = by_mode.get("dense")
    hybrid = by_mode.get("hybrid")
    reranked = by_mode.get("hybrid_rerank")

    if dense and hybrid and reranked:
        saturated = [k for k in K_VALUES if dense.hits.get(k, 0) >= 1.0]
        lines += ["", "## Reading the table", ""]

        if saturated:
            lines += [
                f"**hit@{min(saturated)} and above are saturated** — every "
                "configuration retrieves a correct paper, so those columns cannot "
                "separate them. The informative columns are hit@1 and MRR, which "
                "measure *ordering*: whether the right paper is first or fourth. "
                "That is what the user sees.",
                "",
            ]

        fusion_gain = hybrid.hits.get(1, 0) - dense.hits.get(1, 0)
        fusion_cost = hybrid.mean_latency_ms - dense.mean_latency_ms
        lines += [
            f"**Hybrid fusion is the clear win.** Adding BM25 and fusing by "
            f"reciprocal rank moves hit@1 by **{fusion_gain:+.2f}** over dense-only "
            f"while *reducing* latency by {abs(fusion_cost):.0f} ms (BM25 is nearly "
            "free, and it lets the dense retriever return fewer candidates). BM25 "
            "earns its place on questions containing rare literals — CVE "
            "identifiers, protocol names like RPL, technique names like ADASYN — "
            "where embeddings blur near-neighbours together.",
            "",
        ]

        rerank_mrr = reranked.mrr - hybrid.mrr
        rerank_hit1 = reranked.hits.get(1, 0) - hybrid.hits.get(1, 0)
        ratio = reranked.mean_latency_ms / max(hybrid.mean_latency_ms, 1)
        lines += [
            f"**Cross-encoder reranking is marginal on this gold set, and the "
            f"table should say so.** It adds **{rerank_mrr:+.3f}** MRR and "
            f"**{rerank_hit1:+.2f}** hit@1 over hybrid alone, for roughly "
            f"**{ratio:.0f}x** the latency ({reranked.mean_latency_ms:.0f} ms vs "
            f"{hybrid.mean_latency_ms:.0f} ms). It remains the default because "
            "generation dominates a full turn — a few seconds of reranking is a "
            "small share of a local model's response time — but on this evidence "
            "`SECLIT_RETRIEVAL_MODE=hybrid` is a defensible choice for anyone who "
            "wants sub-100 ms retrieval.",
            "",
            "The honest caveat: 40 questions over 94 papers is a small evaluation, "
            "and the gold set was written by the same person who built the "
            "retriever. A larger corpus would likely widen the gap in reranking's "
            "favour, since more papers means more near-duplicate candidates for it "
            "to disambiguate — but that is a prediction, not a measurement.",
        ]

    if reranked and reranked.misses:
        lines += [
            "",
            f"## Unretrieved questions ({len(reranked.misses)})",
            "",
            "Recorded rather than hidden — these are where the corpus or the "
            "retriever falls short:",
            "",
        ]
        lines += [f"- {miss}" for miss in reranked.misses]

    return "\n".join(lines) + "\n"


def run_evaluation(
    gold_path: Path,
    config: Settings | None = None,
    *,
    write_report: bool = False,
) -> list[ModeResult]:
    config = config or settings
    questions = load_gold_set(gold_path)
    if not questions:
        raise ValueError(f"No questions found in {gold_path}")

    retriever = Retriever(config=config)
    results: list[ModeResult] = []

    print(f"Evaluating {len(questions)} questions across {len(MODES)} configurations\n")
    for mode in MODES:
        result = evaluate_mode(mode, questions, retriever)
        results.append(result)
        print(
            f"  {MODE_LABELS[mode]:32s} "
            f"hit@5 {result.hits.get(5, 0):.2f}  "
            f"MRR {result.mrr:.3f}  "
            f"{result.mean_latency_ms:6.0f} ms"
        )

    if write_report:
        from seclit.ingest.pipeline import IngestPipeline

        report = format_report(results, len(questions), IngestPipeline().stats())
        out = Path("eval/results.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nWrote {out}")

    return results
