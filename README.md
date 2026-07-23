# seclit

A local RAG research assistant over a corpus of cybersecurity papers. Ask
questions in natural language, get answers grounded in the papers, with
**page-accurate citations that are verified rather than trusted**.

Runs entirely on your machine. No API key required.

---

## What makes it different

Most RAG systems ask the model to cite its sources and hope. This one checks.

Every retrieved excerpt gets a marker (`c1`, `c2`, …). The model is required to
tag each factual sentence with one. After generation, **every marker the model
emitted is resolved against the set it was actually shown** — anything that
doesn't resolve is stripped from the answer and counted. Two numbers are
reported with every response:

- **Citation validity** — share of citations pointing at a real retrieved excerpt
- **Grounding** — share of factual sentences carrying any citation

This turns "minimises hallucinations" from a claim into a measurement.

**What it does not catch:** validity measures whether a marker resolves to a
retrieved excerpt — not whether that excerpt actually supports the sentence. A
model citing a real source for a claim it doesn't make still scores 100%.
Closing that gap needs entailment checking, which is documented as the next
increment in [ARCHITECTURE.md](ARCHITECTURE.md#what-this-does-not-catch). What
exists today eliminates *fabricated sources*; it does not yet eliminate
*misattributed claims*.

---

## Quick start

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/),
[Ollama](https://ollama.com). Roughly 8 GB free RAM and 6 GB disk for models.

```bash
git clone https://github.com/rashidi4/seclit-rag.git
cd seclit-rag

# 1. Install dependencies (uv fetches Python 3.12 automatically)
uv sync

# 2. Pull a local model
ollama pull gemma3:12b

# 3. Download the corpus from arXiv (rate-limited, ~6 minutes)
uv run seclit fetch

# 4. Build the index (~30 minutes on an M-series Mac; first run downloads models)
uv run seclit index

# 5. Launch
uv run streamlit run app/streamlit_app.py
```

Ask something from the terminal instead:

```bash
uv run seclit ask "How does zero trust limit lateral movement?"
```

---

## Corpus

96 papers sourced from arXiv's `cs.CR` category, balanced across the six areas
in the brief. Two were rejected by the extraction quality gate (broken font
encodings — see [ARCHITECTURE.md](ARCHITECTURE.md)), leaving **94 indexed**.

| Area | Papers |
|---|---|
| Cloud security | 16 |
| Network security | 16 |
| Threat intelligence | 16 |
| Vulnerability management | 16 |
| Zero trust | 16 |
| Malware analysis | 14 |
| **Total indexed** | **94** (2,402 chunks, 1,451 pages) |

`data/corpus_manifest.jsonl` records every paper with its arXiv ID, title,
authors, subtopic, PDF URL and SHA-256, so the exact corpus can be rebuilt and
verified from scratch.

### Scaling to 300–500 papers

The pipeline is corpus-agnostic and the size is a single flag:

```bash
uv run seclit fetch -n 500 --refresh
uv run seclit index
```

Measured on an M4 Pro (24 GB): download is bounded by arXiv's
one-request-per-three-seconds limit at **~3 s/paper**, and indexing runs at
**~19 s/paper**. So 500 papers is roughly **25 minutes of downloading and
2.6 hours of indexing**, both unattended and both resumable — re-running either
command skips work already done.

---

## Adding papers later

Adding a paper does **not** rebuild the index. Only the new paper's chunks are
written; every existing chunk ID is untouched.

```bash
uv run seclit add /path/to/new-paper.pdf
```

```
Added new-paper: 27 chunks from 14 pages (2402 -> 2429 vectors, no rebuild)
```

Deduplication is by SHA-256 of file content, so re-adding the same paper under a
different filename is a no-op. This behaviour is enforced by tests
(`tests/test_incremental_ingest.py`).

---

## The stack

| Layer | Choice | Why |
|---|---|---|
| PDF extraction | PyMuPDF, page-by-page | Page provenance must be established at extraction to be trustworthy |
| Chunking | Section-aware, 800 tokens, 15% overlap | Chunks carry exact page spans; never cross section boundaries |
| Embeddings | `BAAI/bge-m3` (MIT, 1024-dim, 8192 ctx) | Long context suits paper sections; permissive licence |
| Vector store | ChromaDB (persistent) | Stated preference in the brief |
| Retrieval | Dense + BM25 → RRF → cross-encoder rerank | Security text is full of rare literals that embeddings blur |
| Reranker | `BAAI/bge-reranker-v2-m3` | Term-level query/document interaction a bi-encoder cannot model |
| LLM | Ollama (`gemma3:12b`) by default | Local, free, satisfies the local-execution requirement literally |
| UI | Streamlit | Stated preference in the brief |

Orchestration is custom rather than LangChain/LlamaIndex — reasoning in
[ARCHITECTURE.md](ARCHITECTURE.md).

### Using a hosted model instead

The generation layer is provider-agnostic. Nothing else changes:

```bash
uv sync --extra anthropic
export ANTHROPIC_API_KEY=sk-ant-...
SECLIT_PROVIDER=anthropic uv run streamlit run app/streamlit_app.py
```

Adapters ship for `anthropic`, `openai`, and `gemini`. For a hosted option,
Claude is the recommendation — long context for many excerpts at once and
reliable adherence to the marker format the validator depends on.

---

## Retrieval quality

Measured against a 40-question gold set with known source papers, scored at
paper granularity.

| Configuration | hit@1 | MRR | Latency |
|---|---|---|---|
| Sparse only (BM25) | 0.75 | 0.834 | 11 ms |
| Dense only (bge-m3) | 0.88 | 0.929 | 191 ms |
| **Hybrid (RRF fusion)** | **0.93** | **0.952** | **25 ms** |
| Hybrid + cross-encoder rerank | 0.93 | 0.963 | 3456 ms |

Two findings worth reporting because they cut against the design:

1. **Hybrid fusion is where the gain is** — +0.05 hit@1 over dense, at lower
   latency than dense alone.
2. **Reranking is marginal here** — +0.010 MRR for ~137× the latency. It stays
   on by default because generation dominates a turn, but
   `SECLIT_RETRIEVAL_MODE=hybrid` is a reasonable choice.

An early tuning shortcut (rerank `max_length=512`, validated on one query)
scored *worse* than no reranking at all across the full gold set. Full
discussion in [eval/results.md](eval/results.md).

```bash
uv run seclit eval --write
```

---

## Configuration

Every setting is env-overridable with the `SECLIT_` prefix — see
[.env.example](.env.example).

```bash
SECLIT_OLLAMA_MODEL=llama3.1:8b    # different local model
SECLIT_FINAL_TOP_K=12              # more excerpts per answer
SECLIT_RERANK_CANDIDATES=40        # deeper rerank, slower
SECLIT_DEVICE=cpu                  # force CPU
```

---

## Testing

```bash
uv run pytest -q                  # full suite
uv run pytest -q -m "not slow"    # skip tests needing a built index
```

Tests concentrate on the places where silent failure is most costly: page-span
preservation through chunking, rejection of fabricated citations, and the
incremental-ingest guarantees (no duplication, no rebuild, no ID rewrites).

---

## Project layout

```
src/seclit/
  ingest/      fetch, extract, chunk, embed, store, pipeline
  retrieve/    dense, sparse, fuse, rerank
  generate/    prompt, cite, chat, providers/
  evaluate.py  retrieval evaluation harness
  cli.py
app/           Streamlit interface
eval/          gold set + results
tests/
data/          PDFs, Chroma index, manifest (gitignored)
```

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — how it works and why, including the
  measurements behind the tuning decisions
- **[RECOMMENDATIONS.md](RECOMMENDATIONS.md)** — two alternative stacks worth
  considering (Hermes for refusal-sensitive corpora; Obsidian for a durable
  personal library)

## Licence

MIT. Indexed papers remain under their original licences; the manifest records
each paper's source URL.
