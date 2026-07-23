# Architecture

How the system works, and — more usefully — why each piece is there and what
failure it prevents.

```
                     ┌──────────────┐
  arXiv cs.CR  ─────▶│  fetch.py    │  rate-limited 1 req / 3 s
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
  PDF  ─────────────▶│  extract.py  │  per-page text, reflow, de-hyphenate,
                     └──────┬───────┘  strip refs, quality gate
                            ▼
                     ┌──────────────┐
                     │  chunk.py    │  section-aware, page spans preserved
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │  embed.py    │  bge-m3, 1024-dim, normalised
                     └──────┬───────┘
                            ▼
              ┌─────────────────────────┐
              │  Chroma  +  SQLite      │  vectors    +    ingest ledger
              └────────────┬────────────┘
                           │
   query ──┬──▶ dense (bge-m3) ──┐
           └──▶ sparse (BM25) ───┤
                                 ▼
                          RRF fusion
                                 ▼
                    cross-encoder rerank + score floor
                                 ▼
                     prompt with markers c1..cN
                                 ▼
                        LLM (Ollama by default)
                                 ▼
                   ┌─────────────────────────┐
                   │  citation validator     │  ← every marker checked
                   └─────────────────────────┘
                                 ▼
                          answer + sources
```

---

## The design constraint everything follows from

The brief asks for citations with page numbers and for hallucinations to be
minimised. Those two requirements are the same requirement: **a citation is only
useful if it is verifiable, and it is only verifiable if the page number is
correct and the source was really used.**

So page provenance is established once, at extraction, and carried untouched
through every later stage. Nothing downstream ever reconstructs or infers a page
number. Any design that flattens a PDF into one string and recovers pages later
is guessing, and a guessed page number is worse than none — it looks
authoritative and sends the reader to the wrong place.

---

## Ingestion

### Extraction (`ingest/extract.py`)

Text comes from PyMuPDF **layout blocks**, not raw lines. PDFs break text at
visual line ends, so a naive read produces one word per line on many papers.
Blocks map to paragraph-like units, so lines can be reflowed while paragraph and
heading boundaries survive.

Four cleanups, each for an observed problem in the real corpus:

| Cleanup | Problem it solves |
|---|---|
| **Reflow** | Word-per-line extraction destroys readability and paragraph structure |
| **De-hyphenation** | `vulner-\nability` never matches a query for "vulnerability" |
| **Header/footer removal** | A venue banner repeated on 30 pages becomes the document's most "central" text |
| **Reference stripping** | See below — this one was found by observation, not anticipated |

**De-hyphenation is document-aware.** Joining every line-break hyphen turns
`cluster-\nbased` into `clusterbased`, which then matches nothing — a real cost
in a corpus full of compounds (`zero-trust`, `cross-site`, `signature-based`).
So the document's own inline usage is collected first: if `cluster-based`
appears mid-line anywhere in the paper, the hyphen is restored; otherwise the
word is joined. Across 20 sample papers this preserved 842 genuine compounds.

**Reference stripping was a measured fix, not a precaution.** Early end-to-end
testing produced answers citing `[^c11]` and `[^c17]` when only `c1`–`c8`
existed. The cause was not the prompt: papers contain their own bibliography
markers (`[11]`, `[17]`), those survived into the chunks, and the model echoed
them back as citation markers. Purely numeric brackets are now removed at
extraction. `[CVE-2021-44228]` and `[see Table 2]` are preserved, since only
all-digit contents are matched.

**A quality gate rejects garbled extractions.** Older LaTeX PDFs with broken
font encodings drop characters: `Injector` extracts as `Inje tor`, `module` as
`mo dule`. Measured across the corpus, healthy papers average 5.2–5.6 characters
per word with under 8% short-token fragments; broken ones fall to 3.3–4.0
characters with 19–32% fragments.

Both signals must fire to reject. Requiring either alone was too aggressive —
legitimately short-worded text (a table of values, a repetitive list) drags mean
word length down while the fragment ratio stays near zero. This was caught by a
test whose fixture used repeated short words, not by inspection.

Two of 96 papers were rejected. Indexing them would have been worse than
skipping: unreadable chunks are still *retrievable*, so they surface as
confident nonsense.

### Chunking (`ingest/chunk.py`)

Pages are concatenated into one character stream while a
`(start, end, page_number)` span is recorded per page. Chunk boundaries are
computed against that stream, then each chunk's character span is mapped back
through the index to recover its exact page range.

- **Target 800 tokens, 15% overlap.** Large enough to keep an argument with its
  premises; overlap prevents a claim being severed from its qualifier.
- **Chunks never cross section boundaries.** A chunk spanning the end of "Threat
  Model" and the start of "Evaluation" retrieves poorly for both.
- **Deterministic IDs** (`{paper_id}::{index:04d}`), which is what makes
  re-ingestion an overwrite rather than a duplication.

### Storage (`ingest/store.py`)

Two stores, because they answer different questions:

- **Chroma** — "which chunks are similar to this query?"
- **SQLite catalog** — "have I already ingested this paper, and which chunks are
  its?"

The catalog is what makes incremental ingest possible. Without a ledger keyed by
content hash, the only way to know whether a paper is indexed is to scan the
vector store, and the only safe way to add one is to rebuild. The brief requires
adding papers *without* rebuilding, so the ledger is load-bearing.

Deduplication is by **SHA-256 of file content**, so the same paper saved under
two filenames is indexed once.

---

## Retrieval

Four stages, each preventing a distinct failure:

| Stage | Catches | Without it |
|---|---|---|
| **Dense** (bge-m3) | Paraphrase | "how do I stop lateral movement" misses a chunk that never uses those words |
| **Sparse** (BM25) | Rare literals | `CVE-2021-44228`, `RPL`, `ADASYN` blur into near-neighbours |
| **RRF fusion** | Score incomparability | Cosine ∈ [-1,1] and unbounded BM25 need tuned constants to blend |
| **Cross-encoder rerank** | Term-level interaction | Bi-encoders score query and document separately and cannot model their interaction |

**Why RRF rather than weighted scores.** BM25 is unbounded and corpus-dependent;
cosine similarity is bounded. Normalising them onto a shared scale needs
constants that drift as the corpus grows. Ranks are already scale-free, so RRF
needs no tuning and cannot be destabilised by one retriever emitting unusually
large scores.

**The relevance floor is a hallucination control.** Retrieval always returns
*something* — ask an unrelated question and you get the k least-unrelated
chunks. Passing those to a model invites synthesis from irrelevant context.
Candidates below the floor are dropped so the system can return nothing and say
so.

### What the measurements actually showed

Full results in [eval/results.md](eval/results.md). Two findings are worth
stating plainly, because both contradict what the design intended.

**Hybrid fusion is the real win — not reranking.** Against a dense-only
baseline, adding BM25 and RRF moves hit@1 from 0.88 to 0.93 *and reduces*
latency from 191 ms to 25 ms. Cross-encoder reranking then adds only **+0.010
MRR and +0.00 hit@1** over hybrid, for roughly **137× the latency**. An earlier
draft of this document called reranking "the single biggest quality lever."
That was an assumption, and the evaluation falsified it.

Reranking stays on by default because generation dominates a full turn — a few
seconds is a small share of a local model's 40–50 s response — but on this
evidence `SECLIT_RETRIEVAL_MODE=hybrid` is a defensible choice, and the config
says so.

**A tuning shortcut nearly shipped a regression.** Rerank `max_length` was first
set to 512 after checking that the top-5 papers were unchanged *on a single
query*. Run across all 40 gold questions, that setting scored **worse than no
reranking at all**:

| `max_length` | candidates | hit@1 | MRR | latency |
|---|---|---|---|---|
| 512 | 24 | 0.82 | 0.912 | 2184 ms |
| **2048** | **24** | **0.93** | **0.963** | **3556 ms** |
| 8192 | 24 | 0.93 | 0.963 | 3713 ms |
| 8192 | 40 | 0.95 | 0.975 | 10762 ms |

(Hybrid alone scores 0.952 MRR, so 512 was actively harmful.) 512 truncates
mid-chunk and discards the evidence the cross-encoder needs; 2048 captures the
full benefit, and 8192 adds latency for nothing because chunks rarely exceed
2048 tokens. The lesson is generalisable: one query is not an evaluation.

| Mode | hit@1 | MRR | Mean latency |
|---|---|---|---|
| sparse (BM25) | 0.75 | 0.834 | 11 ms |
| hybrid (no rerank) | 0.93 | 0.952 | 25 ms |
| dense (bge-m3) | 0.88 | 0.929 | 191 ms |
| hybrid + rerank | 0.93 | 0.963 | 3456 ms |

**Caveat:** 40 questions over 94 papers is a small evaluation, and the gold set
was written by the same person who built the retriever. hit@5 is saturated at
1.00 for three of four configurations, so only hit@1 and MRR discriminate.

---

## Generation and citation validation

This is the part that distinguishes the system from a generic RAG pipeline.

**The problem with "please cite your sources".** It is an unenforced request.
Models emit plausible-looking references, and nothing downstream checks them.

**The mechanism here:**

1. Each retrieved chunk is assigned a per-turn marker (`c1`, `c2`, …) at the
   boundary between retrieval and generation, so the set shown to the model is
   exactly the set the validator checks.
2. The system prompt states the valid range explicitly ("you have been given 8
   excerpts, numbered c1 to c8") and requires a marker on every factual
   sentence.
3. After generation, **every emitted marker is resolved against that set.**
   Unresolvable markers are stripped from the answer and counted.
4. Two rates are reported per answer:
   - `validity_rate` — share of citations that resolve to a real retrieved chunk
   - `grounded_rate` — share of claim-bearing sentences carrying any citation

Both are surfaced in the UI, not just logged. A user can see when a citation was
rejected.

**Marker parsing is permissive about form, strict about identity.** Local models
are inconsistent formatters — `[^c1]`, `[c1]`, `(c1)`, `[^c1, c3]` all occur —
so all are accepted and normalised. What is never relaxed is whether `c1`
corresponds to a chunk that was actually retrieved this turn.

**Validation is terminal.** Generated text is never returned directly. When
streaming, raw text is shown live for responsiveness and replaced by the
validated text once the stream completes — the only correct order, since
validation needs the whole answer.

### What this does *not* catch

`validity_rate` measures whether each marker **resolves to a chunk that was
retrieved this turn**. It does not verify that the cited chunk actually supports
the sentence attached to it. A model that cites a real excerpt for a claim that
excerpt does not make scores 100%.

This is not hypothetical. During browser testing the system produced a
confident, fully-cited answer whose central claim — "one study reports a 40%
reduction in lateral movement" — was **copied from the prompt's own formatting
example**, where the figure had been invented as filler. Every marker was valid,
so the answer scored 100% validity and 100% grounding.

Two changes followed:

1. The formatting example is now content-free placeholders
   (`<a claim taken from the first excerpt> [^c1]`), so there is nothing
   plausible to plagiarise. A realistic example is an instruction to copy it.
2. This limitation is documented rather than papered over.

Closing the gap properly requires entailment checking — a second pass asking
whether chunk *N* actually supports sentence *M*. That roughly doubles inference
cost per answer and is the natural next increment. What exists today eliminates
*fabricated sources*, which is the more common and more damaging failure; it
does not yet eliminate *misattributed claims*.

### Multi-turn

Follow-ups ("how does that compare to the second approach?") are meaningless to
a retriever alone. Each turn is rewritten into a standalone query against recent
history before retrieval. Rewriting failures fall back to the original question:
a degraded query beats a failed turn.

---

## Why a custom implementation instead of LangChain or LlamaIndex

Both were considered. The two hard requirements here are **citation fidelity**
and **incremental ingest without rebuild** — and both are things frameworks
abstract in ways that make them harder to guarantee.

- Citation validation needs the exact set of chunks put in front of the model,
  matched against the exact markers it emitted. That is a tight coupling between
  retrieval output and post-generation checking, and it is the whole value
  proposition here.
- Incremental ingest needs deterministic chunk IDs and a content-hash ledger.
  Framework ingestion helpers generally assume a rebuild.
- A local install with fewer transitive dependencies is easier to audit — which
  matters more than usual for a security-focused tool.

The cost is roughly 200 lines of orchestration that a framework would have
provided. The benefit is that every stage is inspectable and the two guarantees
that matter are enforced in code we own.

---

## Provider abstraction

Generation sits behind a two-method interface (`complete`, `stream`). The
default is a local Ollama model — no API key, no per-query cost, "local
execution" satisfied literally. Adapters exist for Claude, GPT, and Gemini;
switching is one environment variable and changes nothing about retrieval,
prompting, or validation.

Keeping the interface minimal is deliberate: every provider supports exactly
these two operations, so none can leak provider-specific behaviour into the
rest of the system.
