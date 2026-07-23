# Alternative stacks worth considering

The delivered system follows the brief: Streamlit, ChromaDB, a local LLM, hybrid
retrieval with validated citations. This document argues two alternatives that
were considered and rejected *for the default build*, but which may suit a
personal research tool better depending on priorities.

Both are proposals, not implementations. The shipped system is the one in
`src/`.

---

## 1. Hermes + Ollama — a refusal problem specific to this corpus

### The problem

This corpus is cybersecurity research. It contains papers on malware behaviour,
reverse engineering, exploitation, evasion of dynamic analysis, and offensive
tooling — for example, in the indexed set:

- *Automatically Attacking Software Reverse Engineering AI Agents*
- *Unmasking the Shadows: Pinpoint the Implementations of Anti-Dynamic Analysis*
- *Malicious cryptography techniques for unreversable malware*
- *On the Reverse Engineering of the Citadel Botnet*

Instruction-tuned assistant models are aligned to be cautious about exactly this
material. The failure mode is not usually a hard refusal — it is quieter and
more corrosive: hedged summaries, omitted technical specifics, added safety
caveats the source never made, or a flat "I can't help with that" when asked to
explain a technique the paper describes in full.

**A research assistant that will not summarise a malware paper is broken for its
stated purpose.** The user already has the papers. They are asking the tool to
read faster than they can.

This is the one failure mode in this project that is genuinely domain-specific,
and it is not addressed anywhere in the brief.

### The proposal

[Nous Research's Hermes line](https://huggingface.co/NousResearch/Hermes-4-70B)
is post-trained explicitly for reduced refusal and neutral alignment while
retaining general instruction-following. Hermes 3 is available on Ollama
officially (3B / 8B / 70B / 405B); Hermes 4.3 (36B) exists as a community build.
On 24 GB of unified memory, Hermes 3 8B runs comfortably alongside the embedding
and reranker models.

Because the generation layer here is already provider-agnostic
(`src/seclit/generate/providers/`), switching is a one-line configuration
change:

```bash
SECLIT_OLLAMA_MODEL=hermes3:8b uv run streamlit run app/streamlit_app.py
```

No change to retrieval, prompting, or citation validation.

### The honest caveat

The widely cited figure — roughly 57% refusals for Hermes versus about 84% for
GPT-4o on RefusalBench — **comes from Nous Research themselves**, not an
independent evaluation. It should be read as directionally indicative, not
established. Anyone quoting it as settled fact has not checked its provenance.

### How to settle it properly

Rather than argue from a vendor benchmark, measure it on the corpus that
matters. The proposed test:

1. Take ~20 questions drawn from the malware-analysis and offensive-security
   papers already indexed — the kind a security researcher would actually ask
   ("how does this sample detect that it is being debugged?").
2. Run each through `gemma3:12b`, `llama3.1:8b`, and `hermes3:8b` with identical
   retrieval context.
3. Score each answer on refusal, hedging, and technical completeness relative to
   what the retrieved excerpts actually say.

This is roughly an afternoon of work and produces a defensible answer for *this*
corpus rather than a general claim. It also composes with the existing citation
metrics: a model that refuses scores poorly on grounding, because it is not
using the context it was given.

**Recommendation:** run the benchmark before committing. If refusal turns out
not to be a practical problem for the questions actually being asked, stay on
`gemma3:12b`, which is stronger at general synthesis.

---

## 2. Obsidian + Claude — a knowledge-management argument

### The observation

The brief describes "a personal research tool, not an enterprise application"
for a single user. That framing has an implication worth taking seriously: the
value of a personal research library compounds through *annotation and
connection*, not just retrieval.

A Streamlit chat window is stateless. Ask a question, get an answer, close the
tab, and nothing accumulates. The user's own thinking — the note that paper A
contradicts paper B, the observation that a technique keeps reappearing — has
nowhere to live.

### The proposal

Convert each PDF into a Markdown note with YAML frontmatter in an
[Obsidian](https://obsidian.md) vault:

```markdown
---
arxiv_id: 2309.03582
title: "Zero Trust: Applications, Challenges, and Opportunities"
authors: [...]
year: 2023
subtopic: zero_trust
---

## Abstract
...

## 3 Threat Model
...
```

Obsidian then provides the human layer: full-text search, backlinks between
papers, graph view, and the user's own margin notes stored alongside the source.
Claude reaches the vault through an MCP server — several mature options exist,
including
[fully local ones](https://lobehub.com/mcp/mthehang-obsidian-agentic-rag) that
run Ollama embeddings over ChromaDB underneath, which is the same retrieval
stack as this build.

**The durable advantage:** the knowledge base outlives the application. Markdown
files in a folder are readable in thirty years. A Chroma collection is readable
as long as the ingestion code still runs.

### The honest tradeoff

This deviates from the brief. It replaces the Streamlit UI, which was an
explicit requirement, and moves the vector store behind an MCP server rather
than exposing it directly. The brief invites justifying a different *vector
database*; it does not invite replacing the interface.

It also assumes the user already works in Obsidian, or wants to. For someone who
does not, this adds a tool to learn in exchange for benefits they may not want.

**Recommendation:** treat this as complementary rather than a substitute. The
ingestion pipeline in `src/seclit/ingest/` already produces exactly the
structured, page-attributed text a vault export needs — adding a
`seclit export --obsidian` command is a small change on top of what exists. That
gets both: the delivered Streamlit tool for question answering, and a portable
Markdown library that survives it.

---

## Summary

| Option | Strongest when | Main cost |
|---|---|---|
| **Delivered build** (Streamlit + Chroma + local LLM) | You want exactly what the brief specifies, running locally today | Chat is stateless; nothing accumulates between sessions |
| **Hermes + Ollama** | The corpus is offensive-security heavy and hedged answers are the main pain | Refusal claims are vendor-reported; needs a real benchmark first |
| **Obsidian + Claude** | The library is long-lived and the user annotates as they read | Departs from the specified UI; assumes Obsidian is wanted |

The Hermes question is worth resolving empirically before launch. The Obsidian
option is best added later as an export path rather than chosen instead.
