# Retrieval evaluation

Corpus: **94 papers**, **2355 chunks**. Gold set: **40 questions** with known source papers.

Scored at paper granularity: a question counts as answered if a correct paper appears in the top *k*. MRR is the mean reciprocal rank of the first correct paper, which separates "ranked first" from "ranked eighth".

| Configuration | hit@1 | hit@3 | hit@5 | hit@10 | MRR | latency |
|---|---|---|---|---|---|---|
| Dense only (bge-m3) | 0.88 | 1.00 | 1.00 | 1.00 | 0.929 | 191 ms |
| Sparse only (BM25) | 0.75 | 0.93 | 0.95 | 0.95 | 0.834 | 11 ms |
| Hybrid (RRF fusion) | 0.93 | 0.97 | 1.00 | 1.00 | 0.952 | 25 ms |
| Hybrid + cross-encoder rerank | 0.93 | 1.00 | 1.00 | 1.00 | 0.963 | 3456 ms |

## Reading the table

**hit@3 and above are saturated** — every configuration retrieves a correct paper, so those columns cannot separate them. The informative columns are hit@1 and MRR, which measure *ordering*: whether the right paper is first or fourth. That is what the user sees.

**Hybrid fusion is the clear win.** Adding BM25 and fusing by reciprocal rank moves hit@1 by **+0.05** over dense-only while *reducing* latency by 166 ms (BM25 is nearly free, and it lets the dense retriever return fewer candidates). BM25 earns its place on questions containing rare literals — CVE identifiers, protocol names like RPL, technique names like ADASYN — where embeddings blur near-neighbours together.

**Cross-encoder reranking is marginal on this gold set, and the table should say so.** It adds **+0.010** MRR and **+0.00** hit@1 over hybrid alone, for roughly **137x** the latency (3456 ms vs 25 ms). It remains the default because generation dominates a full turn — a few seconds of reranking is a small share of a local model's response time — but on this evidence `SECLIT_RETRIEVAL_MODE=hybrid` is a defensible choice for anyone who wants sub-100 ms retrieval.

The honest caveat: 40 questions over 94 papers is a small evaluation, and the gold set was written by the same person who built the retriever. A larger corpus would likely widen the gap in reranking's favour, since more papers means more near-duplicate candidates for it to disambiguate — but that is a prediction, not a measurement.
