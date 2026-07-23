"""Central configuration.

Every knob is env-overridable with the ``SECLIT_`` prefix, so the app can be
retargeted (different provider, different corpus directory) without code edits.
See ``.env.example``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

Provider = Literal["ollama", "anthropic", "openai", "gemini"]

# The six areas named in the project brief. Corpus selection is balanced across
# these so coverage is demonstrable rather than asserted.
SUBTOPICS: dict[str, str] = {
    "network_security": "network security intrusion detection firewall traffic analysis",
    "malware_analysis": "malware analysis binary reverse engineering ransomware botnet",
    "threat_intelligence": "threat intelligence attack attribution APT indicators of compromise",
    "cloud_security": "cloud security container kubernetes serverless multi-tenant isolation",
    "zero_trust": "zero trust architecture continuous authentication microsegmentation access control",
    "vulnerability_management": "vulnerability detection patch management CVE fuzzing static analysis",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SECLIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Paths -------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")

    # ---- Models ------------------------------------------------------------
    # bge-m3: MIT licensed, 1024-dim, 8192-token context. The long context is
    # what lets a whole paper section stay in one chunk.
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    # Cross-encoder rerank is the single biggest precision lever in this stack.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # "mps" on Apple Silicon, "cuda", or "cpu". None = autodetect.
    device: str | None = None

    # ---- Chunking ----------------------------------------------------------
    chunk_target_tokens: int = 800
    chunk_overlap_ratio: float = 0.15
    chunk_min_tokens: int = 120

    # ---- Retrieval ---------------------------------------------------------
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    # Reranking dominates query latency. These defaults come from a sweep over
    # the 40-question gold set, not from a single spot check:
    #
    #   max_length  candidates   hit@1   MRR     latency
    #   512         24           0.82    0.912   2184 ms
    #   2048        24           0.93    0.963   3556 ms
    #   8192        24           0.93    0.963   3713 ms
    #   8192        40           0.95    0.975  10762 ms
    #
    # 512 truncates mid-chunk and scores worse than no reranking at all
    # (hybrid alone: MRR 0.952). 2048 captures the full benefit — 8192 adds
    # latency for nothing, since chunks rarely exceed 2048 tokens. Widening the
    # pool to 40 buys a further +0.012 MRR for 3x the latency, which is not a
    # trade worth making interactively; raise SECLIT_RERANK_CANDIDATES for
    # batch use where latency does not matter.
    rerank_candidates: int = 24
    rerank_max_length: int = 2048
    final_top_k: int = 8
    # dense | sparse | hybrid | hybrid_rerank. Reranking is the default because
    # generation dominates a full turn, but "hybrid" is a defensible choice —
    # on the gold set it gives up only 0.010 MRR for ~137x lower latency.
    retrieval_mode: str = "hybrid_rerank"
    # Chunks scoring below this after reranking are dropped. Prevents padding
    # the context with weak matches, which is a common hallucination source.
    #
    # bge-reranker-v2-m3 emits roughly [0, 1] relevance, not raw logits — an
    # earlier logit-scale default of -6.0 silently disabled this filter.
    # Measured separation on this corpus:
    #   in-corpus  ("zero trust lateral movement")  0.67 - 0.96
    #   off-topic  ("capital of France")            0.01 - 0.53
    #   nonsense   ("best pizza toppings")          0.00 - 0.00
    # 0.30 clears nonsense entirely and most off-topic chunks, while leaving
    # genuine matches untouched.
    rerank_score_floor: float = 0.30

    # ---- Generation --------------------------------------------------------
    provider: Provider = "ollama"
    ollama_model: str = "gemma3:12b"
    ollama_host: str = "http://localhost:11434"
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-2.0-flash"
    max_output_tokens: int = 2048
    # Turns of prior conversation fed to the query rewriter.
    history_window: int = 6

    # ---- Corpus ------------------------------------------------------------
    corpus_size: int = 100
    # arXiv asks for no more than one request per three seconds on a single
    # connection. Exceeding it gets you IP-blocked mid-build.
    arxiv_request_delay: float = 3.0
    arxiv_api_url: str = "https://export.arxiv.org/api/query"

    # ---- Derived paths -----------------------------------------------------
    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "corpus_manifest.jsonl"

    @property
    def catalog_path(self) -> Path:
        """SQLite ledger of ingested papers — drives incremental ingest."""
        return self.data_dir / "catalog.db"

    @property
    def collection_name(self) -> str:
        return "seclit_chunks"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.pdf_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


settings = Settings()
