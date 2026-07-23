"""Embedding via BAAI/bge-m3.

Chosen for three reasons specific to this corpus:

* **8192-token context.** Academic sections are long. A model capped at 512
  tokens would force chunks small enough to sever arguments from their premises.
* **MIT licence.** No redistribution friction for a delivered artifact.
* **1024 dimensions.** Enough capacity for technical prose without inflating the
  index.

bge-m3 needs no instruction prefix on either side, unlike the bge-v1.5 family
which requires a query prefix. Adding one here would *degrade* retrieval, so the
symmetry is deliberate rather than an oversight.

The model is loaded lazily and cached process-wide — importing this module must
stay cheap so tests and CLI help don't pay a model load.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from seclit.config import Settings, settings

_model_lock = threading.Lock()
_model_cache: dict[tuple[str, str], object] = {}


def get_embedder(config: Settings | None = None):
    """Return a cached SentenceTransformer, loading it on first use."""
    config = config or settings
    device = config.resolve_device()
    key = (config.embedding_model, device)

    if key not in _model_cache:
        with _model_lock:
            if key not in _model_cache:
                from sentence_transformers import SentenceTransformer

                _model_cache[key] = SentenceTransformer(config.embedding_model, device=device)
    return _model_cache[key]


def embed_texts(
    texts: Sequence[str],
    config: Settings | None = None,
    *,
    batch_size: int = 8,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed passages. Returns L2-normalised vectors, so dot product == cosine."""
    if not texts:
        return np.zeros((0, (config or settings).embedding_dim), dtype=np.float32)

    model = get_embedder(config)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(text: str, config: Settings | None = None) -> np.ndarray:
    """Embed a single query. Symmetric with ``embed_texts`` by design."""
    return embed_texts([text], config)[0]


def token_counter(config: Settings | None = None):
    """Expose the model's real tokenizer for chunk sizing.

    Chunking defaults to a word-count heuristic so it stays fast and
    dependency-free; the ingestion pipeline swaps in this exact tokenizer so
    chunk sizes match what the model actually sees.
    """
    model = get_embedder(config)
    tokenizer = model.tokenizer

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
