"""Shared fixtures.

Ingestion tests run against a real Chroma store and a real SQLite catalog in a
temp directory, but with embeddings stubbed out. The logic under test —
deduplication, chunk-ID stability, upsert semantics — is independent of vector
quality, and stubbing keeps the suite fast enough to run on every edit.
"""

from __future__ import annotations

import numpy as np
import pytest

from seclit.config import Settings


@pytest.fixture
def config(tmp_path) -> Settings:
    """Settings pointed at an isolated temp directory."""
    settings = Settings(data_dir=tmp_path / "data", device="cpu")
    settings.ensure_dirs()
    return settings


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Replace the embedding model with a deterministic hash-based stub.

    Deterministic so identical text always yields an identical vector, which is
    what makes re-ingestion assertions meaningful.
    """

    def fake_embed(texts, config=None, **kwargs):
        dim = 1024
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vector = rng.standard_normal(dim).astype(np.float32)
            vectors[i] = vector / np.linalg.norm(vector)
        return vectors

    def fake_token_counter(config=None):
        return lambda text: len(text.split())

    monkeypatch.setattr("seclit.ingest.pipeline.embed_texts", fake_embed)
    monkeypatch.setattr("seclit.ingest.pipeline.token_counter", fake_token_counter)
    return fake_embed


def write_pdf(path, pages: list[str]) -> None:
    """Create a real multi-page PDF so extraction runs its actual code path."""
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def make_pdf(tmp_path):
    def _make(name: str, pages: list[str] | None = None):
        pages = pages or [
            "Introduction. " + " ".join(f"Sentence {i} about network security." for i in range(40)),
            "Evaluation. " + " ".join(f"Result {i} shows detection improves." for i in range(40)),
        ]
        path = tmp_path / name
        write_pdf(path, pages)
        return path

    return _make
