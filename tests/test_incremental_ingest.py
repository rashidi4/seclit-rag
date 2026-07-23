"""Incremental ingest tests.

The brief requires adding papers "without rebuilding the entire database". These
tests hold the pipeline to that literally:

* re-running ingestion must not change the index at all;
* adding one paper must change *only* that paper's chunks;
* the same content under a different filename must not be indexed twice.

Each is a specific way a naive implementation silently corrupts an index —
duplicate vectors inflating retrieval, or a rebuild quietly discarding
previously added papers.
"""

from __future__ import annotations

from seclit.ingest.pipeline import IngestPipeline
from seclit.models import PaperMeta


def test_single_ingest_creates_chunks(config, stub_embeddings, make_pdf):
    pipeline = IngestPipeline(config)
    result = pipeline.ingest_pdf(make_pdf("a.pdf"))

    assert result.status == "ingested"
    assert result.n_chunks > 0
    assert pipeline.store.count() == result.n_chunks


def test_reingest_is_idempotent(config, stub_embeddings, make_pdf):
    """Running ingestion twice must leave the index byte-identical."""
    pipeline = IngestPipeline(config)
    pdf = make_pdf("a.pdf")

    pipeline.ingest_pdf(pdf)
    count_after_first = pipeline.store.count()
    ids_after_first = pipeline.store.chunk_ids()

    second = pipeline.ingest_pdf(pdf)

    assert second.status == "skipped"
    assert pipeline.store.count() == count_after_first
    assert pipeline.store.chunk_ids() == ids_after_first


def test_forced_reingest_does_not_duplicate(config, stub_embeddings, make_pdf):
    """Even with --force, deterministic chunk IDs mean upsert overwrites."""
    pipeline = IngestPipeline(config)
    pdf = make_pdf("a.pdf")

    pipeline.ingest_pdf(pdf)
    baseline = pipeline.store.count()

    pipeline.ingest_pdf(pdf, force=True)

    assert pipeline.store.count() == baseline


def test_adding_a_paper_leaves_existing_chunks_untouched(config, stub_embeddings, make_pdf):
    """The core incremental guarantee: no rebuild, no rewrite."""
    pipeline = IngestPipeline(config)

    first = pipeline.ingest_pdf(make_pdf("first.pdf"))
    ids_before = pipeline.store.chunk_ids()
    count_before = pipeline.store.count()

    second = pipeline.ingest_pdf(
        make_pdf("second.pdf", ["Zero trust architecture. " + "Detail. " * 200])
    )

    assert second.status == "ingested"
    # Exactly the new chunks were added.
    assert pipeline.store.count() == count_before + second.n_chunks
    # Every pre-existing chunk ID survives unchanged.
    assert ids_before.issubset(pipeline.store.chunk_ids())
    assert first.n_chunks == len([i for i in ids_before if i.startswith("first::")])


def test_identical_content_under_a_different_name_is_skipped(
    config, stub_embeddings, make_pdf, tmp_path
):
    """Deduplication is by content hash, not filename."""
    import shutil

    pipeline = IngestPipeline(config)
    original = make_pdf("paper.pdf")
    pipeline.ingest_pdf(original)
    baseline = pipeline.store.count()

    copy = tmp_path / "paper-copy.pdf"
    shutil.copy(original, copy)
    result = pipeline.ingest_pdf(copy)

    assert result.status == "skipped"
    assert pipeline.store.count() == baseline


def test_catalog_tracks_papers_and_subtopics(config, stub_embeddings, make_pdf):
    pipeline = IngestPipeline(config)
    pipeline.ingest_pdf(
        make_pdf("net.pdf"),
        PaperMeta(paper_id="net", title="Network Paper", subtopic="network_security"),
    )
    pipeline.ingest_pdf(
        make_pdf("mal.pdf", ["Malware analysis. " + "Detail. " * 200]),
        PaperMeta(paper_id="mal", title="Malware Paper", subtopic="malware_analysis"),
    )

    stats = pipeline.stats()

    assert stats["papers"] == 2
    assert stats["vectors"] == stats["chunks"]
    assert stats["by_subtopic"] == {"network_security": 1, "malware_analysis": 1}


def test_deleting_a_paper_removes_only_its_chunks(config, stub_embeddings, make_pdf):
    pipeline = IngestPipeline(config)
    pipeline.ingest_pdf(make_pdf("keep.pdf"))
    keep_ids = pipeline.store.chunk_ids()

    drop = pipeline.ingest_pdf(make_pdf("drop.pdf", ["Other content. " + "Word. " * 200]))
    assert drop.status == "ingested"

    pipeline.store.delete_paper("drop")

    assert pipeline.store.chunk_ids() == keep_ids


def test_corrupt_pdf_fails_without_halting(config, stub_embeddings, tmp_path):
    """A bad file must be reported, not raise — one bad PDF cannot stop a batch."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")

    result = IngestPipeline(config).ingest_pdf(broken)

    assert result.status == "failed"
    assert result.error


def test_batch_ingest_reports_per_paper_outcomes(config, stub_embeddings, make_pdf, tmp_path):
    pdf_dir = config.pdf_dir
    for name, body in [
        ("p1.pdf", "Network security content. " + "Detail. " * 200),
        ("p2.pdf", "Cloud security content. " + "Detail. " * 200),
    ]:
        source = make_pdf(name, [body])
        (pdf_dir / name).write_bytes(source.read_bytes())
    (pdf_dir / "bad.pdf").write_bytes(b"garbage")

    summary = IngestPipeline(config).ingest_corpus(verbose=False)

    assert summary.ingested == 2
    assert len(summary.failed) == 1
    assert summary.total_chunks > 0
