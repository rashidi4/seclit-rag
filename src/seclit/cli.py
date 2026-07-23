"""Command-line interface.

seclit fetch     # download the corpus from arXiv (rate-limited)
seclit index     # ingest every PDF into the vector store
seclit add FILE  # add one paper without rebuilding
seclit stats     # corpus and index statistics
seclit ask "..." # one-shot question from the terminal
seclit eval      # retrieval evaluation across modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seclit.config import settings


def _cmd_fetch(args: argparse.Namespace) -> int:
    from seclit.ingest.fetch import fetch_corpus

    report = fetch_corpus(total=args.count, refresh_manifest=args.refresh)
    print()
    print(report.summary())
    return 0 if not report.failed else 1


def _cmd_index(args: argparse.Namespace) -> int:
    from seclit.ingest.fetch import read_manifest
    from seclit.ingest.pipeline import IngestPipeline

    pipeline = IngestPipeline()
    summary = pipeline.ingest_corpus(
        manifest=read_manifest(settings.manifest_path), force=args.force
    )
    print()
    print(summary.summary())
    print()
    _print_stats(pipeline.stats())
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    from seclit.ingest.pipeline import IngestPipeline

    path = Path(args.pdf)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    pipeline = IngestPipeline()
    before = pipeline.store.count()
    result = pipeline.ingest_pdf(path, force=args.force)
    after = pipeline.store.count()

    if result.status == "failed":
        print(f"Failed: {result.error}", file=sys.stderr)
        return 1
    if result.status == "skipped":
        print(f"Already indexed (identical content hash): {result.paper_id}")
        return 0

    print(
        f"Added {result.paper_id}: {result.n_chunks} chunks from {result.n_pages} pages "
        f"({before} -> {after} vectors, no rebuild)"
    )
    return 0


def _print_stats(stats: dict) -> None:
    print(f"papers   {stats['papers']}")
    print(f"chunks   {stats['chunks']}")
    print(f"vectors  {stats['vectors']}")
    print(f"pages    {stats['pages']}")
    if stats.get("by_subtopic"):
        print("\nby subtopic:")
        for topic, count in sorted(stats["by_subtopic"].items(), key=lambda kv: -kv[1]):
            print(f"  {topic:26s} {count:3d}")


def _cmd_stats(_: argparse.Namespace) -> int:
    from seclit.ingest.pipeline import IngestPipeline

    _print_stats(IngestPipeline().stats())
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from seclit.generate.chat import ChatEngine

    engine = ChatEngine()
    ok, detail = engine.provider.available()
    if not ok:
        print(f"Provider unavailable: {detail}", file=sys.stderr)
        return 2

    answer = engine.ask(args.question, mode=args.mode)
    print(answer.text)

    if answer.cited_chunks:
        print("\nSources")
        for chunk in answer.cited_chunks:
            print(f"  [^{chunk.marker}] {chunk.title} ({chunk.year}), {chunk.page_label}")

    if answer.report:
        print(
            f"\ncitation validity {answer.report.validity_rate:.0%} | "
            f"grounded {answer.report.grounded_rate:.0%}"
        )
        if answer.report.invalid_markers:
            print(f"stripped invalid markers: {answer.report.invalid_markers}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from seclit.evaluate import run_evaluation

    run_evaluation(Path(args.gold), write_report=args.write)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seclit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download corpus from arXiv")
    p_fetch.add_argument("-n", "--count", type=int, default=settings.corpus_size)
    p_fetch.add_argument("--refresh", action="store_true", help="re-select papers")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_index = sub.add_parser("index", help="ingest all PDFs")
    p_index.add_argument("--force", action="store_true", help="re-ingest existing papers")
    p_index.set_defaults(func=_cmd_index)

    p_add = sub.add_parser("add", help="add one PDF without rebuilding")
    p_add.add_argument("pdf")
    p_add.add_argument("--force", action="store_true")
    p_add.set_defaults(func=_cmd_add)

    sub.add_parser("stats", help="corpus statistics").set_defaults(func=_cmd_stats)

    p_ask = sub.add_parser("ask", help="one-shot question")
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--mode",
        default="hybrid_rerank",
        choices=["dense", "sparse", "hybrid", "hybrid_rerank"],
    )
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="run retrieval evaluation")
    p_eval.add_argument("--gold", default="eval/gold_set.jsonl")
    p_eval.add_argument("--write", action="store_true", help="write eval/results.md")
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
