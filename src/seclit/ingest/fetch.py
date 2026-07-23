"""arXiv corpus acquisition.

Builds a subtopic-balanced corpus from arXiv's ``cs.CR`` category (50k+ papers)
and downloads the PDFs.

**Rate limiting is not optional.** arXiv asks for at most one request every
three seconds on a single connection. Exceeding it gets the client IP blocked,
which stops a build dead. Every network call in this module goes through
``_throttle``; there is no unthrottled path.

Selection is balanced across the six areas in the brief rather than taking the
top-N of one broad query, so the corpus demonstrably covers each area instead of
over-indexing whichever topic happens to dominate recent submissions.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

from seclit.config import SUBTOPICS, Settings, settings
from seclit.ingest.extract import sha256_file
from seclit.models import PaperMeta

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

USER_AGENT = "seclit-rag/0.1 (research assistant; contact via repository)"


@dataclass
class FetchReport:
    requested: int
    resolved: int
    downloaded: int
    skipped: int
    failed: list[tuple[str, str]]

    def summary(self) -> str:
        lines = [
            f"resolved {self.resolved}/{self.requested} papers",
            f"downloaded {self.downloaded}, skipped {self.skipped} (already present)",
        ]
        if self.failed:
            lines.append(f"failed {len(self.failed)}:")
            lines += [f"  - {pid}: {err}" for pid, err in self.failed[:10]]
        return "\n".join(lines)


class ArxivClient:
    """Throttled arXiv API client. One connection, one request per delay."""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or settings
        self._last_request = 0.0
        self._client = httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        wait = self.config.arxiv_request_delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- metadata ------------------------------------------------------------

    def search(self, query: str, max_results: int = 25) -> list[PaperMeta]:
        """Run one arXiv API query and parse the Atom response."""
        self._throttle()
        response = self._client.get(
            self.config.arxiv_api_url,
            params={
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        response.raise_for_status()
        return self._parse_feed(response.text)

    @staticmethod
    def _parse_feed(xml_text: str) -> list[PaperMeta]:
        root = ET.fromstring(xml_text)
        papers: list[PaperMeta] = []

        for entry in root.findall(f"{ATOM}entry"):
            raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
            if not raw_id:
                continue
            # "http://arxiv.org/abs/2401.12345v2" -> "2401.12345"
            arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0]

            title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
            abstract = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
            authors = [
                " ".join((a.findtext(f"{ATOM}name") or "").split())
                for a in entry.findall(f"{ATOM}author")
            ]
            published = entry.findtext(f"{ATOM}published") or ""
            year = int(published[:4]) if published[:4].isdigit() else None

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            for link in entry.findall(f"{ATOM}link"):
                if link.get("title") == "pdf" and link.get("href"):
                    pdf_url = link.get("href", pdf_url)

            papers.append(
                PaperMeta(
                    paper_id=arxiv_id,
                    title=title,
                    authors=[a for a in authors if a],
                    year=year,
                    abstract=abstract,
                    pdf_url=pdf_url,
                )
            )
        return papers

    # -- PDFs ----------------------------------------------------------------

    def download_pdf(self, paper: PaperMeta, dest: Path) -> Path:
        """Download one PDF. Returns the path; raises on non-PDF responses."""
        self._throttle()
        target = dest / f"{paper.paper_id}.pdf"

        with self._client.stream("GET", paper.pdf_url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                raise ValueError(f"expected PDF, got '{content_type}'")

            tmp = target.with_suffix(".pdf.part")
            with open(tmp, "wb") as handle:
                for block in response.iter_bytes(1 << 16):
                    handle.write(block)

        # Sanity check before committing the file — a truncated or HTML error
        # page saved as .pdf poisons the index later.
        with open(tmp, "rb") as handle:
            if handle.read(5) != b"%PDF-":
                tmp.unlink(missing_ok=True)
                raise ValueError("downloaded file is not a valid PDF")

        tmp.replace(target)
        return target


def select_corpus(
    total: int = 100,
    config: Settings | None = None,
    client: ArxivClient | None = None,
) -> list[PaperMeta]:
    """Select a subtopic-balanced set of papers.

    Over-fetches per subtopic because deduplication across topics (papers often
    match several) would otherwise leave later subtopics short.
    """
    config = config or settings
    owned = client is None
    client = client or ArxivClient(config)

    # Ceiling division: floor would leave the corpus short of `total` whenever
    # the count doesn't divide evenly across subtopics.
    per_topic = max(1, -(-total // len(SUBTOPICS)))
    seen: set[str] = set()
    selected: list[PaperMeta] = []

    try:
        for subtopic, keywords in SUBTOPICS.items():
            terms = " OR ".join(f'abs:"{w}"' for w in keywords.split()[:4])
            query = f"cat:cs.CR AND ({terms})"

            try:
                results = client.search(query, max_results=per_topic * 3)
            except Exception as exc:  # noqa: BLE001 - one topic failing is survivable
                print(f"  ! query failed for {subtopic}: {exc}")
                continue

            taken = 0
            for paper in results:
                if paper.paper_id in seen:
                    continue
                seen.add(paper.paper_id)
                paper.subtopic = subtopic
                selected.append(paper)
                taken += 1
                if taken >= per_topic:
                    break
            print(f"  {subtopic:26s} {taken:3d} papers")
    finally:
        if owned:
            client.close()

    return selected[:total]


def write_manifest(papers: list[PaperMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper.to_json(), ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[PaperMeta]:
    if not path.exists():
        return []
    papers: list[PaperMeta] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                papers.append(PaperMeta.from_json(json.loads(line)))
    return papers


def fetch_corpus(
    total: int | None = None,
    config: Settings | None = None,
    *,
    refresh_manifest: bool = False,
) -> FetchReport:
    """Select papers, download PDFs, and write the manifest.

    Re-runnable: existing PDFs are skipped, so an interrupted download resumes
    rather than starting over.
    """
    config = config or settings
    config.ensure_dirs()
    total = total or config.corpus_size

    existing = read_manifest(config.manifest_path)
    if existing and not refresh_manifest:
        print(f"Using existing manifest ({len(existing)} papers)")
        papers = existing
    else:
        print(f"Selecting {total} papers across {len(SUBTOPICS)} subtopics...")
        papers = select_corpus(total, config)
        write_manifest(papers, config.manifest_path)
        print(f"Wrote manifest: {config.manifest_path}")

    downloaded = skipped = 0
    failed: list[tuple[str, str]] = []

    with ArxivClient(config) as client:
        for pos, paper in enumerate(papers, start=1):
            target = config.pdf_dir / f"{paper.paper_id}.pdf"
            if target.exists() and target.stat().st_size > 1024:
                skipped += 1
                if not paper.sha256:
                    paper.sha256 = sha256_file(target)
                continue

            try:
                path = client.download_pdf(paper, config.pdf_dir)
                paper.sha256 = sha256_file(path)
                downloaded += 1
                print(f"  [{pos:3d}/{len(papers)}] {paper.paper_id}  {paper.title[:58]}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed.append((paper.paper_id, str(exc)))
                print(f"  [{pos:3d}/{len(papers)}] FAILED {paper.paper_id}: {exc}")

    # Rewrite the manifest so it carries the hashes, making the corpus verifiable.
    write_manifest(papers, config.manifest_path)

    return FetchReport(
        requested=total,
        resolved=len(papers),
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
    )
