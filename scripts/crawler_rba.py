"""Crawl Reserve Bank of Australia publication listings and ingest PDFs."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Iterator, List, Sequence
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests import Response

from app.db.models import Document, DocumentStatus
from app.db.session import session_scope
from app.storage import MinioStorage

BASE_URL = "https://www.rba.gov.au"
LOGGER = logging.getLogger(__name__)
USER_AGENT = os.environ.get("CRAWLER_USER_AGENT", "rba-doc-intel/0.1 (+https://github.com/ezprogramming)")
def _parse_year_filters(raw_filters: str) -> set[str]:
    filters: set[str] = set()
    current_year = date.today().year
    for token in raw_filters.split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith("+") and token[:-1].isdigit():
            start_year = int(token[:-1])
            for year in range(start_year, current_year + 1):
                filters.add(str(year))
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            if start_str.isdigit():
                start_year = int(start_str)
                end_year = int(end_str) if end_str.isdigit() else current_year
                if start_year > end_year:
                    start_year, end_year = end_year, start_year
                for year in range(start_year, end_year + 1):
                    filters.add(str(year))
                continue
        if token.isdigit():
            filters.add(token)
        else:
            LOGGER.warning("Ignoring invalid year filter token '%s'", token)
    return filters


YEAR_FILTERS = _parse_year_filters(os.getenv("CRAWLER_YEAR_FILTER", ""))


@dataclass(frozen=True)
class PublicationSource:
    """Configuration holder describing how to crawl a publication."""

    name: str
    doc_type: str
    index_url: str
    issue_pattern: re.Pattern[str] | None
    pdf_href_prefix: str


@dataclass
class IssueMetadata:
    url: str
    title: str
    publication_date: date | None


@dataclass
class PdfCandidate:
    pdf_url: str
    link_text: str | None
    issue: IssueMetadata


SOURCES: Sequence[PublicationSource] = (
    PublicationSource(
        name="Statement on Monetary Policy",
        doc_type="SMP",
        index_url=f"{BASE_URL}/publications/smp/",
        issue_pattern=re.compile(r"^/publications/smp/\d{4}/[a-z]{3}/$"),
        pdf_href_prefix="/publications/smp/",
    ),
    PublicationSource(
        name="Financial Stability Review",
        doc_type="FSR",
        index_url=f"{BASE_URL}/publications/fsr/",
        issue_pattern=re.compile(r"^/publications/fsr/\d{4}/[a-z]{3}/$"),
        pdf_href_prefix="/publications/fsr/",
    ),
    PublicationSource(
        name="RBA Annual Report",
        doc_type="ANNUAL_REPORT",
        index_url=f"{BASE_URL}/publications/annual-reports/rba/",
        issue_pattern=re.compile(r"^/publications/annual-reports/rba/\d{4}/$"),
        pdf_href_prefix="/publications/annual-reports/rba/",
    ),
)


def _fetch(url: str) -> Response:
    response = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response


def _extract_issue_links(index_html: str, pattern: re.Pattern[str] | None) -> List[str]:
    if pattern is None:
        return []
    soup = BeautifulSoup(index_html, "html.parser")
    links: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        if pattern.match(href):
            links.add(urljoin(BASE_URL, href))
    return sorted(links)


def _parse_issue_metadata(issue_url: str, issue_html: str) -> IssueMetadata:
    soup = BeautifulSoup(issue_html, "html.parser")
    title_tag = soup.find("meta", attrs={"name": "dcterms.title"})
    title = (
        title_tag["content"].strip()
        if title_tag and title_tag.get("content")
        else (soup.title.string.strip() if soup.title else issue_url)
    )

    publication_date = None
    date_tag = soup.find("meta", attrs={"name": "dc.date"})
    if date_tag and date_tag.get("content"):
        try:
            publication_date = datetime.strptime(date_tag["content"], "%Y-%m-%d").date()
        except ValueError:
            LOGGER.warning("Unable to parse publication date for %s", issue_url)

    return IssueMetadata(url=issue_url, title=title, publication_date=publication_date)


def _extract_pdf_candidates(issue_html: str, metadata: IssueMetadata, prefix: str) -> List[PdfCandidate]:
    soup = BeautifulSoup(issue_html, "html.parser")
    candidates: dict[str, PdfCandidate] = {}
    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        if ".pdf" not in href.lower():
            continue
        if prefix and not href.startswith(prefix):
            continue
        pdf_url = urljoin(BASE_URL, href)
        link_text = anchor.get_text(strip=True) or None
        candidates[pdf_url] = PdfCandidate(pdf_url=pdf_url, link_text=link_text, issue=metadata)
    return list(candidates.values())


def _download_pdf(source_url: str) -> Path:
    with NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)
    response = _fetch(source_url)
    try:
        with open(temp_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1_048_576):
                if chunk:
                    fh.write(chunk)
    finally:
        response.close()
    return temp_path


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _object_key(doc_type: str, publication_date: date | None, pdf_url: str) -> str:
    filename = Path(urlparse(pdf_url).path).name or "document.pdf"
    if publication_date:
        return f"raw/{doc_type.lower()}/{publication_date.year}/{filename}"
    return f"raw/{doc_type.lower()}/undated/{filename}"


def _document_exists(content_hash: str) -> bool:
    with session_scope() as session:
        return bool(
            session.query(Document.id)
            .filter(Document.content_hash == content_hash)
            .first()
        )


def _register_document(
    source: PublicationSource,
    candidate: PdfCandidate,
    object_key: str,
    content_hash: str,
    content_length: int,
) -> None:
    with session_scope() as session:
        title = candidate.issue.title
        if candidate.link_text and candidate.link_text.lower() not in {"download pdf", "pdf"}:
            title = f"{title} – {candidate.link_text}"
        document = Document(
            source_system=source.name,
            source_url=candidate.pdf_url,
            s3_key=object_key,
            doc_type=source.doc_type,
            title=title,
            publication_date=candidate.issue.publication_date,
            content_hash=content_hash,
            content_length=content_length,
            status=DocumentStatus.NEW.value,
        )
        session.add(document)


def ingest_source(source: PublicationSource, storage: MinioStorage) -> int:
    LOGGER.info("Scanning %s (%s)", source.name, source.index_url)
    try:
        index_resp = _fetch(source.index_url)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unable to fetch index %s: %s", source.index_url, exc)
        return 0

    issue_urls = _extract_issue_links(index_resp.text, source.issue_pattern)
    if not issue_urls:
        issue_urls = [source.index_url]
        filtered_notice = False
    else:
        filtered_notice = True
    if YEAR_FILTERS and filtered_notice:
        before = len(issue_urls)
        issue_urls = [
            url
            for url in issue_urls
            if any(f"/{year}/" in url for year in YEAR_FILTERS)
        ]
        LOGGER.info(
            "Applying year filter %s: %d -> %d issue pages",
            sorted(YEAR_FILTERS),
            before,
            len(issue_urls),
        )
        if not issue_urls:
            LOGGER.warning(
                "Year filter %s removed all issue pages for %s; skipping source",
                sorted(YEAR_FILTERS),
                source.doc_type,
            )
            return 0

    LOGGER.info("Found %d issue pages for %s", len(issue_urls), source.doc_type)

    ingested = 0
    for issue_url in issue_urls:
        try:
            issue_resp = _fetch(issue_url)
            metadata = _parse_issue_metadata(issue_url, issue_resp.text)
            candidates = _extract_pdf_candidates(issue_resp.text, metadata, source.pdf_href_prefix)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to parse issue %s: %s", issue_url, exc)
            continue

        for candidate in candidates:
            temp_path: Path | None = None
            try:
                temp_path = _download_pdf(candidate.pdf_url)
                content_hash, content_length = _sha256_file(temp_path)
                if _document_exists(content_hash):
                    LOGGER.debug("Duplicate detected for %s", candidate.pdf_url)
                    continue
                object_key = _object_key(source.doc_type, candidate.issue.publication_date, candidate.pdf_url)
                storage.upload_file(storage.raw_bucket, object_key, temp_path)
                _register_document(
                    source=source,
                    candidate=candidate,
                    object_key=object_key,
                    content_hash=content_hash,
                    content_length=content_length,
                )
                LOGGER.info("Registered %s (%s)", candidate.pdf_url, object_key)
                ingested += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to ingest %s: %s", candidate.pdf_url, exc)
            finally:
                if temp_path:
                    temp_path.unlink(missing_ok=True)
    return ingested


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    storage = MinioStorage()
    total_new = 0
    for source in SOURCES:
        total_new += ingest_source(source, storage)
    LOGGER.info("Crawler complete. New documents ingested: %d", total_new)


if __name__ == "__main__":
    main()
