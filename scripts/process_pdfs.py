"""Process pending PDFs: extract text, clean, chunk, and persist."""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

from sqlalchemy import select

from app.db.models import Chunk as ChunkModel
from app.db.models import Document, DocumentStatus, Page
from app.db.session import session_scope
from app.pdf import cleaner, parser
from app.pdf.chunker import Chunk as ChunkSegment, chunk_pages
from app.storage import MinioStorage

LOGGER = logging.getLogger(__name__)


def _fetch_pending_document_ids(limit: int) -> List[str]:
    with session_scope() as session:
        stmt = select(Document.id).where(Document.status == DocumentStatus.NEW.value).limit(limit)
        return [str(row[0]) for row in session.execute(stmt)]


def _download_to_temp(storage: MinioStorage, s3_key: str) -> Path:
    with NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)
    storage.download_file(storage.raw_bucket, s3_key, temp_path)
    return temp_path


def _persist_pages(session, document: Document, raw_pages: List[str], clean_pages: List[str]) -> None:
    session.query(Page).filter(Page.document_id == document.id).delete(synchronize_session=False)
    for index, (raw_text, clean_text) in enumerate(zip(raw_pages, clean_pages, strict=True), start=1):
        session.add(
            Page(
                document_id=document.id,
                page_number=index,
                raw_text=raw_text,
                clean_text=clean_text,
            )
        )


def _persist_chunks(session, document: Document, segments: List[ChunkSegment]) -> None:
    session.query(ChunkModel).filter(ChunkModel.document_id == document.id).delete(synchronize_session=False)
    for segment in segments:
        session.add(
            ChunkModel(
                document_id=document.id,
                page_start=segment.page_start,
                page_end=segment.page_end,
                chunk_index=segment.chunk_index,
                text=segment.text,
            )
        )


def process_document(document_id: str, storage: MinioStorage) -> None:
    temp_path: Path | None = None
    try:
        with session_scope() as session:
            document = session.get(Document, document_id)
            if not document:
                LOGGER.warning("Document %s not found", document_id)
                return
            LOGGER.info("Processing document %s (%s)", document_id, document.title)
            temp_path = _download_to_temp(storage, document.s3_key)
            raw_pages = parser.extract_pages(temp_path)
            clean_pages = [cleaner.clean_text(page) for page in raw_pages]
            _persist_pages(session, document, raw_pages, clean_pages)
            document.status = DocumentStatus.TEXT_EXTRACTED.value

            segments = chunk_pages(clean_pages)
            _persist_chunks(session, document, segments)
            document.status = DocumentStatus.CHUNKS_BUILT.value

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to process %s: %s", document_id, exc)
        with session_scope() as session:
            document = session.get(Document, document_id)
            if document:
                document.status = DocumentStatus.FAILED.value
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def main(batch_size: int = 10) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    storage = MinioStorage()
    document_ids = _fetch_pending_document_ids(batch_size)
    if not document_ids:
        LOGGER.info("No NEW documents to process.")
        return
    for document_id in document_ids:
        process_document(document_id, storage)
    LOGGER.info("Processed %d documents.", len(document_ids))


if __name__ == "__main__":
    main()
