"""Process pending PDFs: extract text, clean, chunk, and persist.

This script processes PDF documents in parallel using ThreadPoolExecutor:
- Downloads PDFs from MinIO
- Extracts text per page
- Cleans headers/footers
- Chunks text for RAG
- Persists to PostgreSQL

Performance:
- Sequential: ~1 doc/min
- Parallel (4 workers): ~4 docs/min (4x speedup)
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
                section_hint=segment.section_hint,
            )
        )


def process_document(document_id: str, storage: MinioStorage) -> None:
    """Process a single PDF document through the pipeline.

    Steps:
    1. Download PDF from MinIO to temp file
    2. Extract raw text per page
    3. Detect repeating headers/footers across all pages
    4. Clean each page (remove headers/footers, normalize whitespace)
    5. Save pages to database
    6. Chunk cleaned pages for RAG
    7. Save chunks to database
    8. Update document status

    Why detect headers/footers first?
    - RBA documents have consistent headers/footers across pages
    - Better to analyze all pages together than clean individually
    - Example: "Reserve Bank of Australia" may not match regex if slightly varied,
      but will be detected as repeating in 90% of pages
    """
    temp_path: Path | None = None
    try:
        with session_scope() as session:
            document = session.get(Document, document_id)
            if not document:
                LOGGER.warning("Document %s not found", document_id)
                return

            LOGGER.info("Processing document %s (%s)", document_id, document.title)

            # Download PDF from object storage
            temp_path = _download_to_temp(storage, document.s3_key)

            # Extract raw text per page
            raw_pages = parser.extract_pages(temp_path)

            # Detect repeating headers/footers across ALL pages
            # This catches patterns that regex might miss
            repeating_headers, repeating_footers = cleaner.detect_repeating_headers_footers(
                raw_pages
            )

            # Clean each page using both pattern-based and frequency-based detection
            clean_pages = [
                cleaner.clean_text(page, repeating_headers, repeating_footers)
                for page in raw_pages
            ]

            # Persist both raw and clean text to database
            # Why keep raw? Allows re-processing with improved cleaning later
            _persist_pages(session, document, raw_pages, clean_pages)
            document.status = DocumentStatus.TEXT_EXTRACTED.value

            # Chunk cleaned pages for RAG
            segments = chunk_pages(clean_pages)
            _persist_chunks(session, document, segments)
            document.status = DocumentStatus.CHUNKS_BUILT.value

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to process %s: %s", document_id, exc)
        # Mark document as failed so it doesn't get stuck in pipeline
        with session_scope() as session:
            document = session.get(Document, document_id)
            if document:
                document.status = DocumentStatus.FAILED.value
    finally:
        # Always cleanup temp file to avoid disk space issues
        if temp_path:
            temp_path.unlink(missing_ok=True)


def main(batch_size: int = 10, max_workers: int = 4) -> None:
    """Process pending PDFs in parallel batches.

    Args:
        batch_size: Number of documents to fetch per iteration
        max_workers: Number of parallel worker threads (4 recommended for I/O-bound tasks)

    How it works:
    1. Fetch batch_size document IDs from database (status=NEW)
    2. Submit all to ThreadPoolExecutor for parallel processing
    3. Each worker: downloads PDF → extracts text → cleans → chunks → saves
    4. Wait for all workers to complete before next batch
    5. Repeat until no pending documents remain

    Why parallel?
    - PDF processing is I/O-bound (downloading from MinIO, reading pages)
    - Multiple threads can download/process simultaneously
    - 4 workers typically gives 3-4x speedup on modern systems
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Create single storage instance (thread-safe MinIO client)
    storage = MinioStorage()
    total_processed = 0
    total_failed = 0

    LOGGER.info(f"Starting PDF processing with {max_workers} parallel workers")

    # Process documents in batches until none remain
    while True:
        # Fetch next batch of pending document IDs
        document_ids = _fetch_pending_document_ids(batch_size)
        if not document_ids:
            break

        LOGGER.info(f"Processing batch of {len(document_ids)} documents...")

        # Process batch in parallel using ThreadPoolExecutor
        # Why ThreadPoolExecutor? Best for I/O-bound tasks (file downloads, DB writes)
        # ProcessPoolExecutor would be overkill (CPU-bound tasks only)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all document processing tasks to thread pool
            # Returns a dict mapping Future → document_id for tracking
            future_to_doc_id = {
                executor.submit(process_document, doc_id, storage): doc_id
                for doc_id in document_ids
            }

            # Wait for tasks to complete and handle results
            # as_completed() yields futures as they finish (not submission order)
            for future in as_completed(future_to_doc_id):
                doc_id = future_to_doc_id[future]
                try:
                    # future.result() will re-raise any exception from worker thread
                    future.result()
                    total_processed += 1
                    LOGGER.info(f"✓ Completed: {doc_id}")
                except Exception as exc:  # noqa: BLE001
                    # Log errors but continue processing other documents
                    total_failed += 1
                    LOGGER.error(f"✗ Failed {doc_id}: {exc}")

        LOGGER.info(
            f"Batch complete. Processed: {len(document_ids)} "
            f"(Total success: {total_processed}, failed: {total_failed})"
        )

    # Final summary
    if total_processed == 0 and total_failed == 0:
        LOGGER.info("No NEW documents to process.")
    else:
        LOGGER.info(
            f"Processing complete. Success: {total_processed}, Failed: {total_failed}"
        )


if __name__ == "__main__":
    # Parse command-line arguments for flexibility
    # Usage: uv run scripts/process_pdfs.py --batch-size 20 --workers 8
    arg_parser = argparse.ArgumentParser(description="Process PDFs in parallel")
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Documents to fetch per batch (default: 10)"
    )
    arg_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker threads (default: 4, recommended: 2-8)"
    )
    args = arg_parser.parse_args()

    main(batch_size=args.batch_size, max_workers=args.workers)
