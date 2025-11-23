"""Robust document ingestion pipeline - PDF to chunks in one pass.

This script handles the complete ingestion workflow:
1. Download PDF from MinIO
2. Extract text + tables simultaneously (single PDF pass)
3. Clean text and create chunks
4. Persist everything to PostgreSQL
5. Handle errors gracefully with proper status tracking

Design principles:
- Single pass: Process PDF once, extract everything
- Idempotent: Safe to re-run (won't duplicate data)
- Robust: Continues processing other docs if one fails
- Fast: Parallel workers for I/O-bound tasks
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, cast

from app.config import get_settings
from app.db.models import Chunk as ChunkModel
from app.db.models import Document, DocumentStatus, Page, Table
from app.db.session import session_scope
from app.pdf import cleaner, parser
from app.pdf.chunker import chunk_pages
from app.pdf.table_extractor import TableExtractor
from app.storage import MinioStorage
from sqlalchemy import select, update

LOGGER = logging.getLogger(__name__)


def _reset_orphaned_processing() -> int:
    """Reset documents stuck in PROCESSING from crashed runs."""
    with session_scope() as session:
        stmt = (
            update(Document)
            .where(Document.status == DocumentStatus.PROCESSING.value)
            .values(status=DocumentStatus.NEW.value)
        )
        result = session.execute(stmt)
        count = result.rowcount
        if count > 0:
            LOGGER.warning("Reset %d orphaned PROCESSING documents", count)
        return count


def reset_all_to_new() -> int:
    """Reset all CHUNKS_BUILT documents back to NEW to force reprocessing.

    Also clears associated chunks, pages, and tables to ensure clean reprocessing.
    """
    with session_scope() as session:
        # Get all CHUNKS_BUILT document IDs
        stmt = select(Document.id).where(Document.status == DocumentStatus.CHUNKS_BUILT.value)
        doc_ids = [str(row[0]) for row in session.execute(stmt)]

        if not doc_ids:
            LOGGER.info("No CHUNKS_BUILT documents to reset")
            return 0

        # Clear associated data (idempotent)
        for doc_id in doc_ids:
            session.query(Page).filter(Page.document_id == doc_id).delete()
            session.query(ChunkModel).filter(ChunkModel.document_id == doc_id).delete()
            session.query(Table).filter(Table.document_id == doc_id).delete()

        # Reset status to NEW
        stmt = (
            update(Document)
            .where(Document.status == DocumentStatus.CHUNKS_BUILT.value)
            .values(status=DocumentStatus.NEW.value)
        )
        result = session.execute(stmt)
        count = result.rowcount

        LOGGER.info("Reset %d documents to NEW status (cleared chunks/pages/tables)", count)
        return count


def _fetch_pending_documents(limit: int) -> List[str]:
    """Fetch document IDs that need processing."""
    with session_scope() as session:
        stmt = (
            select(Document.id)
            .where(Document.status == DocumentStatus.NEW.value)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return [str(row[0]) for row in session.execute(stmt)]


def _download_pdf(storage: MinioStorage, s3_key: str) -> Path:
    """Download PDF to temp file."""
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        temp_path = Path(tmp_file.name)
    storage.download_file(storage.raw_bucket, s3_key, temp_path)
    return temp_path


def _table_to_text(table_data: dict) -> str:
    """Convert table to semantic format for RAG (simplified).

    Example output:
        Economic Forecasts (Table 3.1)
        GDP — 2024: 2.1%, 2025: 2.5%
        Inflation — 2024: 3.5%, 2025: 2.8%
    """
    rows = table_data.get("data") or []
    if not rows:
        return ""

    lines = []

    # Add caption if available (handle None values)
    caption = (table_data.get("caption") or "").strip()
    page_num = table_data.get("page_number", "")
    if caption:
        lines.append(f"{caption} (Page {page_num})")
    elif page_num:
        lines.append(f"Table Data (Page {page_num})")

    # Format rows: Metric — col1: val1, col2: val2, ...
    headers = list(rows[0].keys())
    metric_col = headers[0] if headers else None
    value_cols = headers[1:] if len(headers) > 1 else []

    for row in rows[:50]:  # Limit to 50 rows
        metric = str(row.get(metric_col, "")).strip()
        if not metric:
            continue

        values = []
        for col in value_cols:
            val = str(row.get(col, "")).strip()
            if val:
                values.append(f"{col}: {val}")

        if values:
            lines.append(f"{metric} — {', '.join(values)}")
        else:
            lines.append(f"• {metric}")

    if len(rows) > 50:
        lines.append(f"... ({len(rows) - 50} more rows)")

    return "\n".join(lines)


def ingest_document(document_id: str, storage: MinioStorage, extractor: TableExtractor) -> None:
    """Ingest a single PDF document - text + tables in one pass.

    Three-phase pattern for production robustness:
    1. Fast transaction: Lock document, mark PROCESSING
    2. Heavy work: Download, extract, process (no DB lock)
    3. Fast transaction: Persist results, mark CHUNKS_BUILT
    """
    temp_path: Path | None = None
    s3_key: str | None = None
    title: str = ""

    try:
        # === PHASE 1: Claim document ===
        with session_scope() as session:
            document = session.get(Document, document_id)
            if document is None:
                LOGGER.warning("Document %s not found", document_id)
                return
            document = cast(Document, document)

            document.status = DocumentStatus.PROCESSING.value
            s3_key = document.s3_key
            title = document.title

        LOGGER.info("Processing: %s", title)

        # === PHASE 2: Extract everything (no DB transaction) ===
        temp_path = _download_pdf(storage, s3_key)

        # Extract text pages
        raw_pages = parser.extract_pages(temp_path)
        repeating_headers, repeating_footers = cleaner.detect_repeating_headers_footers(raw_pages)
        clean_pages = [
            cleaner.clean_text(page, repeating_headers, repeating_footers) for page in raw_pages
        ]

        # Create text chunks with quality threshold from config (Phase 6)
        text_segments = chunk_pages(clean_pages, quality_threshold=settings.chunk_quality_threshold)

        # Extract tables from PDF (inline, same file handle)
        tables_data = []
        for page_num in range(1, len(raw_pages) + 1):
            page_tables = extractor.extract_tables(temp_path, page_num)
            if page_tables:
                LOGGER.info("  Found %d table(s) on page %d", len(page_tables), page_num)
                for table in page_tables:
                    table["page_number"] = page_num
                    tables_data.append(table)

        # === PHASE 3: Persist everything ===
        with session_scope() as session:
            document = session.get(Document, document_id)
            if document is None:
                LOGGER.warning("Document %s vanished", document_id)
                return
            document = cast(Document, document)

            # Clear existing data (idempotent - safe to re-run)
            session.query(Page).filter(Page.document_id == document.id).delete()
            session.query(ChunkModel).filter(ChunkModel.document_id == document.id).delete()
            session.query(Table).filter(Table.document_id == document.id).delete()

            # Persist pages
            for index, (raw_text, clean_text) in enumerate(
                zip(raw_pages, clean_pages, strict=True), start=1
            ):
                session.add(
                    Page(
                        document_id=document.id,
                        page_number=index,
                        raw_text=raw_text,
                        clean_text=clean_text,
                    )
                )

            # Persist text chunks
            for segment in text_segments:
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

            # Persist tables and table chunks
            chunk_index_offset = len(text_segments)
            for idx, table_data in enumerate(tables_data):
                # Store structured table
                table_row = Table(
                    document_id=document.id,
                    page_number=table_data["page_number"],
                    structured_data=table_data["data"],
                    caption=table_data.get("caption"),
                    accuracy=int(table_data.get("accuracy", 0)),
                )
                session.add(table_row)
                session.flush()  # Get table_row.id

                # Create chunk from table
                table_text = _table_to_text(table_data)
                if table_text:
                    session.add(
                        ChunkModel(
                            document_id=document.id,
                            table_id=table_row.id,
                            page_start=table_data["page_number"],
                            page_end=table_data["page_number"],
                            chunk_index=chunk_index_offset + idx,
                            text=table_text,
                            section_hint=f"Table (page {table_data['page_number']})",
                        )
                    )

            # Mark complete
            document.status = DocumentStatus.CHUNKS_BUILT.value

        LOGGER.info(
            "✓ Completed: %s (%d chunks, %d tables)",
            title,
            len(text_segments) + len(tables_data),
            len(tables_data),
        )

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed: %s - %s", title, exc)
        # Mark as failed so it doesn't get stuck
        with session_scope() as session:
            document = session.get(Document, document_id)
            if document:
                document.status = DocumentStatus.FAILED.value

    finally:
        # Cleanup temp file
        if temp_path:
            temp_path.unlink(missing_ok=True)


def main(batch_size: int = 16, max_workers: int = 2) -> None:
    """Process pending documents in parallel batches.

    Args:
        batch_size: Documents to fetch per batch
        max_workers: Parallel worker threads (2-4 recommended)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Reset orphaned documents from previous crashes
    _reset_orphaned_processing()

    # Create shared resources (thread-safe)
    storage = MinioStorage()
    extractor = TableExtractor(min_table_accuracy=70.0)

    total_processed = 0
    total_failed = 0

    LOGGER.info("Starting ingestion with %d parallel workers", max_workers)

    while True:
        # Fetch batch
        document_ids = _fetch_pending_documents(batch_size)
        if not document_ids:
            break

        LOGGER.info("Processing batch of %d documents...", len(document_ids))

        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(ingest_document, doc_id, storage, extractor): doc_id
                for doc_id in document_ids
            }

            for future in as_completed(future_to_doc):
                doc_id = future_to_doc[future]
                try:
                    future.result()
                    total_processed += 1
                except Exception as exc:  # noqa: BLE001
                    total_failed += 1
                    LOGGER.error("✗ Failed %s: %s", doc_id, exc)

        LOGGER.info("Batch complete. Total: %d success, %d failed", total_processed, total_failed)

    if total_processed == 0 and total_failed == 0:
        LOGGER.info("No NEW documents to process.")
    else:
        LOGGER.info("Ingestion complete. Success: %d, Failed: %d", total_processed, total_failed)


if __name__ == "__main__":
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest documents (text + tables)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.pdf_batch_size,
        help=f"Documents per batch (default: {settings.pdf_batch_size})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.pdf_max_workers,
        help=f"Parallel workers (default: {settings.pdf_max_workers})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all CHUNKS_BUILT documents to NEW status (forces reprocessing)",
    )
    args = parser.parse_args()

    # Handle reset mode
    if args.reset:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        count = reset_all_to_new()
        LOGGER.info("Reset complete. Run 'make ingest' to reprocess %d documents.", count)
    else:
        # Normal ingestion
        main(batch_size=args.batch_size, max_workers=args.workers)
