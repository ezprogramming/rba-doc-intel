"""Standalone table extraction pipeline for RBA PDFs.

This script scans documents, extracts structured tables via Camelot, and
persists both structured rows and flattened table chunks for RAG retrieval.
Run this after `make process` to keep table data in sync without slowing down
the main text-processing pipeline.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Sequence, Tuple

from app.config import get_settings
from app.db.models import Chunk as ChunkModel
from app.db.models import Document, DocumentStatus, Table
from app.db.session import session_scope
from app.pdf import parser
from app.pdf.chunker import Chunk as ChunkSegment
from app.pdf.table_extractor import TableExtractor
from app.storage import MinioStorage
from sqlalchemy import func, select

LOGGER = logging.getLogger(__name__)


def _download_to_temp(storage: MinioStorage, s3_key: str) -> Path:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        temp_path = Path(tmp_file.name)
    storage.download_file(storage.raw_bucket, s3_key, temp_path)
    return temp_path


def _normalize_header(label: str, index: int) -> str:
    """Normalize table header while preserving important information.

    Improvements:
    - Preserve multi-word headers (don't truncate)
    - Keep special chars like %, $, year indicators
    - Clean only excessive whitespace/punctuation
    - Better fallback naming
    """
    # Remove excessive whitespace and edge punctuation
    cleaned = (label or "").strip(" :\n\t")

    # Preserve internal structure (don't replace multiple spaces)
    cleaned = " ".join(cleaned.split())

    # Better fallback: use original index-based name if empty
    if not cleaned:
        return f"Col_{index}"

    return cleaned


def _format_row_sentence(row: dict, headers: List[str]) -> str:
    parts: List[str] = []
    for header in headers:
        value = str(row.get(header, "")).strip()
        if not value:
            continue
        parts.append(f"{header}: {value}")
    return "; ".join(parts)


def _table_to_text(table: dict, max_rows: int = 50, format_style: str = "hybrid") -> tuple[str, List[str]]:
    """Convert table to precise, structured format with multiple rendering options.

    Format styles:
    - 'semantic': Natural language (GDP — Jun 2024: 1.6%, Dec 2024: 2.3%)
    - 'markdown': Markdown table format (preserves exact structure)
    - 'hybrid': Both markdown + semantic summary (default)

    Improvements:
    - Increased max_rows from 20 to 50 (less truncation)
    - Preserves exact numeric precision
    - Better header detection (multi-word, special chars)
    - Markdown format option for exact structure preservation
    """
    rows = table.get("data") or []
    if not rows:
        return "", []

    raw_headers = list(rows[0].keys())
    headers = [_normalize_header(str(col), idx) for idx, col in enumerate(raw_headers)]

    # Metadata
    caption = (table.get("caption") or "").strip()
    page_num = table.get("page_number", "")
    accuracy = table.get("accuracy", 0)

    line_items: List[str] = []

    # Add context header
    if caption:
        line_items.append(f"{caption}")
    elif headers and len(headers) > 1:
        line_items.append("Table Data")

    line_items.append(f"(Page {page_num}, {accuracy}% accuracy)\n")

    # MARKDOWN FORMAT - preserves exact structure
    if format_style in ("markdown", "hybrid"):
        # Header row
        header_line = "| " + " | ".join(headers) + " |"
        line_items.append(header_line)
        # Separator
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        line_items.append(separator)

        # Data rows
        for row in rows[:max_rows]:
            cells = []
            for h in raw_headers:
                val = str(row.get(h, "")).strip()
                cells.append(val or "—")
            row_line = "| " + " | ".join(cells) + " |"
            line_items.append(row_line)

        if len(rows) > max_rows:
            line_items.append(f"\n... ({len(rows) - max_rows} more rows)")

        # Add blank line between formats
        if format_style == "hybrid":
            line_items.append("\n**Semantic Summary:**")

    # SEMANTIC FORMAT - for LLM reasoning
    if format_style in ("semantic", "hybrid"):
        # Detect metric column (usually first column)
        metric_col = raw_headers[0] if raw_headers else None
        value_cols = raw_headers[1:] if len(raw_headers) > 1 else []

        # Format each row as "Metric — value1, value2, value3..."
        for row in rows[:max_rows]:
            metric = str(row.get(metric_col, "")).strip()

            # Skip empty rows
            if not metric:
                continue

            # Collect values from other columns
            values = []
            for col in value_cols:
                val = str(row.get(col, "")).strip()
                if val:
                    # Preserve numeric precision - don't truncate
                    col_name = str(col)
                    values.append(f"{col_name}: {val}")

            # Create semantic line
            if values:
                line_items.append(f"{metric} — {', '.join(values)}")
            else:
                # Single-column table (rare)
                line_items.append(f"• {metric}")

        if len(rows) > max_rows and format_style != "hybrid":
            line_items.append(f"\n... ({len(rows) - max_rows} more rows)")

    return "\n".join(line_items), headers


def _metrics_from_headers(headers: List[str], max_metrics: int = 4) -> List[str]:
    metrics: List[str] = []
    for header in headers:
        token = header.lower()
        keywords = ("gdp", "cpi", "inflation", "unemployment", "gni", "%", "rate")
        if any(keyword in token for keyword in keywords):
            metrics.append(header)
        elif token.endswith(("%", "growth", "index", "rate")):
            metrics.append(header)
        if len(metrics) >= max_metrics:
            break
    return metrics


def _build_table_chunks(tables: List[dict], start_index: int) -> List[dict]:
    segments: List[dict] = []
    chunk_index = start_index
    for table_idx, table in enumerate(tables):
        text, headers = _table_to_text(table)
        if not text:
            continue
        page_number = table.get("page_number", 1)
        accuracy = table.get("accuracy")
        hint = f"Table (page {page_number})"
        if accuracy is not None:
            hint += f" · acc {accuracy}%"
        metrics = _metrics_from_headers(headers)
        if metrics:
            hint += f" · metrics: {', '.join(metrics)}"
        segments.append(
            {
                "segment": ChunkSegment(
                    text=text,
                    page_start=max(page_number - 1, 0),
                    page_end=max(page_number - 1, 0),
                    chunk_index=chunk_index,
                    section_hint=hint,
                ),
                "table_index": table_idx,
            }
        )
        chunk_index += 1
    return segments


def _extract_tables_from_pdf(pdf_path: Path, page_count: int) -> List[dict]:
    extractor = TableExtractor()
    tables: List[dict] = []
    for page_idx in range(page_count):
        page_num = page_idx + 1
        metadata = extractor.extract_page_metadata(pdf_path, page_num)
        for table in metadata.get("tables", []):
            accuracy_val = table.get("accuracy")
            tables.append(
                {
                    "page_number": page_num,
                    "data": table.get("data", []),
                    "bbox": table.get("bbox"),
                    "accuracy": int(accuracy_val) if accuracy_val is not None else None,
                    "caption": table.get("caption"),  # Use extracted caption from table_extractor
                }
            )
    return tables


def _persist_tables_and_chunks(document_id: str, tables: List[dict]) -> None:
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            LOGGER.warning("Document %s vanished before table persist", document_id)
            return

        # Delete old table chunks first (before deleting tables)
        # This prevents orphaned chunks with table_id = NULL
        session.query(ChunkModel).filter(
            ChunkModel.document_id == doc.id,
            ChunkModel.table_id.isnot(None)
        ).delete(synchronize_session=False)

        # Now delete old tables
        session.query(Table).filter(Table.document_id == doc.id).delete(synchronize_session=False)
        session.flush()

        persisted_tables: List[Table] = []
        for table in tables:
            table_row = Table(
                document_id=doc.id,
                page_number=table["page_number"],
                structured_data=table["data"],
                bbox=table.get("bbox"),
                accuracy=table.get("accuracy"),
                caption=table.get("caption"),
            )
            session.add(table_row)
            session.flush()
            persisted_tables.append(table_row)

        max_index = session.scalar(
            select(func.max(ChunkModel.chunk_index)).where(ChunkModel.document_id == doc.id)
        )
        next_index = (max_index or -1) + 1
        chunk_segments = _build_table_chunks(tables, start_index=next_index)
        for payload in chunk_segments:
            segment = payload["segment"]
            table_idx = payload["table_index"]
            table_row = persisted_tables[table_idx] if table_idx < len(persisted_tables) else None
            session.add(
                ChunkModel(
                    document_id=doc.id,
                    page_start=segment.page_start,
                    page_end=segment.page_end,
                    chunk_index=segment.chunk_index,
                    text=segment.text,
                    section_hint=segment.section_hint,
                    table_id=table_row.id if table_row else None,
                )
            )

        if doc.status == DocumentStatus.EMBEDDED.value:
            doc.status = DocumentStatus.CHUNKS_BUILT.value


def process_document(document_id: str, s3_key: str) -> Tuple[str, List[dict]]:
    storage = MinioStorage()
    temp_path: Path | None = None
    try:
        temp_path = _download_to_temp(storage, s3_key)
        page_count = len(parser.extract_pages(temp_path))
        if page_count == 0:
            LOGGER.info("Document %s has no pages", document_id)
            return document_id, []

        tables = _extract_tables_from_pdf(temp_path, page_count)
        if not tables:
            LOGGER.info("No tables detected for %s", document_id)
            return document_id, []

        LOGGER.info("Extracted %s tables for %s", len(tables), document_id)
        return document_id, tables

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to extract tables for %s: %s", document_id, exc)
        return document_id, []
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def _fetch_target_documents(
    limit: int,
    document_ids: Sequence[str] | None,
    force: bool
) -> List[Tuple[str, str]]:
    with session_scope() as session:
        if document_ids:
            stmt = (
                select(Document.id, Document.s3_key)
                .where(Document.id.in_(document_ids))
                .with_for_update(skip_locked=True)
            )
        else:
            exists_tables = (
                select(Table.id)
                .where(Table.document_id == Document.id)
                .exists()
            )

            if force:
                # Force mode: can process any doc (before or after embeddings)
                stmt = (
                    select(Document.id, Document.s3_key)
                    .where(
                        Document.status.in_(
                            [DocumentStatus.CHUNKS_BUILT.value, DocumentStatus.EMBEDDED.value]
                        )
                    )
                )
            else:
                # Incremental mode: only process docs before embeddings
                # This prevents accidentally invalidating embeddings
                stmt = (
                    select(Document.id, Document.s3_key)
                    .where(Document.status == DocumentStatus.CHUNKS_BUILT.value)
                    .where(~exists_tables)
                )

            stmt = stmt.with_for_update(skip_locked=True)
        stmt = stmt.limit(limit)
        return [(str(row[0]), row[1]) for row in session.execute(stmt)]


def run_pipeline(
    batch_size: int,
    workers: int,
    document_ids: Sequence[str] | None,
    force: bool,
) -> None:
    storage_warning = "Processing tables with %s workers"
    LOGGER.info(storage_warning, workers)

    # When force=True or specific docs requested, fetch ALL targets once to avoid infinite loop
    # Otherwise, fetch in batches and loop until queue is empty
    if force or document_ids:
        all_targets = _fetch_target_documents(limit=10000, document_ids=document_ids, force=force)
        if not all_targets:
            LOGGER.info("No documents found for table extraction.")
            return
        LOGGER.info("Force mode: processing all %s documents in batches of %s", len(all_targets), batch_size)

        # Process in batches
        for batch_start in range(0, len(all_targets), batch_size):
            targets = all_targets[batch_start:batch_start + batch_size]
            _process_batch(targets, workers)
    else:
        # Incremental mode: fetch and process new documents until queue is empty
        while True:
            targets = _fetch_target_documents(batch_size, document_ids=None, force=False)
            if not targets:
                LOGGER.info("No documents pending table extraction.")
                break
            _process_batch(targets, workers)


def _process_batch(targets: List[Tuple[str, str]], workers: int) -> None:
    """Process a batch of documents for table extraction."""
    LOGGER.info("Processing table batch of %s documents", len(targets))
    completed = 0

    if workers <= 1:
        for doc_id, s3_key in targets:
            _, tables = process_document(doc_id, s3_key)
            # Always persist, even with 0 tables (prevents infinite loop)
            _persist_tables_and_chunks(doc_id, tables)
            if tables:
                completed += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_doc = {
                executor.submit(process_document, doc_id, s3_key): doc_id
                for doc_id, s3_key in targets
            }
            for future in as_completed(future_to_doc):
                doc_id = future_to_doc[future]
                try:
                    _, tables = future.result()
                    # Always persist, even with 0 tables (prevents infinite loop)
                    _persist_tables_and_chunks(doc_id, tables)
                    if tables:
                        completed += 1
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Table extraction failed for %s: %s", doc_id, exc)

    LOGGER.info("Batch complete. Tables extracted for %s/%s documents", completed, len(targets))


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Extract structured tables for processed PDFs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.table_batch_size,
        help=f"Documents to fetch per iteration (default from .env: {settings.table_batch_size})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.table_max_workers,
        help=(
            f"Parallel processes for table extraction "
            f"(default from .env: {settings.table_max_workers})"
        ),
    )
    parser.add_argument(
        "--document-id",
        action="append",
        help="Process specific document ID(s) (can be repeated)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract tables even if existing rows are present",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(
        batch_size=args.batch_size,
        workers=args.workers,
        document_ids=args.document_id,
        force=args.force,
    )


if __name__ == "__main__":
    main()
