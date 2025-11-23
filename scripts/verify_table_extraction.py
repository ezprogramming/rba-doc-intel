"""Verify table extraction quality.

This script helps you inspect extracted tables and compare them with the original PDFs.

Usage:
    uv run python scripts/verify_table_extraction.py
    # Or via make:
    make run CMD="uv run python scripts/verify_table_extraction.py"
"""

from __future__ import annotations

import json
import logging
import sys
from typing import List

from app.db.models import Chunk as ChunkModel
from app.db.models import Document
from app.db.models import Table as TableModel
from app.db.session import session_scope
from sqlalchemy import desc, func, select

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def format_table_rows(structured_data: List[dict], max_rows: int = 5) -> str:
    """Format table rows for display."""
    if not structured_data:
        return "  (empty table)"

    lines = []

    # Get headers
    headers = list(structured_data[0].keys())

    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))

    for row in structured_data[:max_rows]:
        for header in headers:
            value_len = len(str(row.get(header, "")))
            col_widths[header] = max(col_widths[header], value_len)

    # Print header row
    header_line = " | ".join(
        str(h).ljust(col_widths[h])[:30]  # Truncate long headers to 30 chars
        for h in headers
    )
    lines.append(f"  {header_line}")
    lines.append(f"  {'-' * min(len(header_line), 120)}")

    # Print data rows
    for i, row in enumerate(structured_data[:max_rows]):
        row_line = " | ".join(
            str(row.get(h, "")).ljust(col_widths[h])[:30]  # Truncate long values
            for h in headers
        )
        lines.append(f"  {row_line}")

    if len(structured_data) > max_rows:
        lines.append(f"  ... ({len(structured_data) - max_rows} more rows)")

    return "\n".join(lines)


def show_document_tables(doc_id: str) -> None:
    """Show all tables extracted from a specific document."""
    with session_scope() as session:
        # Get document info
        doc = session.get(Document, doc_id)
        if not doc:
            LOGGER.error(f"Document {doc_id} not found")
            return

        print_separator()
        print(f"DOCUMENT: {doc.title}")
        print(f"Type: {doc.doc_type}")
        print(f"Date: {doc.publication_date}")
        print(f"S3 Key: {doc.s3_key}")
        print_separator()
        print()

        # Get all tables for this document
        stmt = (
            select(TableModel)
            .where(TableModel.document_id == doc_id)
            .order_by(TableModel.page_number, TableModel.id)
        )
        tables = session.scalars(stmt).all()

        if not tables:
            print("No tables extracted from this document.\n")
            return

        print(f"Found {len(tables)} table(s):\n")

        for i, table in enumerate(tables, 1):
            print(f"TABLE {i} - Page {table.page_number}")
            print(f"  ID: {table.id}")
            print(f"  Caption: {table.caption or '(no caption)'}")
            print(f"  Accuracy: {table.accuracy}%" if table.accuracy else "  Accuracy: N/A")
            print(f"  Rows: {len(table.structured_data)}")
            num_cols = len(table.structured_data[0].keys()) if table.structured_data else 0
            print(f"  Columns: {num_cols}")

            # Check if this table has linked chunks
            chunk_stmt = (
                select(func.count()).select_from(ChunkModel).where(ChunkModel.table_id == table.id)
            )
            chunk_count = session.scalar(chunk_stmt)
            print(f"  Linked chunks: {chunk_count}")

            print("\n  Preview (first 5 rows):")
            print(format_table_rows(table.structured_data, max_rows=5))
            print()

        # Show linked chunk samples
        print_separator("-")
        print("CHUNK SAMPLES (how tables appear in RAG context):\n")

        for i, table in enumerate(tables[:3], 1):  # Show first 3 tables
            chunk_stmt = select(ChunkModel).where(ChunkModel.table_id == table.id).limit(1)
            chunk = session.scalar(chunk_stmt)

            if chunk:
                print(f"TABLE {i} → Chunk {chunk.id}")
                print(f"  Section: {chunk.section_hint or 'N/A'}")
                print("  Chunk text (first 300 chars):")
                print(f"  {chunk.text[:300]}...")
                print()


def show_recent_documents(limit: int = 10) -> None:
    """Show recently processed documents with table counts."""
    with session_scope() as session:
        # Get documents with table counts
        stmt = (
            select(
                Document.id,
                Document.title,
                Document.doc_type,
                Document.publication_date,
                Document.status,
                func.count(TableModel.id).label("table_count"),
            )
            .outerjoin(TableModel, Document.id == TableModel.document_id)
            .group_by(Document.id)
            .order_by(desc(Document.created_at))
            .limit(limit)
        )

        results = session.execute(stmt).all()

        print_separator()
        print(f"RECENT DOCUMENTS (showing {len(results)}):")
        print_separator()
        print()

        for row in results:
            doc_id = str(row.id)
            title = row.title[:70] + "..." if len(row.title) > 70 else row.title

            print(f"• {title}")
            print(f"  ID: {doc_id}")
            print(f"  Type: {row.doc_type}, Date: {row.publication_date}, Status: {row.status}")
            print(f"  Tables: {row.table_count}")
            print()


def show_table_statistics() -> None:
    """Show overall table extraction statistics."""
    with session_scope() as session:
        # Total tables
        total_tables = session.scalar(select(func.count()).select_from(TableModel))

        # Tables by accuracy range
        high_accuracy = session.scalar(
            select(func.count()).select_from(TableModel).where(TableModel.accuracy >= 95)
        )

        medium_accuracy = session.scalar(
            select(func.count())
            .select_from(TableModel)
            .where(TableModel.accuracy >= 80, TableModel.accuracy < 95)
        )

        low_accuracy = session.scalar(
            select(func.count()).select_from(TableModel).where(TableModel.accuracy < 80)
        )

        # Tables with/without captions
        with_caption = session.scalar(
            select(func.count())
            .select_from(TableModel)
            .where(TableModel.caption.isnot(None), TableModel.caption != "")
        )

        # Linked chunks
        linked_chunks = session.scalar(
            select(func.count()).select_from(ChunkModel).where(ChunkModel.table_id.isnot(None))
        )

        print_separator()
        print("TABLE EXTRACTION STATISTICS:")
        print_separator()
        print(f"\nTotal tables extracted: {total_tables}")
        print("\nAccuracy distribution:")
        if total_tables > 0:
            high_pct = high_accuracy / total_tables * 100
            med_pct = medium_accuracy / total_tables * 100
            low_pct = low_accuracy / total_tables * 100
            print(f"  ≥95%: {high_accuracy} ({high_pct:.1f}%)")
            print(f"  80-94%: {medium_accuracy} ({med_pct:.1f}%)")
            print(f"  <80%: {low_accuracy} ({low_pct:.1f}%)")
        else:
            print("  ≥95%: 0")
            print("  80-94%: 0")
            print("  <80%: 0")
        print(f"\nTables with captions: {with_caption}/{total_tables}")
        print(f"Chunks linked to tables: {linked_chunks}")
        print()


def export_table_to_json(table_id: int, output_file: str) -> None:
    """Export a specific table to JSON for detailed inspection."""
    with session_scope() as session:
        table = session.get(TableModel, table_id)
        if not table:
            LOGGER.error(f"Table {table_id} not found")
            return

        data = {
            "table_id": table.id,
            "document_id": str(table.document_id),
            "page_number": table.page_number,
            "caption": table.caption,
            "accuracy": table.accuracy,
            "bbox": table.bbox,
            "rows": table.structured_data,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"✓ Exported table {table_id} to {output_file}")


def main() -> int:
    """Main verification workflow."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "stats":
            show_table_statistics()
            return 0

        elif command == "doc" and len(sys.argv) > 2:
            doc_id = sys.argv[2]
            show_document_tables(doc_id)
            return 0

        elif command == "export" and len(sys.argv) > 3:
            table_id = int(sys.argv[2])
            output_file = sys.argv[3]
            export_table_to_json(table_id, output_file)
            return 0

        else:
            print("Usage:")
            print("  python scripts/verify_table_extraction.py stats")
            print("  python scripts/verify_table_extraction.py doc <document_id>")
            print("  python scripts/verify_table_extraction.py export <table_id> <output.json>")
            return 1

    # Default: show recent documents and stats
    show_table_statistics()
    print()
    show_recent_documents(limit=10)

    print("\nTo inspect a specific document's tables:")
    print("  python scripts/verify_table_extraction.py doc <document_id>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
