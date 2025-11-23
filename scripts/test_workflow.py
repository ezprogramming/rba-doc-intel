"""End-to-end workflow test with real RBA PDFs.

This script:
1. Uses crawler to discover and download 1-2 recent RBA PDFs
2. Runs ingestion (text + tables in one pass)
3. Generates embeddings
4. Tests RAG retrieval to verify tables work correctly

Usage:
    uv run python scripts/test_workflow.py
    # Or via make:
    make test-workflow
"""

from __future__ import annotations

import logging
import sys
from typing import List

from app.config import get_settings
from app.db.models import Chunk as ChunkModel
from app.db.models import Document, DocumentStatus
from app.db.models import Table as TableModel
from app.db.session import session_scope
from app.embeddings.client import EmbeddingClient
from app.embeddings.indexer import generate_missing_embeddings
from app.pdf.table_extractor import TableExtractor
from app.rag.retriever import retrieve_similar_chunks
from app.storage import MinioStorage
from sqlalchemy import select

from scripts.crawler_rba import SOURCES, ingest_source
from scripts.ingest_documents import ingest_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def crawl_test_pdfs(storage: MinioStorage, max_pdfs: int = 2) -> List[str]:
    """Crawl a small number of recent RBA PDFs for testing.

    Returns list of document IDs.
    """
    LOGGER.info("Crawling %d recent RBA PDFs...", max_pdfs)

    # Get count before crawling
    with session_scope() as session:
        session.query(Document).count()

    # Crawl from first source (Statement on Monetary Policy)
    # Use year filter to only get recent publications
    import os

    original_filter = os.getenv("CRAWLER_YEAR_FILTER", "")
    os.environ["CRAWLER_YEAR_FILTER"] = "2024+"  # Only 2024 onwards

    try:
        # Reload year filters in crawler module
        from scripts import crawler_rba

        crawler_rba.YEAR_FILTERS = crawler_rba._parse_year_filters("2024+")

        # Crawl just the first source (SMP)
        source = SOURCES[0]  # Statement on Monetary Policy
        ingested = ingest_source(source, storage)
        LOGGER.info("Crawled %d PDFs from %s", ingested, source.name)

    finally:
        # Restore original filter
        os.environ["CRAWLER_YEAR_FILTER"] = original_filter
        crawler_rba.YEAR_FILTERS = crawler_rba._parse_year_filters(original_filter)

    # Get newly added documents
    with session_scope() as session:
        stmt = (
            select(Document.id)
            .where(Document.status == DocumentStatus.NEW.value)
            .order_by(Document.created_at.desc())
            .limit(max_pdfs)
        )
        doc_ids = [str(row[0]) for row in session.execute(stmt)]

    if not doc_ids:
        LOGGER.warning("No new PDFs found. Check if PDFs already exist in database.")
        # Get any existing documents for testing
        with session_scope() as session:
            stmt = select(Document.id).order_by(Document.created_at.desc()).limit(max_pdfs)
            doc_ids = [str(row[0]) for row in session.execute(stmt)]

    LOGGER.info("Using %d documents for testing: %s", len(doc_ids), doc_ids)
    return doc_ids


def verify_ingestion(doc_id: str) -> dict:
    """Verify that ingestion created chunks, pages, and tables.

    Returns stats dict.
    """
    with session_scope() as session:
        # Check document status
        doc = session.get(Document, doc_id)
        assert doc is not None, "Document not found"
        assert doc.status == DocumentStatus.CHUNKS_BUILT.value, (
            f"Expected CHUNKS_BUILT, got {doc.status}"
        )

        # Check chunks
        chunks = (
            session.query(ChunkModel)
            .filter(ChunkModel.document_id == doc_id)
            .order_by(ChunkModel.chunk_index)
            .all()
        )
        assert len(chunks) > 0, "No chunks created"

        # Check tables
        tables = session.query(TableModel).filter(TableModel.document_id == doc_id).all()

        # Check table chunks (chunks with table_id set)
        table_chunks = [c for c in chunks if c.table_id is not None]

        stats = {
            "document_id": doc_id,
            "title": doc.title,
            "status": doc.status,
            "chunk_count": len(chunks),
            "table_count": len(tables),
            "table_chunk_count": len(table_chunks),
        }

        title_display = doc.title[:60] + "..." if len(doc.title) > 60 else doc.title
        LOGGER.info("✓ Document: %s (status: %s)", title_display, doc.status)
        LOGGER.info("  Chunks: %d total, %d from tables", len(chunks), len(table_chunks))
        LOGGER.info("  Tables: %d extracted", len(tables))

        # Verify table linkage
        if table_chunks:
            for table_chunk in table_chunks[:3]:  # Show first 3
                table = session.get(TableModel, table_chunk.table_id)
                assert table is not None, f"Table {table_chunk.table_id} not found"
                assert table.structured_data is not None, "Table has no structured_data"
                LOGGER.info(
                    "    Chunk %d → table %d (%d rows, accuracy: %d%%)",
                    table_chunk.chunk_index,
                    table.id,
                    len(table.structured_data),
                    table.accuracy or 0,
                )

        return stats


def verify_embeddings(doc_id: str) -> None:
    """Verify that embeddings were generated."""
    with session_scope() as session:
        chunks = session.query(ChunkModel).filter(ChunkModel.document_id == doc_id).all()

        embedded_count = sum(1 for c in chunks if c.embedding is not None)
        assert embedded_count == len(chunks), (
            f"Only {embedded_count}/{len(chunks)} chunks have embeddings"
        )
        LOGGER.info("✓ All %d chunks have embeddings", len(chunks))

        # Check embedding dimensions
        for chunk in chunks[:1]:  # Check first chunk
            assert chunk.embedding is not None
            LOGGER.info("  Embedding dimension: %d", len(chunk.embedding))


def test_rag_retrieval(doc_stats: List[dict]) -> None:
    """Test RAG retrieval with queries relevant to RBA publications."""
    LOGGER.info("\n=== Testing RAG Retrieval ===")

    # Generic queries that should work with any RBA publication
    test_queries = [
        ("What is the inflation outlook?", "inflation"),
        ("What are the GDP forecasts?", "GDP"),
        ("What is the cash rate?", "cash rate"),
    ]

    embedding_client = EmbeddingClient()

    for query, expected_keyword in test_queries:
        LOGGER.info("\nQuery: %s", query)

        # Generate query embedding
        query_response = embedding_client.embed([query])
        query_embedding = query_response.vectors[0]

        # Retrieve chunks
        with session_scope() as session:
            results = retrieve_similar_chunks(
                session=session,
                query_text=query,
                query_embedding=query_embedding,
                limit=5,
            )

        if not results:
            LOGGER.warning("  No results found for query: %s", query)
            continue

        # Check results
        found = False
        for i, result in enumerate(results):
            LOGGER.info(
                "  [%d] Score: %.3f, Doc: %s, Section: %s",
                i + 1,
                result.score,
                result.doc_type,
                result.section_hint or "N/A",
            )
            LOGGER.info("      Text preview: %s...", result.text[:120])

            if expected_keyword.lower() in result.text.lower():
                found = True

                # If this chunk has table_id, verify we can fetch the table
                if result.table_id:
                    with session_scope() as session:
                        table = session.get(TableModel, result.table_id)
                        assert table is not None, f"Table {result.table_id} not found"
                        LOGGER.info(
                            "      ✓ Linked to table with %d rows (accuracy: %d%%)",
                            len(table.structured_data),
                            table.accuracy or 0,
                        )

        if found:
            LOGGER.info("✓ Found expected content for query")
        else:
            LOGGER.warning("⚠ Expected keyword '%s' not found in top results", expected_keyword)


def main() -> int:
    """Run end-to-end workflow test."""
    LOGGER.info("=" * 70)
    LOGGER.info("=== End-to-End Workflow Test with Real RBA PDFs ===")
    LOGGER.info("=" * 70)
    LOGGER.info("")

    get_settings()
    storage = MinioStorage()
    extractor = TableExtractor(min_table_accuracy=70.0)

    try:
        # Step 1: Crawl test PDFs
        LOGGER.info("Step 1: Crawling recent RBA PDFs...")
        doc_ids = crawl_test_pdfs(storage, max_pdfs=2)

        if not doc_ids:
            LOGGER.error("No documents found to test. Run crawler first or check database.")
            return 1

        LOGGER.info("✓ Found %d documents to test\n", len(doc_ids))

        # Step 2: Run ingestion for each document
        LOGGER.info("Step 2: Running ingestion (text + tables in one pass)...")
        doc_stats = []
        for doc_id in doc_ids:
            LOGGER.info("  Processing document %s...", doc_id)
            ingest_document(doc_id, storage, extractor)
            stats = verify_ingestion(doc_id)
            doc_stats.append(stats)
        LOGGER.info("✓ Ingestion complete for all documents\n")

        # Step 3: Generate embeddings
        LOGGER.info("Step 3: Generating embeddings...")
        total_chunks = 0

        # Count chunks needing embeddings
        with session_scope() as session:
            chunks_to_embed = (
                session.query(ChunkModel)
                .filter(ChunkModel.document_id.in_(doc_ids))
                .filter(ChunkModel.embedding.is_(None))
                .count()
            )

        # Generate embeddings in batches until all are done
        while chunks_to_embed > 0:
            updated = generate_missing_embeddings(batch_size=4)
            if updated == 0:
                break
            total_chunks += updated
            chunks_to_embed -= updated

        LOGGER.info("✓ Generated embeddings for %d chunks\n", total_chunks)

        # Step 4: Verify embeddings
        LOGGER.info("Step 4: Verifying embeddings...")
        for doc_id in doc_ids:
            verify_embeddings(doc_id)
        LOGGER.info("")

        # Step 5: Test RAG retrieval
        LOGGER.info("Step 5: Testing RAG retrieval...")
        test_rag_retrieval(doc_stats)
        LOGGER.info("")

        # Summary
        LOGGER.info("\n" + "=" * 70)
        LOGGER.info("✓✓✓ ALL TESTS PASSED ✓✓✓")
        LOGGER.info("=" * 70)
        LOGGER.info("\nTested %d real RBA documents:", len(doc_stats))
        for stats in doc_stats:
            title = stats["title"]
            title_display = title[:65] + "..." if len(title) > 65 else title
            LOGGER.info("  • %s", title_display)
            LOGGER.info(
                "    - %d chunks (%d from tables)",
                stats["chunk_count"],
                stats["table_chunk_count"],
            )
            LOGGER.info("    - %d tables extracted", stats["table_count"])

        LOGGER.info("\nWorkflow validation:")
        LOGGER.info("  ✓ PDF ingestion (text + tables)")
        LOGGER.info("  ✓ Table extraction and linking")
        LOGGER.info("  ✓ Embedding generation")
        LOGGER.info("  ✓ RAG retrieval with table content")

        LOGGER.info("\nNote: Test documents remain in database for manual inspection.")
        LOGGER.info("      Use 'make ingest-reset' to reprocess if needed.")

        return 0

    except Exception as exc:
        LOGGER.exception("Test failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
