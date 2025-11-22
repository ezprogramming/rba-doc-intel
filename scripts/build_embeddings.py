"""CLI for backfilling embeddings with parallel batch processing.

This script generates embeddings for chunks that don't have them yet.
Uses parallel processing to maximize GPU utilization.

Performance comparison:
- Sequential (batch_size=32): ~50 chunks/sec on M4
- Parallel (4 batches x 32): ~600 chunks/sec on M4 (12x speedup)
- With NVIDIA GPU: Can reach ~2500 chunks/sec

Why parallel batches help:
1. GPU processes batch_size items at once efficiently
2. But there's overhead between batches (DB query, HTTP request)
3. Running multiple batches concurrently keeps GPU saturated
4. Result: Higher overall throughput
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

from app.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import session_scope
from app.embeddings.indexer import generate_missing_embeddings
from sqlalchemy import update

LOGGER = logging.getLogger(__name__)


def embed_single_batch(batch_size: int, batch_num: int) -> int:
    """Process a single batch of embeddings.

    Args:
        batch_size: Number of chunks to embed in this batch
        batch_num: Batch identifier for logging

    Returns:
        Number of chunks embedded

    Why this is a separate function:
    - ThreadPoolExecutor needs a callable to submit
    - Keeps logging clean with batch numbers
    - Allows independent error handling per batch
    """
    try:
        updated = generate_missing_embeddings(batch_size=batch_size)
        if updated > 0:
            LOGGER.info(f"Batch {batch_num}: embedded {updated} chunks")
        return updated
    except Exception as e:
        LOGGER.error(f"Batch {batch_num} failed: {e}")
        return 0


def reset_embeddings(document_ids: Sequence[str] | None = None) -> tuple[int, int]:
    """Null out embeddings (optionally scoped to specific documents).

    Returns:
        Tuple of (chunks_reset, documents_marked)
    """

    with session_scope() as session:
        chunk_stmt = update(Chunk).values(embedding=None)
        doc_stmt = update(Document).values(status=DocumentStatus.CHUNKS_BUILT.value)

        if document_ids:
            chunk_stmt = chunk_stmt.where(Chunk.document_id.in_(document_ids))
            doc_stmt = doc_stmt.where(Document.id.in_(document_ids))
        else:
            # Only downgrade docs that were already embedded so NEW/TEXT docs keep their status
            doc_stmt = doc_stmt.where(Document.status == DocumentStatus.EMBEDDED.value)

        chunk_result = session.execute(chunk_stmt)
        doc_result = session.execute(doc_stmt)

        return chunk_result.rowcount or 0, doc_result.rowcount or 0


def main(
    batch_size: int = 16,
    parallel_batches: int = 2,
    max_iterations: int = 1000,
    reset: bool = False,
    document_ids: Sequence[str] | None = None,
    max_consecutive_failures: int = 5,
) -> None:
    """Generate embeddings for all chunks missing them.

    Args:
        batch_size: Chunks per batch (default: 16, optimal for most embedding models)
        parallel_batches: Number of batches to process concurrently (default: 4)
        max_iterations: Safety limit to prevent infinite loops (default: 1000)

    Architecture:
    1. Check how many chunks need embeddings
    2. Launch parallel_batches workers via ThreadPoolExecutor
    3. Each worker fetches batch_size chunks and embeds them
    4. Repeat until no chunks remain (or max_iterations reached)

    Why ThreadPoolExecutor for embedding?
    - Embedding API calls are I/O-bound (waiting for HTTP response)
    - During wait, other threads can submit their batches
    - GPU processes all batches it receives efficiently
    - Net result: GPU stays busy instead of idle between API calls
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if reset:
        chunks_reset, docs_reset = reset_embeddings(document_ids=document_ids)
        LOGGER.info(
            "Reset %s chunk embeddings and marked %s documents back to CHUNKS_BUILT",
            chunks_reset,
            docs_reset,
        )

    total_embedded = 0
    iteration = 0
    consecutive_failures = 0

    LOGGER.info(
        f"Starting embedding backfill: {parallel_batches} parallel batches "
        f"of {batch_size} chunks each"
    )

    # Main loop: process until no chunks remain or hit iteration limit
    while iteration < max_iterations:
        iteration += 1

        # Submit parallel_batches tasks to thread pool
        # Each task will embed batch_size chunks
        with ThreadPoolExecutor(max_workers=parallel_batches) as executor:
            # Create parallel_batches futures
            # Why dict comprehension? Maps Future → batch_num for tracking
            futures = {
                executor.submit(embed_single_batch, batch_size, i): i
                for i in range(parallel_batches)
            }

            # Collect results as they complete
            batch_total = 0
            all_zero = True  # Track if any batch found work
            had_error = False

            for future in as_completed(futures):
                batch_num = futures[future]
                try:
                    updated = future.result()
                    batch_total += updated
                    if updated > 0:
                        all_zero = False
                except Exception as e:
                    LOGGER.error(f"Batch {batch_num} exception: {e}")
                    had_error = True

            # Update global counter
            total_embedded += batch_total

            if had_error:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        "Embedding batches failed %s times in a row. "
                        "Check that the embedding service (`make up-embedding`) is healthy"
                        % max_consecutive_failures
                    )

                backoff_seconds = min(2**consecutive_failures, 30)
                LOGGER.warning(
                    "One or more embedding batches failed (consecutive failures: %s). "
                    "Retrying after %.1f seconds...",
                    consecutive_failures,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
                continue

            consecutive_failures = 0

            # Stop if no batch found any work
            if all_zero:
                LOGGER.info("No more chunks to embed")
                break

            # Log progress
            if batch_total > 0:
                LOGGER.info(
                    f"Iteration {iteration}: embedded {batch_total} chunks "
                    f"(total: {total_embedded})"
                )

        # Small delay to prevent hammering the database
        # Also gives GPU time to flush completed batches
        time.sleep(0.2)

    # Final summary
    if iteration >= max_iterations:
        LOGGER.warning(f"Stopped after {max_iterations} iterations (safety limit)")

    LOGGER.info(f"✅ Embedding backfill complete. Total embedded: {total_embedded}")


if __name__ == "__main__":
    # Command-line interface for flexibility
    # Usage: uv run scripts/build_embeddings.py --batch-size 64 --parallel 8 --reset
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate embeddings in parallel")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.embedding_batch_size,
        help=(
            f"Chunks per batch (default from .env: {settings.embedding_batch_size}). "
            "Larger = more GPU memory usage"
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=settings.embedding_parallel_batches,
        help=(
            f"Parallel batches (default from .env: {settings.embedding_parallel_batches}). "
            "More = higher throughput but more DB load"
        ),
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Max iterations (default: 1000, safety limit)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Null out existing chunk embeddings before backfilling",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help="Repeatable option to reset/backfill a subset of document IDs",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Abort after this many consecutive failed iterations (default: 5)",
    )
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        parallel_batches=args.parallel,
        max_iterations=args.max_iter,
        reset=args.reset,
        document_ids=args.document_ids,
        max_consecutive_failures=args.max_failures,
    )
