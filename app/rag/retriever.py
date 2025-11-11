"""Chunk retrieval logic with optional cross-encoder reranking.

Retrieval Pipeline:
===================
1. Hybrid retrieval (bi-encoder + full-text search)
   → Fast, casts wide net (top-100 candidates)

2. Optional reranking (cross-encoder)
   → Slower, more accurate (top-100 → top-10)

When to use reranking?
- Complex queries requiring deep semantic understanding
- Production RAG systems where answer quality matters
- Trade-off: +200-500ms latency for +25-40% accuracy

When to skip reranking?
- Simple keyword queries (full-text search already works well)
- Latency-critical applications
- Development/debugging (faster iteration)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: str
    text: str
    doc_type: str
    title: str
    publication_date: str | None
    page_start: int | None
    page_end: int | None
    section_hint: str | None
    score: float


# Default weighting aligns with Pinecone/Anthropic guidance:
# semantic vectors drive most of the signal, but we leave room for
# lexical boosts so identifiers/dates aren't lost.
SEMANTIC_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3


def retrieve_similar_chunks(
    session: Session,
    query_text: str,
    query_embedding: Sequence[float],
    limit: int = 5,
    semantic_weight: float = SEMANTIC_WEIGHT,
    lexical_weight: float = LEXICAL_WEIGHT,
    rerank: bool = False,
    rerank_multiplier: int = 10,
) -> List[RetrievedChunk]:
    """Return the top-k chunks using hybrid semantic + lexical retrieval.

    Args:
        session: Database session
        query_text: User query text (used for lexical search and reranking)
        query_embedding: Query embedding vector (used for semantic search)
        limit: Number of final results to return (default: 5)
        semantic_weight: Weight for semantic similarity (default: 0.7)
        lexical_weight: Weight for full-text search (default: 0.3)
        rerank: Whether to use cross-encoder reranking (default: False)
        rerank_multiplier: How many candidates to retrieve before reranking (default: 10)
                          → Retrieve limit * rerank_multiplier, rerank to limit
                          → Example: limit=5, multiplier=10 → retrieve 50, rerank to 5

    Returns:
        List of RetrievedChunk objects, sorted by score (descending)

    How reranking works:
    ====================
    Without reranking (rerank=False):
        1. Hybrid retrieval (bi-encoder + full-text)
        2. Return top-5 by weighted score

    With reranking (rerank=True):
        1. Hybrid retrieval (bi-encoder + full-text) → top-50 candidates
        2. Cross-encoder reranks top-50 → top-5 most relevant
        3. Return reranked top-5

    Why retrieve more candidates before reranking?
    - Bi-encoder retrieval is fast but imprecise (recall@50 > recall@5)
    - Cross-encoder is slow but precise (rescoring 50 is manageable, 10K is not)
    - Two-stage pipeline: fast recall, slow precision

    Performance impact:
    - Without reranking: ~10-50ms
    - With reranking (limit=5, multiplier=10): ~200-500ms
    - Accuracy gain: +25-40% on complex queries

    When to enable reranking?
    - Production RAG systems (quality > latency)
    - Complex analytical queries
    - When bi-encoder retrieval quality is insufficient
    """

    # Determine how many candidates to retrieve
    # If reranking: retrieve limit * rerank_multiplier (e.g., 5 * 10 = 50)
    # If not reranking: retrieve limit * 2 (legacy behavior for hybrid search)
    # Why * 2 without reranking? Hybrid search combines vector + lexical,
    # so we over-fetch to ensure we have enough candidates after deduplication
    retrieval_limit = limit * rerank_multiplier if rerank else limit * 2

    logger.debug(
        f"Retrieving {retrieval_limit} candidates "
        f"(rerank={'enabled' if rerank else 'disabled'}, target={limit})"
    )

    combined: Dict[int, dict] = {}

    # Step 1: Semantic search (bi-encoder vector similarity)
    # Why cosine distance? Embeddings are L2-normalized, so cosine = dot product
    # Fast: pgvector uses HNSW index for approximate nearest neighbor
    distance = Chunk.embedding.cosine_distance(query_embedding)
    vector_stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.text,
            Chunk.page_start,
            Chunk.page_end,
            Document.doc_type,
            Document.title,
            Document.publication_date,
            Chunk.section_hint,
            distance.label("distance"),
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(retrieval_limit)  # Use dynamic limit based on reranking
    )

    vector_rows = session.execute(vector_stmt).all()

    for row in vector_rows:
        entry = combined.setdefault(
            row.id,
            {
                "chunk_id": row.id,
                "document_id": str(row.document_id),
                "text": row.text,
                "doc_type": row.doc_type,
                "title": row.title,
                "publication_date": row.publication_date,
                "page_start": row.page_start,
                "page_end": row.page_end,
                "section_hint": row.section_hint,
                "semantic": 0.0,
                "lexical": 0.0,
            },
        )
        entry["semantic"] = max(entry["semantic"], 1 - float(row.distance))

    # Step 2: Lexical search (full-text search with PostgreSQL tsvector)
    # Why full-text search? Captures keyword matches that embeddings might miss
    # Example: "RBA meeting on 2024-05-07" → date and entity better matched by keywords
    query_text = (query_text or "").strip()
    if lexical_weight > 0 and query_text:
        ts_document = Chunk.text_tsv
        ts_query = func.websearch_to_tsquery("english", query_text)
        lexical_score = func.ts_rank_cd(ts_document, ts_query).label("lexical_rank")
        lexical_stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.text,
                Chunk.page_start,
                Chunk.page_end,
            Document.doc_type,
            Document.title,
            Document.publication_date,
            Chunk.section_hint,
            lexical_score,
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.text_tsv.is_not(None),
                lexical_score > 0,
                Chunk.text_tsv.op('@@')(ts_query),
            )
            .order_by(desc(lexical_score))
            .limit(retrieval_limit)  # Use dynamic limit based on reranking
        )

        lexical_rows = session.execute(lexical_stmt).all()
        for row in lexical_rows:
            entry = combined.setdefault(
                row.id,
                {
                    "chunk_id": row.id,
                    "document_id": str(row.document_id),
                    "text": row.text,
                    "doc_type": row.doc_type,
                    "title": row.title,
                    "publication_date": row.publication_date,
                    "page_start": row.page_start,
                    "page_end": row.page_end,
                    "section_hint": row.section_hint,
                    "semantic": 0.0,
                    "lexical": 0.0,
                },
            )
            entry["lexical"] = max(entry["lexical"], float(row.lexical_rank))

    semantic_max = max((item["semantic"] for item in combined.values()), default=0.0)
    lexical_max = max((item["lexical"] for item in combined.values()), default=0.0)

    results: List[RetrievedChunk] = []
    for item in combined.values():
        semantic_score = item["semantic"] / semantic_max if semantic_max > 0 else 0.0
        lexical_score = item["lexical"] / lexical_max if lexical_max > 0 else 0.0
        final_score = 0.0
        if semantic_weight > 0:
            final_score += semantic_weight * semantic_score
        if lexical_weight > 0:
            final_score += lexical_weight * lexical_score

        results.append(
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                doc_type=item["doc_type"],
                title=item["title"],
                publication_date=
                    item["publication_date"].isoformat() if item["publication_date"] else None,
                page_start=item["page_start"],
                page_end=item["page_end"],
                section_hint=item["section_hint"],
                score=final_score,
            )
        )

    # Step 3: Sort by hybrid score (tie-breaking by id) and optionally rerank
    results.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))

    # If reranking is disabled, return top-k by hybrid score
    if not rerank:
        logger.debug(f"Returning top-{limit} results (no reranking)")
        return results[:limit]

    # Step 4: Rerank using cross-encoder
    # Why lazy import? Reranker loads ~90MB model, only load if needed
    from app.rag.reranker import create_reranker

    logger.info(f"Reranking {len(results)} candidates to top-{limit}")

    try:
        # Initialize reranker (model loaded lazily on first use)
        reranker = create_reranker()

        # Convert RetrievedChunk to dict format expected by reranker
        # Why convert? Reranker expects dict with specific keys
        # (id, document_id, text, score, metadata)
        candidates = [
            {
                "id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "score": chunk.score,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_hint": chunk.section_hint,
            }
            for chunk in results
        ]

        # Rerank candidates using cross-encoder
        # This rescores each candidate based on query-document interaction
        reranked = reranker.rerank(
            query=query_text,
            chunks=candidates,
            top_k=limit
        )

        # Convert reranked results back to RetrievedChunk format
        # Why convert back? Pipeline expects RetrievedChunk dataclass
        final_results = []
        for ranked_chunk in reranked:
            # Find original chunk to get metadata (doc_type, title, publication_date)
            original = next(
                (r for r in results if r.chunk_id == ranked_chunk.chunk_id),
                None
            )
            if original is None:
                logger.warning(f"Reranked chunk {ranked_chunk.chunk_id} not found in original results")
                continue

            final_results.append(
                RetrievedChunk(
                    chunk_id=ranked_chunk.chunk_id,
                    document_id=ranked_chunk.document_id,
                    text=ranked_chunk.text,
                    doc_type=original.doc_type,
                    title=original.title,
                    publication_date=original.publication_date,
                    page_start=ranked_chunk.page_start,
                    page_end=ranked_chunk.page_end,
                    section_hint=ranked_chunk.section_hint,
                    # Use rerank_score as final score
                    # Why? Cross-encoder is more accurate than bi-encoder for ranking
                    score=ranked_chunk.rerank_score,
                )
            )

        logger.info(
            f"Reranking complete. Top result score: "
            f"{final_results[0].score:.3f} (was {results[0].score:.3f})"
        )

        return final_results

    except Exception as e:
        # If reranking fails, fall back to hybrid results
        # Why graceful degradation? Reranking is an enhancement, not critical
        logger.error(f"Reranking failed: {e}. Falling back to hybrid results")
        return results[:limit]
