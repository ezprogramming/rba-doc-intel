"""Chunk retrieval logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document


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
) -> List[RetrievedChunk]:
    """Return the top-k chunks using hybrid semantic + lexical retrieval."""

    combined: Dict[int, dict] = {}

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
        .limit(limit * 2)
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

    query_text = (query_text or "").strip()
    if lexical_weight > 0 and query_text:
        ts_document = func.to_tsvector("english", Chunk.text)
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
            .where(lexical_score > 0)
            .order_by(desc(lexical_score))
            .limit(limit * 2)
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

    results.sort(key=lambda chunk: chunk.score, reverse=True)
    return results[:limit]
