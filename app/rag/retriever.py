"""Chunk retrieval logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sqlalchemy import select
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
    score: float


def retrieve_similar_chunks(
    session: Session, query_embedding: Sequence[float], limit: int = 5
) -> List[RetrievedChunk]:
    """Return the top-k most similar chunks using pgvector cosine distance."""

    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.text,
            Chunk.page_start,
            Chunk.page_end,
            Document.doc_type,
            Document.title,
            Document.publication_date,
            distance.label("distance"),
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )

    rows = session.execute(stmt).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=str(row.document_id),
            text=row.text,
            doc_type=row.doc_type,
            title=row.title,
            publication_date=row.publication_date.isoformat() if row.publication_date else None,
            page_start=row.page_start,
            page_end=row.page_end,
            score=float(row.distance),
        )
        for row in rows
    ]
