"""Embedding backfill pipeline."""

from __future__ import annotations

import os
from typing import Set
from uuid import UUID

from sqlalchemy import func, select

from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import session_scope
from app.embeddings.client import EmbeddingClient


def generate_missing_embeddings(batch_size: int | None = None) -> int:
    """Populate embeddings for chunks where the vector is null."""
    if batch_size is None:
        batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
    client = EmbeddingClient()
    updated = 0

    with session_scope() as session:
        stmt = (
            select(Chunk)
            .where(Chunk.embedding.is_(None))
            .order_by(Chunk.id)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        chunks = session.scalars(stmt).all()
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        response = client.embed(texts)
        updated_doc_ids: Set[UUID] = set()
        for chunk, vector in zip(chunks, response.vectors):
            chunk.embedding = vector
            updated += 1
            updated_doc_ids.add(chunk.document_id)

        for doc_id in updated_doc_ids:
            remaining = session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_id == doc_id, Chunk.embedding.is_(None))
            )
            if remaining == 0:
                document = session.get(Document, doc_id)
                if document:
                    document.status = DocumentStatus.EMBEDDED.value

    return updated
