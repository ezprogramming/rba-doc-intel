"\"\"\"Embedding backfill pipeline.\"\"\""

from __future__ import annotations

from typing import Iterable, List

from sqlalchemy import select

from app.db.models import Chunk
from app.db.session import session_scope
from app.embeddings.client import EmbeddingClient


def generate_missing_embeddings(batch_size: int = 32) -> int:
    """Populate embeddings for chunks where the vector is null."""
    client = EmbeddingClient()
    updated = 0

    with session_scope() as session:
        stmt = (
            select(Chunk)
            .where(Chunk.embedding.is_(None))
            .order_by(Chunk.id)
            .limit(batch_size)
        )
        chunks = session.scalars(stmt).all()
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        response = client.embed(texts)
        for chunk, vector in zip(chunks, response.vectors):
            chunk.embedding = vector
            updated += 1

    return updated

