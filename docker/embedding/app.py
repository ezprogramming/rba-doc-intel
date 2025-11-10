"""Minimal embedding inference server for local development."""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("MODEL_ID", "nomic-ai/nomic-embed-text-v1.5")
BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: List[str] = Field(..., description="Texts to embed")


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    index: int


app = FastAPI(title="Local Embedding Server", version="1.0.0")
model = SentenceTransformer(MODEL_ID, trust_remote_code=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/embeddings")
def embeddings(payload: EmbeddingRequest) -> dict[str, List[EmbeddingResponse]]:
    if not payload.input:
        raise HTTPException(status_code=400, detail="Input must contain at least one string")
    vectors = model.encode(payload.input, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=False)
    data = [EmbeddingResponse(embedding=vec.tolist(), index=i) for i, vec in enumerate(vectors)]
    return {"data": data}
