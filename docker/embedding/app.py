"""Minimal embedding inference server with GPU acceleration support.

Supports:
- Apple Silicon (M1/M2/M3/M4) via MPS backend
- NVIDIA GPUs via CUDA
- CPU fallback
"""

from __future__ import annotations

import logging
import os
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "nomic-ai/nomic-embed-text-v1.5")
BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
# Optional override: cuda | mps | cpu
PREFERRED_DEVICE = os.environ.get("EMBEDDING_DEVICE", "").strip().lower()


def get_device() -> str:
    """Choose best device with optional override via EMBEDDING_DEVICE."""
    def _warn_fallback(target: str) -> None:
        logger.warning(f"Requested device '{target}' not available; falling back to auto-detect")

    # Honor explicit override if available
    if PREFERRED_DEVICE:
        if PREFERRED_DEVICE == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using NVIDIA GPU (override): {gpu_name}")
            return "cuda"
        if PREFERRED_DEVICE == "mps" and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon GPU (override)")
            return "mps"
        if PREFERRED_DEVICE == "cpu":
            logger.info("Using CPU (override)")
            return "cpu"
        _warn_fallback(PREFERRED_DEVICE)

    # Auto-detect
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using NVIDIA GPU: {gpu_name}")
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU (Metal Performance Shaders)")
        return "mps"

    logger.warning("No GPU detected, using CPU (this will be slower)")
    return "cpu"


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: List[str] = Field(..., description="Texts to embed")


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    index: int


# Initialize FastAPI app
app = FastAPI(title="Local Embedding Server", version="1.1.0")

# Detect device and load model
DEVICE = get_device()
logger.info(f"Loading model: {MODEL_ID}")
model = SentenceTransformer(MODEL_ID, device=DEVICE, trust_remote_code=True)
logger.info(f"Model loaded successfully on {DEVICE}")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "batch_size": BATCH_SIZE
    }


@app.post("/embeddings")
def embeddings(payload: EmbeddingRequest) -> dict[str, List[EmbeddingResponse]]:
    """Generate embeddings for input texts.

    Args:
        payload: Request containing list of texts to embed

    Returns:
        Dictionary with embeddings and metadata

    Raises:
        HTTPException: If input is empty or embedding fails
    """
    if not payload.input:
        raise HTTPException(
            status_code=400,
            detail="Input must contain at least one string"
        )

    try:
        # Generate embeddings with L2 normalization for better cosine similarity
        vectors = model.encode(
            payload.input,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Enable L2 normalization
            show_progress_bar=False
        )

        data = [
            EmbeddingResponse(embedding=vec.tolist(), index=i)
            for i, vec in enumerate(vectors)
        ]

        logger.debug(f"Generated {len(data)} embeddings")
        return {"data": data}

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )
