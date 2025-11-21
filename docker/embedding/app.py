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
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

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

# Load tokenizer and model using transformers (more control than sentence-transformers)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)

# Set model to evaluation mode (no training)
model.eval()

MAX_SEQ_LENGTH = 8192
logger.info(f"Model loaded successfully on {DEVICE} (max_seq_length: {MAX_SEQ_LENGTH})")


def mean_pooling(model_output, attention_mask):
    """Mean pooling - take average of all token embeddings, ignoring padding.

    This is the standard pooling strategy for BERT-based models.
    """
    token_embeddings = model_output[0]  # First element: token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


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
        # Tokenize with proper padding and truncation
        # padding='longest' = pad all sequences to length of longest in batch (efficient)
        # truncation=True = truncate sequences longer than max_length
        encoded_input = tokenizer(
            payload.input,
            padding='longest',  # Dynamic padding to longest in batch
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors='pt'
        ).to(DEVICE)

        # Generate embeddings without gradient computation (faster)
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Apply mean pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings (L2 normalization for cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Convert to list
        vectors = embeddings.cpu().numpy()

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
