"""Reranking retrieved chunks using cross-encoder models.

Why reranking?
==============
Initial retrieval (bi-encoder embeddings) is fast but sometimes imprecise:
- Bi-encoders: Encode query and documents separately, then compute similarity
  → Fast (just vector math) but less accurate (can't see query-document interaction)
- Cross-encoders: Encode query + document TOGETHER, then classify relevance
  → Slower (need full forward pass per pair) but much more accurate

Typical RAG pipeline:
1. Bi-encoder retrieves top-100 candidates (fast, ~10ms)
2. Cross-encoder reranks top-100 → top-10 (slower, ~500ms, but only on candidates)
3. LLM uses top-10 for generation

Performance impact:
- Retrieval recall@100: ~85% (bi-encoder gets relevant docs in top-100)
- Reranking precision@10: +25-40% improvement (cross-encoder picks best 10)
- End-to-end answer quality: +15-25% (measured by human eval, keyword match)

Why not use cross-encoder for initial retrieval?
- Too slow: 10,000 chunks × cross-encoder = 10,000 forward passes = minutes
- Bi-encoder: 10,000 chunks × vector similarity = milliseconds

Industry examples:
- Pinecone: Recommends bi-encoder (initial) + cross-encoder (rerank)
- Cohere: Offers rerank API (cross-encoder as a service)
- Weaviate: Hybrid search + reranking module
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Default cross-encoder model for reranking
# Why ms-marco-MiniLM-L-6-v2?
# - Trained on MS MARCO passage ranking dataset (1M+ query-document pairs)
# - Small: 6-layer MiniLM (22M params, ~90MB)
# - Fast: ~20-30ms per query-document pair on CPU, ~5ms on GPU
# - Good accuracy: 0.39 MRR@10 on MS MARCO dev (comparable to much larger models)
# Alternative: cross-encoder/ms-marco-MiniLM-L-12-v2 (better accuracy, 2x slower)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RankedChunk:
    """A chunk with its reranking score.

    Fields:
        chunk_id: Database ID of the chunk
        document_id: Parent document UUID
        text: Chunk text content
        page_start: First page number
        page_end: Last page number
        section_hint: Optional section/heading hint
        original_score: Initial retrieval score (cosine similarity, BM25, or hybrid)
        rerank_score: Cross-encoder relevance score (-10 to +10, higher = more relevant)
        rank: Position in reranked list (1 = most relevant)

    Why separate original_score and rerank_score?
    - Debugging: See if reranker agrees with initial retrieval
    - Blending: Could combine both scores (e.g., 0.7*rerank + 0.3*original)
    - Analytics: Measure how much reranking changes the order
    """

    chunk_id: int
    document_id: str
    text: str
    page_start: int
    page_end: int
    section_hint: Optional[str]
    original_score: float
    rerank_score: float
    rank: int


class Reranker:
    """Reranks retrieved chunks using a cross-encoder model.

    Usage:
        reranker = Reranker()

        # After initial retrieval (bi-encoder or hybrid)
        initial_results = retriever.search(query, top_k=100)

        # Rerank to get best 10
        reranked = reranker.rerank(
            query="What is the RBA's inflation target?",
            chunks=initial_results,
            top_k=10
        )

        # Use top-10 for LLM context
        context = "\n\n".join([chunk.text for chunk in reranked])

    Implementation details:
    - Lazy loading: Model loaded only on first rerank() call (saves memory if unused)
    - Batch scoring: Scores all candidates in one forward pass (faster than sequential)
    - GPU support: Auto-detects CUDA/MPS (same as embedding service)
    """

    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None, batch_size: int = 32
    ):
        """Initialize reranker (model loaded lazily on first use).

        Args:
            model_name: HuggingFace cross-encoder model ID.
                       Defaults to ms-marco-MiniLM-L-6-v2 (fast, good accuracy).
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto-detect).
            batch_size: Number of query-document pairs to score per batch.
                       Higher = faster but more memory.
                       Default 32 works well on 16GB RAM systems.

        Why lazy loading?
        - Reranking might be optional (disabled in config)
        - Model is ~90MB, no need to load if never used
        - Allows fast service startup even if reranking is slow to initialize
        """
        self.model_name = model_name or DEFAULT_RERANKER_MODEL
        self.device = device  # None = auto-detect on first load
        self.batch_size = batch_size
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        """Load cross-encoder model (called lazily on first rerank)."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading cross-encoder model: {self.model_name}")

        # Auto-detect device if not specified
        # Priority: CUDA > MPS > CPU (same as embedding service)
        device = self.device
        if device is None:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using NVIDIA GPU for reranking")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Silicon GPU (MPS) for reranking")
            else:
                device = "cpu"
                logger.warning("No GPU detected, using CPU for reranking (slower)")

        # Load cross-encoder
        # Why default_activation_function=None?
        # - MS MARCO models are already trained with sigmoid output
        # - Raw logits give better calibrated scores for ranking
        self._model = CrossEncoder(
            self.model_name,
            device=device,
            default_activation_function=None,  # Use raw logits
        )

        logger.info(f"Cross-encoder loaded on device: {device}")
        return self._model

    def rerank(self, query: str, chunks: List[dict], top_k: int = 10) -> List[RankedChunk]:
        """Rerank chunks by query-document relevance using cross-encoder.

        Args:
            query: User query text
            chunks: Initial retrieval results (from retriever.search())
                   Each chunk dict must have:
                   - id: chunk ID
                   - document_id: parent document UUID
                   - text: chunk content
                   - score: initial retrieval score
                   - page_start, page_end, section_hint: metadata
            top_k: Number of top results to return after reranking

        Returns:
            List of RankedChunk objects, sorted by rerank_score (descending)

        How it works:
        1. Create query-document pairs: [(query, chunk1.text), (query, chunk2.text), ...]
        2. Score all pairs in batches using cross-encoder
        3. Sort by cross-encoder score (descending)
        4. Return top-k

        Performance example (100 candidates → 10 results):
        - CPU: ~2-3 seconds
        - GPU (NVIDIA/M4): ~200-500ms
        - Accuracy gain: +25-40% precision@10
        """
        if not chunks:
            logger.warning("No chunks to rerank, returning empty list")
            return []

        # Load model (lazy, only on first call)
        model = self._load_model()

        # Create query-document pairs for cross-encoder
        # Format: [(query, doc1), (query, doc2), ...]
        # Why this format? CrossEncoder.predict() expects list of text pairs
        pairs = [(query, chunk["text"]) for chunk in chunks]

        logger.info(f"Reranking {len(chunks)} chunks with cross-encoder")

        # Score all pairs in batches
        # Why batch_size? Trade-off between speed and memory
        # - Larger batches = faster (GPU parallelism) but more memory
        # - Smaller batches = slower but safer for limited RAM
        # Default 32 is good balance for most systems
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,  # Avoid cluttering logs
        )

        # Combine chunks with their rerank scores
        ranked_chunks = []
        for idx, (chunk, rerank_score) in enumerate(zip(chunks, scores, strict=True)):
            ranked_chunks.append(
                RankedChunk(
                    chunk_id=chunk["id"],
                    document_id=chunk["document_id"],
                    text=chunk["text"],
                    page_start=chunk.get("page_start", 0),
                    page_end=chunk.get("page_end", 0),
                    section_hint=chunk.get("section_hint"),
                    original_score=chunk.get("score", 0.0),
                    rerank_score=float(rerank_score),
                    rank=idx + 1,  # Will be updated after sorting
                )
            )

        # Sort by rerank score (descending)
        # Why not use original_score? Cross-encoder is more accurate for final ranking
        ranked_chunks.sort(key=lambda x: x.rerank_score, reverse=True)

        # Update rank positions after sorting
        for idx, chunk in enumerate(ranked_chunks[:top_k]):
            chunk.rank = idx + 1

        # Log reranking statistics
        if ranked_chunks:
            logger.info(
                f"Reranking complete. Top score: {ranked_chunks[0].rerank_score:.3f}, "
                f"Bottom score: {ranked_chunks[-1].rerank_score:.3f}"
            )

            # Log how much reranking changed the order
            # Example: If original rank 50 becomes rank 1, that's a big change
            original_rank_of_top = next(
                (i for i, c in enumerate(chunks) if c["id"] == ranked_chunks[0].chunk_id), -1
            )
            if original_rank_of_top > 5:
                logger.info(
                    f"Reranking promoted chunk from position {original_rank_of_top + 1} to 1 "
                    f"(score improved from {ranked_chunks[0].original_score:.3f} "
                    f"to {ranked_chunks[0].rerank_score:.3f})"
                )

        return ranked_chunks[:top_k]


def create_reranker() -> Reranker:
    """Factory function to create reranker with config from environment.

    Why factory function?
    - Centralized configuration loading
    - Easier to mock in tests
    - Can add caching/singleton pattern if needed

    Environment variables (optional):
        RERANKER_MODEL_NAME: HuggingFace model ID (default: ms-marco-MiniLM-L-6-v2)
        RERANKER_DEVICE: Device to run on (default: auto-detect)
        RERANKER_BATCH_SIZE: Batch size for scoring (default: 32)

    Usage:
        from app.rag.reranker import create_reranker

        reranker = create_reranker()
        results = reranker.rerank(query, chunks, top_k=10)
    """
    import os

    return Reranker(
        model_name=os.getenv("RERANKER_MODEL_NAME"),  # None = use default
        device=os.getenv("RERANKER_DEVICE"),  # None = auto-detect
        batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "32")),
    )
