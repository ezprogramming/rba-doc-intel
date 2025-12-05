"""Embedding client using LiteLLM for unified provider access.

LiteLLM provides a unified embedding API for multiple providers:
- OpenAI: text-embedding-3-small, text-embedding-ada-002
- Ollama: ollama/nomic-embed-text, ollama/mxbai-embed-large
- Azure: azure/text-embedding-ada-002
- Hugging Face: huggingface/<model>
- And many more...

Configuration:
- EMBEDDING_MODEL_NAME: Model with provider prefix (e.g., "ollama/nomic-embed-text")
- EMBEDDING_API_BASE_URL: Base URL for API (e.g., "http://llm:11434" for Ollama)
- LLM_API_KEY: API key (optional for local providers)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# Suppress verbose litellm logging
litellm.set_verbose = False


@dataclass
class EmbeddingResponse:
    vectors: List[List[float]]


class EmbeddingClient:
    """LiteLLM-based embedding client with automatic retry logic.

    Features:
    - Unified API for multiple embedding providers
    - Exponential backoff retry for transient failures
    - Configurable timeout
    """

    def __init__(self):
        settings = get_settings()
        self._model_name = settings.embedding_model_name
        self._api_base = settings.embedding_api_base_url.rstrip("/")
        self._api_key = settings.llm_api_key or "not-needed"
        self._timeout = settings.embedding_api_timeout

        # Configure LiteLLM for Ollama
        if self._model_name.startswith("ollama/"):
            litellm.api_base = self._api_base

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def embed(self, texts: List[str]) -> EmbeddingResponse:
        """Generate embeddings for input texts with automatic retry.

        Args:
            texts: List of strings to embed

        Returns:
            EmbeddingResponse containing vectors

        Raises:
            RuntimeError: If all retry attempts fail
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")

        try:
            response = litellm.embedding(
                model=self._model_name,
                input=texts,
                api_base=self._api_base,
                api_key=self._api_key,
                timeout=self._timeout,
            )

            # Extract embeddings from response
            vectors = [item["embedding"] for item in response.data]

            logger.debug(f"Generated {len(vectors)} embeddings successfully")
            return EmbeddingResponse(vectors=vectors)

        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise RuntimeError(f"Embedding request failed: {e}") from e
