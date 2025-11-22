"""Embedding client abstraction with retry logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResponse:
    vectors: List[List[float]]


class EmbeddingClient:
    """HTTP-based embedding model client with automatic retry logic.

    Features:
    - Exponential backoff retry for transient failures
    - Configurable timeout
    - Support for API key authentication
    """

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.embedding_api_base_url.rstrip("/")
        self._model_name = settings.embedding_model_name
        self._api_key = settings.llm_api_key
        self._timeout = settings.embedding_api_timeout

    @retry(
        retry=retry_if_exception_type(
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            )
        ),
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
            requests.exceptions.RequestException: If all retry attempts fail
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")

        payload = {"model": self._model_name, "input": texts}
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        try:
            response = requests.post(
                f"{self._base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            vectors = [item["embedding"] for item in data["data"]]

            logger.debug(f"Generated {len(vectors)} embeddings successfully")
            return EmbeddingResponse(vectors=vectors)

        except requests.exceptions.HTTPError as e:
            logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding request failed: {e}")
            raise
