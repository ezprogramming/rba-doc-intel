"""Embedding client abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import requests

from app.config import get_settings


@dataclass
class EmbeddingResponse:
    vectors: List[List[float]]


class EmbeddingClient:
    """Simple HTTP-based embedding model client."""

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.embedding_api_base_url.rstrip("/")
        self._model_name = settings.embedding_model_name
        self._api_key = settings.llm_api_key
        self._timeout = settings.embedding_api_timeout

    def embed(self, texts: List[str]) -> EmbeddingResponse:
        payload = {"model": self._model_name, "input": texts}
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = requests.post(
            f"{self._base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        vectors = [item["embedding"] for item in data["data"]]
        return EmbeddingResponse(vectors=vectors)
