"""Generic LLM client wrapper."""

from __future__ import annotations

from typing import List

import requests

from app.config import get_settings


class LLMClient:
    def __init__(self):
        settings = get_settings()
        self._base_url = settings.llm_api_base_url.rstrip("/")
        self._model_name = settings.llm_model_name
        self._api_key = settings.llm_api_key

    def complete(self, system_prompt: str, messages: List[dict]) -> str:
        payload = {"model": self._model_name, "system": system_prompt, "messages": messages}
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = requests.post(f"{self._base_url}/chat/completions", json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

