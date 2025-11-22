"""Generic LLM client wrapper."""

from __future__ import annotations

import json
from typing import Callable, List

import requests

from app.config import get_settings


class LLMClient:
    def __init__(self):
        settings = get_settings()
        self._base_url = settings.llm_api_base_url.rstrip("/")
        self._model_name = settings.llm_model_name
        self._api_key = settings.llm_api_key

    def _build_payload(self, system_prompt: str, messages: List[dict], stream: bool) -> dict:
        prompt_parts = [system_prompt]
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        full_prompt = "\n\n".join(prompt_parts)
        return {
            "model": self._model_name,
            "prompt": full_prompt,
            "stream": stream,
            # Performance optimizations for CPU inference
            "options": {
                "num_predict": 2048,  # Max tokens to generate (allow detailed responses)
                "num_ctx": 4096,  # Context window size (reduced from 32K default)
                "temperature": 0.2,  # Low temperature for factual, focused RAG responses
                "num_thread": 4,  # CPU threads (adjust based on available cores)
                "top_p": 0.9,  # Nucleus sampling for quality
            },
        }

    def complete(self, system_prompt: str, messages: List[dict]) -> str:
        payload = self._build_payload(system_prompt, messages, stream=False)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            headers=headers,
            timeout=600,  # Increased to 10 minutes for CPU inference
        )
        if response.status_code == 404:
            raise RuntimeError(
                "LLM model not found on the Ollama server. "
                "Run 'docker compose exec llm ollama pull {model}' and retry.".format(
                    model=self._model_name
                )
            )
        response.raise_for_status()
        data = response.json()
        return data["response"]

    def stream(
        self,
        system_prompt: str,
        messages: List[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        payload = self._build_payload(system_prompt, messages, stream=True)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        final_text = ""
        with requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            headers=headers,
            stream=True,
            timeout=600,  # Increased to 10 minutes for CPU inference
        ) as response:
            if response.status_code == 404:
                raise RuntimeError(
                    "LLM model not found on the Ollama server. "
                    "Run 'docker compose exec llm ollama pull {model}' and retry.".format(
                        model=self._model_name
                    )
                )
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                data = json.loads(raw_line.decode("utf-8"))
                if data.get("done"):
                    break
                token = data.get("response", "")
                if token:
                    final_text += token
                    if on_token:
                        on_token(token)
                if data.get("error"):
                    raise RuntimeError(data["error"])
        return final_text
