"""LLM client wrapper using LiteLLM for unified provider access.

LiteLLM provides a unified API to call 100+ LLM providers:
- Ollama: ollama/llama3, ollama/mistral, ollama/phi3
- OpenAI: gpt-4, gpt-3.5-turbo
- Anthropic: claude-3-sonnet-20240229, claude-3-opus-20240229
- Azure: azure/gpt-4, azure/gpt-35-turbo
- And many more...

Configuration:
- LLM_MODEL_NAME: Model name with provider prefix (e.g., "ollama/llama3")
- LLM_API_BASE_URL: Base URL for the API (e.g., "http://llm:11434" for Ollama)
- LLM_API_KEY: API key (optional for local providers like Ollama)
"""

from __future__ import annotations

import logging
from typing import Callable, List

import litellm
from litellm import completion

from app.config import get_settings

logger = logging.getLogger(__name__)

# Suppress verbose litellm logging
litellm.set_verbose = False


class LLMClient:
    """Unified LLM client using LiteLLM."""

    def __init__(self):
        settings = get_settings()
        self._model_name = settings.llm_model_name
        self._api_base = settings.llm_api_base_url.rstrip("/")
        self._api_key = settings.llm_api_key or "not-needed"

        # Configure LiteLLM for the provider
        # Ollama needs special base URL handling
        if self._model_name.startswith("ollama/"):
            litellm.api_base = self._api_base

    def _build_messages(self, system_prompt: str, messages: List[dict]) -> List[dict]:
        """Convert to LiteLLM message format."""
        result = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            result.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            )
        return result

    def complete(self, system_prompt: str, messages: List[dict]) -> str:
        """Generate a completion (non-streaming).

        Args:
            system_prompt: System instructions for the LLM
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Generated text response
        """
        formatted_messages = self._build_messages(system_prompt, messages)

        try:
            response = completion(
                model=self._model_name,
                messages=formatted_messages,
                api_base=self._api_base,
                api_key=self._api_key,
                timeout=600,  # 10 minutes for CPU inference
                temperature=0.2,  # Low temperature for factual RAG responses
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    def stream(
        self,
        system_prompt: str,
        messages: List[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Generate a streaming completion.

        Args:
            system_prompt: System instructions for the LLM
            messages: List of message dicts with 'role' and 'content'
            on_token: Optional callback invoked for each token

        Returns:
            Complete generated text
        """
        formatted_messages = self._build_messages(system_prompt, messages)
        final_text = ""

        try:
            response = completion(
                model=self._model_name,
                messages=formatted_messages,
                api_base=self._api_base,
                api_key=self._api_key,
                timeout=600,
                temperature=0.2,
                max_tokens=2048,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    final_text += delta
                    if on_token:
                        on_token(delta)

            return final_text

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise RuntimeError(f"LLM streaming failed: {e}") from e
