"""Centralized configuration loading for the RBA Document Intelligence Platform."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    """Immutable container for environment-driven settings."""

    database_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_raw_bucket: str
    minio_derived_bucket: str
    embedding_model_name: str
    embedding_api_base_url: str
    embedding_api_timeout: int
    llm_model_name: str
    llm_api_base_url: str
    llm_api_key: str | None = None

    # Reranking settings (optional, for improved retrieval accuracy)
    # Why configurable? Production systems may want reranking, dev may not (faster iteration)
    use_reranking: bool = False  # Enable cross-encoder reranking
    reranker_model_name: str | None = None  # HuggingFace model ID (None = use default)
    reranker_device: str | None = None  # 'cpu', 'cuda', 'mps', or None for auto-detect
    reranker_batch_size: int = 32  # Batch size for reranking (higher = faster but more memory)


def _get_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""

    missing = [key for key in ("DATABASE_URL", "MINIO_ENDPOINT") if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        database_url=os.environ["DATABASE_URL"],
        minio_endpoint=os.environ["MINIO_ENDPOINT"],
        minio_access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        minio_secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        minio_secure=_get_bool(os.environ.get("MINIO_SECURE")),
        minio_raw_bucket=os.environ.get("MINIO_BUCKET_RAW_PDF", "rba-raw-pdf"),
        minio_derived_bucket=os.environ.get("MINIO_BUCKET_DERIVED", "rba-derived"),
        embedding_model_name=os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-default"),
        embedding_api_base_url=os.environ.get("EMBEDDING_API_BASE_URL")
        or os.environ.get("LLM_API_BASE_URL", "http://localhost:8000"),
        embedding_api_timeout=int(os.environ.get("EMBEDDING_API_TIMEOUT", "120")),
        # Default to qwen2.5:7b (4.7B params, good balance of speed vs quality)
        # Why qwen2.5:7b over 1.5b?
        # - 3-5x better reasoning on complex queries
        # - Better citation quality and numerical understanding
        # - Acceptable latency (1-2s response time on M4)
        # Alternative: llama3.3:8b for similar performance
        llm_model_name=os.environ.get("LLM_MODEL_NAME", "qwen2.5:7b"),
        llm_api_base_url=os.environ.get("LLM_API_BASE_URL", "http://localhost:8000"),  # placeholder
        llm_api_key=os.environ.get("LLM_API_KEY"),
        # Reranking configuration (optional)
        # Why disabled by default? Adds latency, not all users need improved accuracy
        # Enable with: USE_RERANKING=1 in .env
        use_reranking=_get_bool(os.environ.get("USE_RERANKING"), default=False),
        reranker_model_name=os.environ.get("RERANKER_MODEL_NAME"),  # None = use default
        reranker_device=os.environ.get("RERANKER_DEVICE"),  # None = auto-detect
        reranker_batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "32")),
    )
