"""Centralized configuration loading for the RBA Document Intelligence Platform."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


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
    embedding_batch_size: int
    embedding_parallel_batches: int

    # PDF processing settings
    pdf_batch_size: int
    pdf_max_workers: int

    # Table extraction settings
    table_batch_size: int
    table_max_workers: int

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


def _get_positive_int(key: str, value: str, min_value: int = 1) -> int:
    """Parse and validate a positive integer from environment variable.

    Args:
        key: Environment variable name (for error messages)
        value: Raw string value from os.environ
        min_value: Minimum allowed value (default: 1)

    Returns:
        Validated positive integer

    Raises:
        ValueError: If value is not a positive integer or below min_value
    """
    try:
        parsed = int(value)
    except ValueError as e:
        raise ValueError(f"{key} must be an integer, got: {value}") from e

    if parsed < min_value:
        raise ValueError(f"{key} must be >= {min_value}, got: {parsed}")

    return parsed


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""

    required_keys = (
        "DATABASE_URL",
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET_RAW_PDF",
        "MINIO_BUCKET_DERIVED",
        "EMBEDDING_MODEL_NAME",
        "EMBEDDING_API_BASE_URL",
        "EMBEDDING_API_TIMEOUT",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_PARALLEL_BATCHES",
        "PDF_BATCH_SIZE",
        "PDF_MAX_WORKERS",
        "TABLE_BATCH_SIZE",
        "TABLE_MAX_WORKERS",
        "LLM_MODEL_NAME",
        "LLM_API_BASE_URL",
    )
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        database_url=os.environ["DATABASE_URL"],
        minio_endpoint=os.environ["MINIO_ENDPOINT"],
        minio_access_key=os.environ["MINIO_ACCESS_KEY"],
        minio_secret_key=os.environ["MINIO_SECRET_KEY"],
        minio_secure=_get_bool(os.environ.get("MINIO_SECURE")),
        minio_raw_bucket=os.environ["MINIO_BUCKET_RAW_PDF"],
        minio_derived_bucket=os.environ["MINIO_BUCKET_DERIVED"],
        embedding_model_name=os.environ["EMBEDDING_MODEL_NAME"],
        embedding_api_base_url=os.environ["EMBEDDING_API_BASE_URL"],
        embedding_api_timeout=_get_positive_int(
            "EMBEDDING_API_TIMEOUT", os.environ["EMBEDDING_API_TIMEOUT"], min_value=1
        ),
        embedding_batch_size=_get_positive_int(
            "EMBEDDING_BATCH_SIZE", os.environ["EMBEDDING_BATCH_SIZE"], min_value=1
        ),
        embedding_parallel_batches=_get_positive_int(
            "EMBEDDING_PARALLEL_BATCHES", os.environ["EMBEDDING_PARALLEL_BATCHES"], min_value=1
        ),
        pdf_batch_size=_get_positive_int(
            "PDF_BATCH_SIZE", os.environ["PDF_BATCH_SIZE"], min_value=1
        ),
        pdf_max_workers=_get_positive_int(
            "PDF_MAX_WORKERS", os.environ["PDF_MAX_WORKERS"], min_value=1
        ),
        table_batch_size=_get_positive_int(
            "TABLE_BATCH_SIZE", os.environ["TABLE_BATCH_SIZE"], min_value=1
        ),
        table_max_workers=_get_positive_int(
            "TABLE_MAX_WORKERS", os.environ["TABLE_MAX_WORKERS"], min_value=1
        ),
        llm_model_name=os.environ["LLM_MODEL_NAME"],
        llm_api_base_url=os.environ["LLM_API_BASE_URL"],
        llm_api_key=os.environ.get("LLM_API_KEY"),
        # Reranking configuration (optional)
        # Why disabled by default? Adds latency, not all users need improved accuracy
        # Enable with: USE_RERANKING=1 in .env
        use_reranking=_get_bool(os.environ.get("USE_RERANKING"), default=False),
        reranker_model_name=os.environ.get("RERANKER_MODEL_NAME"),  # None = use default
        reranker_device=os.environ.get("RERANKER_DEVICE"),  # None = auto-detect
        reranker_batch_size=_get_positive_int(
            "RERANKER_BATCH_SIZE", os.environ.get("RERANKER_BATCH_SIZE", "32"), min_value=1
        ),
    )
