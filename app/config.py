\"\"\"Centralized configuration loading for the RBA Document Intelligence Platform.\"\"\"

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    \"\"\"Immutable container for environment-driven settings.\"\"\"

    database_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_raw_bucket: str
    minio_derived_bucket: str
    embedding_model_name: str
    llm_model_name: str
    llm_api_base_url: str
    llm_api_key: str | None = None


def _get_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    \"\"\"Load settings once per process.\"\"\"

    missing = [key for key in (\"DATABASE_URL\", \"MINIO_ENDPOINT\") if not os.getenv(key)]
    if missing:
        raise RuntimeError(f\"Missing required environment variables: {', '.join(missing)}\")

    return Settings(
        database_url=os.environ[\"DATABASE_URL\"],
        minio_endpoint=os.environ[\"MINIO_ENDPOINT\"],
        minio_access_key=os.environ.get(\"MINIO_ACCESS_KEY\", \"minioadmin\"),
        minio_secret_key=os.environ.get(\"MINIO_SECRET_KEY\", \"minioadmin\"),
        minio_secure=_get_bool(os.environ.get(\"MINIO_SECURE\")),
        minio_raw_bucket=os.environ.get(\"MINIO_BUCKET_RAW_PDF\", \"rba-raw-pdf\"),
        minio_derived_bucket=os.environ.get(\"MINIO_BUCKET_DERIVED\", \"rba-derived\"),
        embedding_model_name=os.environ.get(\"EMBEDDING_MODEL_NAME\", \"text-embedding-default\"),
        llm_model_name=os.environ.get(\"LLM_MODEL_NAME\", \"command-rag\"),
        llm_api_base_url=os.environ.get(\"LLM_API_BASE_URL\", \"http://localhost:8000\"),  # placeholder
        llm_api_key=os.environ.get(\"LLM_API_KEY\"),
    )
