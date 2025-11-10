# RBA Document Intelligence Platform – Build Plan

## Phase 0 – Environment & Docker Baseline
- Author `pyproject.toml` + `uv.lock` (Python 3.11 target) with core deps: `sqlalchemy`, `pgvector`, `pymupdf`/`pdfplumber`, `boto3`, `requests`, `beautifulsoup4`, `streamlit`, `pytest`, etc.
- Create `.env.example` covering `DATABASE_URL`, `MINIO_*`, `EMBEDDING_MODEL_NAME`, `LLM_*`.
- Write `docker-compose.yml` with four services:
  1. `postgres` (pgvector-enabled) seeded via init script.
  2. `minio` + console, configured with buckets `rba-raw-pdf`, `rba-derived`.
  3. `embedding` – local FastAPI/Text-Embeddings server (CPU torch + sentence-transformers) hosting `nomic-ai/nomic-embed-text-v1.5` on port 8080 with a cached HF volume.
  4. `app` image built from `Dockerfile` (python:3.11-slim), running `uv sync` on build and mounting repo for live dev; default command `streamlit run app/ui/streamlit_app.py` after `wait_for_services.py` + `bootstrap_db.py`.
- Add helper script `scripts/wait_for_services.py` consumed by the app container before running migrations or Streamlit.

## Phase 1 – Core Skeleton
- Lay out directories exactly as spec (`app/config.py`, `app/db/`, `app/storage/`, `app/pdf/`, `app/embeddings/`, `app/rag/`, `app/ui/`, `scripts/`).
- Implement `app/config.py` reading env vars with defaults + validation.
- Build SQLAlchemy base models (`documents`, `pages`, `chunks`, `chat_sessions`, `chat_messages`) with pgvector column types in `app/db/models.py`; create session factory in `app/db/session.py`.
- Add MinIO storage adapter (`app/storage/base.py`, `minio_s3.py`) handling streaming upload/download/bucket ensure.
- Provide Alembic migrations or custom bootstrap script to create tables; wire into docker app entrypoint.

## Phase 2 – Ingestion & Processing Pipelines
- `scripts/crawler_rba.py`: crawl official SMP/FSR listings via `requests` + `BeautifulSoup`, dedupe via Postgres, store PDFs in MinIO (`raw/<doc_type>/filename.pdf`), mark `documents.status=NEW`. Allow optional `CRAWLER_YEAR_FILTER` env var so engineers can limit work to specific years during debugging.
- `app/pdf/parser.py`: stream PDF text extraction (choose pymupdf/pdfplumber).
- `app/pdf/cleaner.py`: normalize whitespace, strip headers/footers.
- `app/pdf/chunker.py`: token-aware chunking with page bounds + optional section hints.
- `scripts/process_pdfs.py`: iterate docs by status, extract pages, run cleaner/chunker, populate `pages`/`chunks`, update statuses (`TEXT_EXTRACTED`, `CHUNKS_BUILT`).

## Phase 3 – Embeddings & RAG Layer
- `app/embeddings/client.py`: pluggable embedding interface (env-driven base URL/model/API key + timeout) hitting the embedding service.
- `app/embeddings/indexer.py`: find chunks missing embeddings, batch call client, persist vectors.
- `scripts/build_embeddings.py`: CLI entry to run indexer; integrate into docker workflow (`docker compose run app uv run scripts/build_embeddings.py`).
- `app/rag/retriever.py`: similarity search via SQL query (pgvector `cosine_distance`) with filters (doc_type/date).
- `app/rag/llm_client.py`: generic chat/complete wrapper.
- `app/rag/pipeline.py`: `answer_query(query, session_id=None)` retrieving context, building prompt, calling LLM, persisting chat messages, returning `{answer, evidence[], analysis}`.

## Phase 4 – UI & Operational Tools
- `app/ui/streamlit_app.py`: chat interface, session list, evidence accordion, env-driven connectors; run inside docker app service.
- `scripts/debug_dump.py`: inspect DB rows, check document counts, verify storage connectivity.
- Logging/metrics: configure structured logging (JSON or simple text) to stdout for docker aggregation; optionally add lightweight health endpoint invoked via Streamlit sidebar diagnostics.

## Phase 5 – Testing, CI, and Docs
- Testing: add `tests/` mirroring modules with pytest fixtures mocking MinIO/Postgres (use Moto/localstack alternatives or temporary sqlite+fake storage); include integration test invoking crawler→processor→retriever on sample PDFs stored under `tests/fixtures`.
- CI: GitHub Actions workflow running `docker compose up -d postgres minio`, `docker compose run app uv run pytest -q`, plus lint (optional `ruff`, `mypy`).
- Docs: keep `claude.md` authoritative spec, maintain `AGENTS.md` for quick contributor onboarding, and expand `README.md` with docker commands (`docker compose up -d`, `docker compose run app uv run scripts/...`), troubleshooting, and env var descriptions.

## Operational Considerations
- Schedule ingestion scripts via cron/Kubernetes Jobs calling `docker compose run app ...`.
- Back up Postgres and MinIO volumes; document retention policies for raw PDFs vs derived artifacts.
- Define alerting for crawler failures or embedding backlog (simple CLI exit codes consumed by external scheduler).
