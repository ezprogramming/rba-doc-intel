# Repository Guidelines

## Overview & Architecture Mandates
This platform ingests Reserve Bank of Australia PDFs, cleans text, builds embeddings, and serves a Streamlit RAG UI. The stack is fixed: Python 3.x + `uv`, PostgreSQL with `pgvector`, MinIO for object storage, and Streamlit for the UI. PDFs must be parsed with either `pymupdf` or `pdfplumber` (stick to one). Do **not** introduce new services (Kafka, Redis, alternate vector DBs, extra UIs) unless the spec in `CLAUDE.md` is updated.

## Project Structure & Modules
Follow the prescribed layout:
- `app/config.py` centralizes env loading.
- `app/db/` contains SQLAlchemy models and session helpers.
- `app/storage/` defines the MinIO adapter.
- `app/pdf/` handles parsing, cleaning, chunking.
- `app/embeddings/` owns embedding client + batch indexer.
- `app/rag/` includes retriever, LLM client, pipeline entry (`answer_query`).
- `app/ui/streamlit_app.py` renders the chat UI.
- `scripts/` hosts operational entry points (`crawler_rba.py`, `process_pdfs.py`, `build_embeddings.py`, `debug_dump.py`).
Never rename top-level directories or move modules without explicit approval.

## Environment, Build & Run Commands
- `docker compose build app && docker compose run --rm app uv sync` — prepare the containerized dev env.
- `docker compose up` — launch Postgres, MinIO, and the Streamlit app; the app container now runs `scripts/wait_for_services.py` followed by `scripts/bootstrap_db.py` so pgvector + schema migrations land automatically.
- `docker compose run --rm app uv run scripts/crawler_rba.py` — crawl RBA sites, push PDFs to MinIO, insert metadata.
- `docker compose run --rm app uv run scripts/process_pdfs.py` — pull pending docs, produce cleaned chunks.
- `docker compose run --rm app uv run scripts/build_embeddings.py` — fill missing pgvector embeddings.
- `docker compose run --rm app uv run streamlit run app/ui/streamlit_app.py` — start the UI manually if needed.
- `docker compose run --rm app uv run pytest` — execute the core test suite (lint optional via `ruff`).
Configure endpoints via `.env` (mirrors `.env.example`). Never hard-code credentials or paths; everything flows through `app/config.py`.

## Coding Style & Data Contracts
Use four-space indentation, PEP 8 naming, and thorough type hints. Keep SQLAlchemy models in `app/db/models.py` (documents, pages, chunks, chat tables) matching the schema in `CLAUDE.md`. Expose pure functions/classes in library modules and reserve side effects for scripts. Structure RAG responses as `{answer, evidence[], analysis}` and persist chat interactions per the spec. Documents must record `source_url`, `content_hash`, and `content_length` so repeated crawler runs can dedupe safely.

## Testing & Data Quality
Place tests under `tests/` mirroring module paths (`tests/rag/test_pipeline.py`, etc.). Prefer `pytest` fixtures to mock MinIO/Postgres; add lightweight integration tests that process sample PDFs end-to-end. Every change should keep `uv run pytest -q` green and cover regression points (crawler edge cases, chunking boundaries, retrieval filters). New helpers (e.g. cleaners, analysis formatters) should ship with unit coverage before touching ingestion/RAG code.

## Contribution Workflow & Security
Commit messages follow `<area>: <imperative summary>` and PRs must describe the change, list new env vars/commands, reference issues, and attach evidence (logs, UI screenshots). Flag schema or data migrations prominently. Secrets live in `.env` (gitignored) and flow through `app/config.py`. Validate storage/database connectivity before merging, and avoid uploading proprietary PDFs or sensitive excerpts to public channels.
