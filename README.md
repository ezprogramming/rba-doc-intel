# RBA Document Intelligence Platform

Local-first setup for crawling, processing, and querying Reserve Bank of Australia PDF publications via a Retrieval-Augmented Generation (RAG) workflow.

## Prerequisites

- Docker & Docker Compose v2
- `uv` (optional for running scripts on the host)

## Quick Start

1. Copy `.env.example` to `.env` and adjust credentials if needed.
2. Build and install dependencies inside the app image:

   ```bash
   docker compose build app
   docker compose run --rm app uv sync
   ```

3. Launch the stack (Postgres + MinIO + Streamlit app):

   ```bash
   docker compose up
   ```

Streamlit will be reachable on `http://localhost:${STREAMLIT_SERVER_PORT:-8501}`.

4. Start the embedding and LLM services (run once, keep them running while you work):

   ```bash
   docker compose up -d embedding llm
   # Pull the lightweight multilingual LLM once
   docker compose exec llm ollama pull qwen2:7b
   ```

If you prefer to ensure the schema manually outside of `docker compose up`, run:

```bash
docker compose run --rm app uv run python scripts/bootstrap_db.py
```

## Running Pipelines

Use the same container for operational scripts so dependencies stay consistent:

```bash
docker compose run --rm app uv run scripts/crawler_rba.py
docker compose run --rm app uv run scripts/process_pdfs.py
docker compose run --rm app uv run scripts/build_embeddings.py
# Or run them all sequentially:
docker compose run --rm app uv run python scripts/refresh_pdfs.py
```

Set `CRAWLER_YEAR_FILTER` in `.env` (for example, `CRAWLER_YEAR_FILTER=2024`) to limit ingestion to specific years while debugging. The crawler remains idempotent, so you can widen or clear the filter later and rerun the same commands to backfill the rest of the corpus.

`scripts/debug_dump.py` prints current document/page/chunk counts for quick sanity checks.

## Testing & Linting

```bash
docker compose run --rm app uv run pytest
docker compose run --rm app uv run ruff check
```

## Regenerating Dependencies

After editing `pyproject.toml`, rebuild or re-run `uv sync` inside the container to refresh the virtual environment. Commit the updated `uv.lock` once you have synced successfully.
