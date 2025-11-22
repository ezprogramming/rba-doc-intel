# RBA Document Intelligence Platform

Local-first setup for crawling, processing, and querying Reserve Bank of Australia PDF publications via a Retrieval-Augmented Generation (RAG) workflow.

## Prerequisites

- Docker & Docker Compose v2
- `uv` (optional for running scripts on the host)

## Quick Start

1. Copy `.env.example` to `.env` and fill in **all** required values (Postgres DSN, MinIO buckets/keys, embedding/LLM endpoints). Defaults have been removed to avoid accidental local creds leaking into containers.
2. Build and install dependencies inside the app image:

   ```bash
   make bootstrap
   ```

   Run `make help` anytime to list all available targets; pass extra script-specific flags via `ARGS="..."` (for example, `make embeddings ARGS="--reset"`).

3. Launch the stack (Postgres + MinIO + Streamlit app):

   ```bash
   make up
   ```

Streamlit will be reachable on `http://localhost:${STREAMLIT_SERVER_PORT:-8501}` with live token streaming and thumbs up/down feedback on every response.

4. Start the embedding and LLM services (run once, keep them running while you work):

   ```bash
   make up-models
   # Pull the lightweight multilingual LLM once (1.5B optimized for CPU)
   make llm-pull MODEL=qwen2.5:1.5b
   ```

## Running Pipelines

Use the same container for operational scripts so dependencies stay consistent:

```bash
make crawl
make process
make tables            # new standalone Camelot stage (lattice + stream)
make embeddings        # uses EMBEDDING_BATCH_SIZE / EMBEDDING_PARALLEL_BATCHES from .env
# Or run them all sequentially:
make refresh
```

### Table extraction workflow

- Run `make tables` (optionally with `ARGS="--force"` or `ARGS="--document-id <uuid>"`) after `make process` to extract structured Camelot tables and generate enriched table chunks. Each chunk now includes a caption, column list, row summaries, inferred metric tags, and a `table_id` back-reference so the UI/pipeline can fetch the precise structured rows (stored in the `tables` table) for citations.
- After tuning the formatter in `scripts/extract_tables.py`, rerun `make tables ARGS="--force"` followed by `make embeddings` so the updated text is re-embedded. Evidence payloads now surface both the enriched chunk text **and** the underlying JSON rows, enabling downstream verification or CSV renders without a separate lookup.
- Existing deployments should apply the lightweight index migration before reprocessing tables to avoid btree size errors:  
  `docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /app/docker/postgres/initdb.d/05_rebuild_chunk_index.sql`

### Chunking & Retrieval Defaults

- **Chunk window:** 768-token cap with ~15 % overlap. This mirrors Pinecone/Anthropic guidance for production RAG – big enough for narrative coherence, small enough to keep embedding latency down.
- **Section hints:** The chunker scans the first 200 characters for headings (`3.2 Inflation`, `Chapter 4`, `Box A`) and stores them in `chunks.section_hint`. The Streamlit UI surfaces these hints inside the evidence expander so analysts immediately see where a quote came from.
- **Hybrid similarity search:** Retrieval fuses cosine similarity (`pgvector` HNSW index) with Postgres full-text scores (`ts_rank_cd` on a persisted `tsvector` column). This is the “semantic + lexical” hybrid pattern Pinecone, Weaviate, and Cohere now recommend for enterprise search because identifiers/dates often rely on raw keyword matches.

### Observability hooks

- `app/rag/hooks.py` exposes a light pub/sub bus. The RAG pipeline emits lifecycle events (`rag:query_started`, `rag:retrieval_complete`, `rag:stream_chunk`, `rag:answer_completed`) and the Streamlit UI emits `ui:question_submitted`, `ui:answer_rendered`, `ui:message_committed`, and `ui:feedback_recorded`.
- Subscribe anywhere (scripts, tests) to tap into these events without touching business logic:

  ```python
  from app.rag.hooks import hooks

  hooks.subscribe("rag:answer_completed", lambda event, payload: print(payload))
  ```

- A default debug-level subscriber is registered, so setting `LOG_LEVEL=DEBUG` shows the hook stream in container logs.

### Re-embedding after chunk strategy changes

Whenever you tweak chunk sizes/cleaning you should wipe the old vectors so embeddings reflect the new text spans:

```bash
make embeddings-reset
```

The `embeddings-reset` target nulls all `chunks.embedding` values (or pass `ARGS="--document-id <uuid>"` to target a subset) and downgrades document statuses back to `CHUNKS_BUILT`. The script then refills embeddings using `EMBEDDING_BATCH_SIZE` / `EMBEDDING_PARALLEL_BATCHES` from `.env` (4 and 2 for CPU-only systems) so the embedding container stays responsive; override via `ARGS="--batch-size ... --parallel ..."` if you have GPU resources.

Set `CRAWLER_YEAR_FILTER` in `.env` (for example, `CRAWLER_YEAR_FILTER=2024` or `CRAWLER_YEAR_FILTER=2023+` to extend through the current year) to limit ingestion to specific years while debugging. The crawler remains idempotent, so you can widen or clear the filter later and rerun the same commands to backfill the rest of the corpus.

`scripts/debug_dump.py` prints current document/page/chunk counts for quick sanity checks.

## Feedback & Fine-tuning (LoRA + DPO)

1. Export preference pairs from stored thumbs-up/down feedback:

   ```bash
   make export-feedback ARGS="--output data/feedback_pairs.jsonl"
   ```

2. Train a lightweight LoRA adapter with TRL's DPOTrainer (default base model: `microsoft/phi-2`):

   ```bash
   make finetune ARGS="--dataset data/feedback_pairs.jsonl --output-dir models/rba-lora-dpo"
   ```

   The job fits on a single GPU or M-series Mac. The resulting adapter lives under `models/rba-lora-dpo` and can be loaded alongside the base model for evaluation.

### What’s in the stack now?

- **Chunking:** recursive, paragraph-aware splitter capped at ~768 tokens with 15 % overlap; section headers are stored as `section_hint` for richer evidence.
- **Retrieval:** pgvector cosine search fused with Postgres `ts_rank_cd` keyword matches for hybrid semantic + lexical recall.
- **LLM UX:** the Streamlit chat streams responses token-by-token from Ollama (default `qwen2.5:1.5b` optimized for CPU with 4K context window, configurable generation limits), so answers start appearing while the long-form completion is still running.
- **Feedback loop:** analysts can rate each assistant reply (thumbs up/down); ratings land in the `feedback` table and have dedicated unit tests (`tests/ui/test_feedback.py`). Feedback events also emit via the hook bus for downstream analytics.
- **Auto-restarting embedding service:** the embedding container now runs with `restart: unless-stopped` and inherits `EMBEDDING_BATCH_SIZE` from `.env` (4 for CPU-only systems, increase for GPU); set `EMBEDDING_DEVICE=cuda|mps|cpu` to force a specific accelerator. See `docs/PARALLEL_PROCESSING.md` for production tuning.

## FAQ

**Why run Postgres inside Docker if it’s “just a database”?**

Keeping Postgres (and pgvector) inside Docker Compose ensures consistent extensions, locales, and init scripts (`docker/postgres/initdb.d/00_extensions.sql`, `01_create_tables.sql`, `02_create_indexes.sql`). You get reproducible migrations on every fresh `make up` without having to manage a separate local instance.

**How many SQL files does Postgres apply?**

Init scripts now include:
- `01_create_tables.sql` (base schema)
- `02_create_indexes.sql` (vector/full-text indexes + triggers)
- `03_seed_eval_examples.sql` (seed golden eval queries)
- `04_add_chunk_table_link.sql` (FK from chunks → tables)
- `05_rebuild_chunk_index.sql` (recreate `idx_chunks_document_id` without bulky text)
- `06_add_charts_table.sql` (chart metadata + chunk FK)

These run automatically on fresh databases; apply `05_*.sql` manually on existing DBs before reprocessing table chunks to avoid oversized btrees.

**What chunk sizes do enterprise teams use?**

Industry playbooks (Pinecone’s 2024 “Chunking Strategies”, Cohere’s 2023 RAG guide, Anthropic’s 2024 retrieval post) converge on 600–1,000 tokens with 10–20 % overlap. That band stays under typical embedding limits (1,024 tokens for `nomic-embed-text`) while keeping enough context for financial narratives.

**Why specifically 768 tokens here?**

`nomic-ai/nomic-embed-text-v1.5` returns 768-dimensional vectors, and the model is calibrated on ~800-token windows. Aligning chunk windows with the pre-training context length avoids truncation on the embedding side and keeps Ollama models (e.g., `qwen2.5`) comfortably within their 4k context budgets after we add instructions + citations.

**How long does a backfill take for 768–900 token chunks?**

Performance varies significantly by hardware:
- **CPU-only** (stable config: batch=4, parallel=2): ~0.28 chunks/sec. A corpus of 2,587 chunks takes ~2.6 hours.
- **GPU** (e.g., batch=32, parallel=8): ~120 chunks/sec. The same corpus embeds in under a minute.

For CPU systems, the conservative batch sizes prevent memory exhaustion and server crashes. See `docs/PARALLEL_PROCESSING.md` for detailed tuning guidance.

## Testing & Linting

```bash
make test
make lint
```

> Tip: running the full test suite requires Docker Desktop (for Postgres/MinIO) and access to the `uv` cache directory. For a quick smoke test of the feedback helpers you can run `make test ARGS="tests/ui/test_feedback.py"`.

## Regenerating Dependencies

After editing `pyproject.toml`, rebuild or re-run `uv sync` inside the container to refresh the virtual environment. Commit the updated `uv.lock` once you have synced successfully.
