# RBA Document Intelligence Platform

Local-first setup for crawling, processing, and querying Reserve Bank of Australia PDF publications via a Retrieval-Augmented Generation (RAG) workflow.

## Prerequisites

- Docker & Docker Compose v2
- `uv` (optional for running scripts on the host)

## Quick Start

1. Copy `.env.example` to `.env` and adjust credentials if needed.
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
   # Pull the lightweight multilingual LLM once
   make llm-pull MODEL=qwen2.5:1.5b
   ```

## Running Pipelines

Use the same container for operational scripts so dependencies stay consistent:

```bash
make crawl
make process
make embeddings ARGS="--batch-size 24 --parallel 2"
# Or run them all sequentially:
make refresh
```

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

The `embeddings-reset` target nulls all `chunks.embedding` values (or pass `ARGS="--document-id <uuid>"` to target a subset) and downgrades document statuses back to `CHUNKS_BUILT`. The script then refills embeddings with smaller default batches (24 chunks, 2 workers) so the CPU embedding container stays responsive; override via `ARGS` if you have more headroom.

Set `CRAWLER_YEAR_FILTER` in `.env` (for example, `CRAWLER_YEAR_FILTER=2024`) to limit ingestion to specific years while debugging. The crawler remains idempotent, so you can widen or clear the filter later and rerun the same commands to backfill the rest of the corpus.

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
- **LLM UX:** the Streamlit chat streams responses token-by-token from Ollama (default `qwen2.5:1.5b`), so answers start appearing while the long-form completion is still running.
- **Feedback loop:** analysts can rate each assistant reply (thumbs up/down); ratings land in the `feedback` table and have dedicated unit tests (`tests/ui/test_feedback.py`). Feedback events also emit via the hook bus for downstream analytics.
- **Auto-restarting embedding service:** the embedding container now runs with `restart: unless-stopped` and a conservative `EMBEDDING_BATCH_SIZE=16`, so long-running backfills survive transient OOMs on CPU hosts.

## FAQ

**Why run Postgres inside Docker if it’s “just a database”?**

Keeping Postgres (and pgvector) inside Docker Compose ensures consistent extensions, locales, and init scripts (`docker/postgres/initdb.d/00_extensions.sql`, `01_create_tables.sql`, `02_create_indexes.sql`). You get reproducible migrations on every fresh `make up` without having to manage a separate local instance.

**How many SQL files does Postgres apply?**

There are exactly two logical migrations now: `01_create_tables.sql` (base schema) and `02_create_indexes.sql` (vector/full-text indexes plus triggers). The numbering leaves room for future migrations (`03_*`, etc.) without renaming past files.

**What chunk sizes do enterprise teams use?**

Industry playbooks (Pinecone’s 2024 “Chunking Strategies”, Cohere’s 2023 RAG guide, Anthropic’s 2024 retrieval post) converge on 600–1,000 tokens with 10–20 % overlap. That band stays under typical embedding limits (1,024 tokens for `nomic-embed-text`) while keeping enough context for financial narratives.

**Why specifically 768 tokens here?**

`nomic-ai/nomic-embed-text-v1.5` returns 768-dimensional vectors, and the model is calibrated on ~800-token windows. Aligning chunk windows with the pre-training context length avoids truncation on the embedding side and keeps Ollama models (e.g., `qwen2.5`) comfortably within their 4k context budgets after we add instructions + citations.

**How long does a backfill take for 768–900 token chunks?**

On an M-series Mac the FastAPI embedding service processes ~120 chunks/second at batch size 32. The current corpus (≈500 chunks) re-embeds in under a minute; doubling chunk length to 900 tokens typically adds ~15 % latency because payloads get larger, but parallel batches keep total wall-clock under two minutes.

## Testing & Linting

```bash
make test
make lint
```

> Tip: running the full test suite requires Docker Desktop (for Postgres/MinIO) and access to the `uv` cache directory. For a quick smoke test of the feedback helpers you can run `make test ARGS="tests/ui/test_feedback.py"`.

## Regenerating Dependencies

After editing `pyproject.toml`, rebuild or re-run `uv sync` inside the container to refresh the virtual environment. Commit the updated `uv.lock` once you have synced successfully.
