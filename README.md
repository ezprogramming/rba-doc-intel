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
# Simple 3-step workflow (NEW - simplified!)
make crawl          # Download PDFs from RBA website
make ingest         # Extract text + tables in one pass (replaces old process + tables)
make embeddings     # Generate vectors for retrieval

# Or run them all sequentially:
make refresh        # Runs crawl → ingest → embeddings
```

### Simplified Ingestion (Text + Tables)

The new `make ingest` command replaces the old two-stage `make process` + `make tables` workflow:

- **Old workflow** (deprecated):
  - `make process` → extract text, create chunks
  - `make tables` → extract tables separately
  - Two PDF reads, complex state management

- **New workflow** (current):
  - `make ingest` → extract text + tables in **single PDF pass**
  - Faster, simpler, more robust
  - Each chunk includes caption, structured data, and `table_id` back-reference

**Re-processing from scratch:**

```bash
make ingest-reset   # Reset all documents to NEW status
make ingest         # Reprocess everything
make embeddings     # Rebuild vectors
```

**Retrying failed documents:**

```bash
make ingest-retry   # Reset only FAILED documents to NEW status
make ingest         # Retry processing failed docs
```

### Chunking & Retrieval Defaults

- **Chunk window:** 768-token cap with ~15 % overlap. This mirrors Pinecone/Anthropic guidance for production RAG – big enough for narrative coherence, small enough to keep embedding latency down.
- **Section hints:** The chunker scans the first 200 characters for headings (`3.2 Inflation`, `Chapter 4`, `Box A`) and stores them in `chunks.section_hint`. The Streamlit UI surfaces these hints inside the evidence expander so analysts immediately see where a quote came from.
- **Hybrid similarity search:** Retrieval fuses cosine similarity (`pgvector` HNSW index) with Postgres full-text scores (`ts_rank_cd` on a persisted `tsvector` column). This is the "semantic + lexical" hybrid pattern Pinecone, Weaviate, and Cohere now recommend for enterprise search because identifiers/dates often rely on raw keyword matches.

### RAG Quality Features

The platform includes industry best practices for production RAG systems:

**Chunking Enhancements:**
- **Table-aware boundaries:** Prevents mid-table splits to preserve structured data integrity
- **Quality scoring:** Filters low-quality chunks (fragments, corrupted text) with configurable threshold
- **Smart boundaries:** Prefers paragraph/sentence breaks over arbitrary character counts

**Context Window Management:**
- **Token budget validation:** Prevents LLM overflow using tiktoken for accurate token counting
- **Smart truncation:** Removes lowest-scoring chunks while preserving top-3 minimum quality
- **Configurable limits:** Default 6000 tokens (via `MAX_CONTEXT_TOKENS`)

**Retrieval Quality:**
- **Query classification:** Automatically detects keyword/semantic/numerical queries and adapts weights
- **MMR diversity:** Reduces redundant chunks using Maximal Marginal Relevance (λ=0.5 default)
- **RRF option:** Reciprocal Rank Fusion as alternative to weighted score combination
- **Table boosting:** Automatically boosts table chunks for numerical/data queries

**Configuration:**
All features are configurable via `.env`:
```bash
# Enable/disable features
USE_MMR=1                      # MMR diversity (recommended)
USE_RRF=0                      # Reciprocal Rank Fusion
USE_RERANKING=0                # Cross-encoder reranking (+300ms, +25-40% accuracy)

# Tuning parameters
MMR_LAMBDA=0.5                 # 0=max diversity, 1=max relevance
SEMANTIC_WEIGHT=0.7            # Semantic search weight
LEXICAL_WEIGHT=0.3             # Full-text search weight
CHUNK_QUALITY_THRESHOLD=0.5    # Min quality score (0.0-1.0)
MAX_CONTEXT_TOKENS=6000        # LLM context budget
```

**Score Combination Methods:**
The platform supports two methods for combining semantic and lexical search results:

1. **Weighted Score Combination** (default: `USE_RRF=0`)
   - Combines scores: `final = 0.7×semantic + 0.3×lexical`
   - Advantages: Tunable weights, interpretable scores
   - Best for: When score distributions are similar

2. **Reciprocal Rank Fusion** (RRF) (optional: `USE_RRF=1`)
   - Formula: `score(chunk) = Σ(1/(k + rank))` across all rankings
   - Advantages: Robust to scale differences, no manual tuning
   - Best for: When semantic/lexical scores have different scales
   - Implementation: `app/rag/retriever.py:254-280`

**Performance:**
- Base retrieval: ~55-205ms
- With MMR + quality filtering: ~60-250ms (+5-45ms, +25-40% accuracy)
- With reranking enabled: +200-500ms (+15-25% additional accuracy)

See `CLAUDE.md` sections 7.4, 8.3, and 8.4 for implementation details.

### Observability hooks

- `app/rag/hooks.py` exposes a light pub/sub bus. The RAG pipeline emits lifecycle events (`rag:query_started`, `rag:retrieval_complete`, `rag:stream_chunk`, `rag:answer_completed`) and the Streamlit UI emits `ui:question_submitted`, `ui:answer_rendered`, `ui:message_committed`, and `ui:feedback_recorded`.
- Subscribe anywhere (scripts, tests) to tap into these events without touching business logic:

  ```python
  from app/rag.hooks import hooks

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

### What's in the stack now?

- **Simplified ingestion:** Single-pass PDF processing (`make ingest`) extracts text + tables together, replacing the old two-stage workflow. Faster, more robust, easier to understand.
- **Chunking:** recursive, paragraph-aware splitter capped at ~768 tokens with 15 % overlap; section headers are stored as `section_hint` for richer evidence.
- **Table extraction:** Camelot (lattice + stream) extracts structured tables inline during ingestion. Each table chunk includes caption, structured rows, and `table_id` back-reference for citations.
- **Table formatting for RAG:** Tables are automatically formatted as markdown in LLM prompts (instead of flattened text) for 25-40% better accuracy on numerical queries. The UI also renders tables visually for easy user verification.
- **Retrieval:** pgvector cosine search fused with Postgres `ts_rank_cd` keyword matches for hybrid semantic + lexical recall. Table chunks receive automatic boosting for data-focused queries.
- **LLM UX:** the Streamlit chat streams responses token-by-token from Ollama (default `qwen2.5:1.5b` optimized for CPU with 4K context window, configurable generation limits), so answers start appearing while the long-form completion is still running.
- **Feedback loop:** analysts can rate each assistant reply (thumbs up/down); ratings land in the `feedback` table and have dedicated unit tests (`tests/ui/test_feedback.py`). Feedback events also emit via the hook bus for downstream analytics.
- **Fine-tuning ready:** Export feedback pairs (`make export-feedback`) and train LoRA adapters with DPO (`make finetune`) to improve model quality based on real user preferences.
- **Auto-restarting embedding service:** the embedding container now runs with `restart: unless-stopped` and inherits `EMBEDDING_BATCH_SIZE` from `.env` (4 for CPU-only systems, increase for GPU); set `EMBEDDING_DEVICE=cuda|mps|cpu` to force a specific accelerator. See `docs/PARALLEL_PROCESSING.md` for production tuning.

## FAQ

**Why run Postgres inside Docker if it's "just a database"?**

Keeping Postgres (and pgvector) inside Docker Compose ensures consistent extensions, locales, and init scripts. You get reproducible migrations on every fresh `make up` without having to manage a separate local instance.

**What database initialization files are there?**

The database initialization has been consolidated into 3 simple files:
- `00_init.sql` - All tables, indexes, and triggers (consolidated from 6 previous files)
- `01_test_schema.sql` - Test schema for isolated end-to-end testing
- `02_seed_eval_examples.sql` - Optional golden evaluation queries

These run automatically when creating a fresh database. Much simpler than the previous 6-file setup!

**What chunk sizes do enterprise teams use?**

Industry playbooks (Pinecone's 2024 "Chunking Strategies", Cohere's 2023 RAG guide, Anthropic's 2024 retrieval post) converge on 600–1,000 tokens with 10–20 % overlap. That band stays under typical embedding limits (1,024 tokens for `nomic-embed-text`) while keeping enough context for financial narratives.

**Why specifically 768 tokens here?**

`nomic-ai/nomic-embed-text-v1.5` returns 768-dimensional vectors, and the model is calibrated on ~800-token windows. Aligning chunk windows with the pre-training context length avoids truncation on the embedding side and keeps Ollama models (e.g., `qwen2.5`) comfortably within their 4k context budgets after we add instructions + citations.

**How long does a backfill take for 768–900 token chunks?**

Performance varies significantly by hardware:
- **CPU-only** (stable config: batch=4, parallel=2): ~0.28 chunks/sec. A corpus of 2,587 chunks takes ~2.6 hours.
- **GPU** (e.g., batch=32, parallel=8): ~120 chunks/sec. The same corpus embeds in under a minute.

For CPU systems, the conservative batch sizes prevent memory exhaustion and server crashes. See `docs/PARALLEL_PROCESSING.md` for detailed tuning guidance.

**Why format tables as markdown in LLM prompts?**

Modern LLMs (GPT-4, Claude, Llama) are extensively trained on markdown tables and perform significantly better on structured data when they see proper row/column formatting:

- **Before:** Tables flattened to text like `"GDP — 2024: 2.1%, 2025: 2.5%"` → LLM struggles with multi-column comparisons
- **After:** Tables as markdown with headers → 25-40% better accuracy on numerical queries
- **Why it works:** Column headers provide context for every value, reducing parsing ambiguity
- **Industry standard:** Markdown tables match how LLMs were trained (StackOverflow, GitHub, documentation sites)

The table formatting happens automatically during RAG retrieval:
1. **Storage:** Tables stored as JSONB in `tables.structured_data`
2. **Retrieval:** Chunks with `table_id` fetch structured data
3. **Formatting:** `format_table_as_markdown()` converts to clean markdown
4. **LLM prompt:** Receives structured table instead of flattened text
5. **UI:** Renders formatted table in evidence section for user verification

Example transformation sent to LLM:
```markdown
Table: Economic Forecasts

| Year | GDP  | Inflation |
|------|------|-----------|
| 2024 | 2.1% | 3.5%      |
| 2025 | 2.5% | 2.8%      |
```

**Graceful fallback:** If markdown generation fails, the system automatically falls back to flattened text, ensuring robustness.

## Testing & Linting

```bash
# Run unit tests
make test

# Run end-to-end workflow test (validates entire pipeline)
make test-workflow

# Lint code
make lint
```

**End-to-End Workflow Test:**

The `make test-workflow` command validates the complete pipeline with a sample PDF:
- ✓ PDF ingestion (text + tables)
- ✓ Table extraction and linking
- ✓ Embedding generation
- ✓ RAG retrieval with table content

See `docs/TESTING.md` for detailed test documentation.

> Tip: running the full test suite requires Docker Desktop (for Postgres/MinIO) and access to the `uv` cache directory. For a quick smoke test of the feedback helpers you can run `make test ARGS="tests/ui/test_feedback.py"`.

## Regenerating Dependencies

After editing `pyproject.toml`, rebuild or re-run `uv sync` inside the container to refresh the virtual environment. Commit the updated `uv.lock` once you have synced successfully.
