# RBA Document Intelligence Platform – Build Plan

## Phase 0 – Environment & Docker Baseline
- Author `pyproject.toml` + `uv.lock` (Python 3.11 target) with core deps: `sqlalchemy`, `pgvector`, `pymupdf`/`pdfplumber`, `boto3`, `requests`, `beautifulsoup4`, `streamlit`, `pytest`, etc.
- Create `.env.example` covering `DATABASE_URL`, `MINIO_*`, `EMBEDDING_MODEL_NAME`, `LLM_*`.
- Write `docker-compose.yml` with four services:
  1. `postgres` (pgvector-enabled) seeded via init script.
  2. `minio` + console, configured with buckets `rba-raw-pdf`, `rba-derived`.
  3. `embedding` – local FastAPI/Text-Embeddings server (CPU torch + sentence-transformers) hosting `nomic-ai/nomic-embed-text-v1.5` on port 8080 with a cached HF volume.
4. `app` image built from `Dockerfile` (python:3.11-slim), running `uv sync` on build and mounting repo for live dev; default command now runs `scripts/wait_for_services.py` before launching Streamlit (database schema comes from `docker/postgres/initdb.d/*.sql`).
- Add helper script `scripts/wait_for_services.py` consumed by the app container before running migrations or Streamlit.

## Phase 1 – Core Skeleton
- Lay out directories exactly as spec (`app/config.py`, `app/db/`, `app/storage/`, `app/pdf/`, `app/embeddings/`, `app/rag/`, `app/ui/`, `scripts/`).
- Implement `app/config.py` reading env vars with defaults + validation.
- Build SQLAlchemy base models (`documents`, `pages`, `chunks`, `chat_sessions`, `chat_messages`) with pgvector column types in `app/db/models.py`; create session factory in `app/db/session.py`.
- Add MinIO storage adapter (`app/storage/base.py`, `minio_s3.py`) handling streaming upload/download/bucket ensure.
- Ship schema migrations under `docker/postgres/initdb.d/*.sql` so Postgres initializes itself on first boot.

## Phase 2 – Ingestion & Processing Pipelines
- `scripts/crawler_rba.py`: crawl official SMP/FSR listings via `requests` + `BeautifulSoup`, dedupe via Postgres, store PDFs in MinIO (`raw/<doc_type>/filename.pdf`), mark `documents.status=NEW`. Allow optional `CRAWLER_YEAR_FILTER` env var so engineers can limit work to specific years during debugging.
- `app/pdf/parser.py`: stream PDF text extraction (choose pymupdf/pdfplumber).
- `app/pdf/cleaner.py`: normalize whitespace, strip headers/footers.
- `app/pdf/chunker.py`: token-aware chunking with page bounds + optional section hints.
- `scripts/process_pdfs.py`: iterate docs by status, extract pages, run cleaner/chunker, populate `pages`/`chunks`, update statuses (`TEXT_EXTRACTED`, `CHUNKS_BUILT`).

## Phase 3 – Embeddings & RAG Layer
- `app/embeddings/client.py`: pluggable embedding interface (env-driven base URL/model/API key + timeout) hitting the embedding service.
- `app/embeddings/indexer.py`: find chunks missing embeddings, batch call client, persist vectors.
- `scripts/build_embeddings.py`: CLI entry to run indexer; integrate into docker workflow via the Makefile target (`make embeddings`).
- `app/rag/retriever.py`: similarity search via SQL query (pgvector `cosine_distance`) with filters (doc_type/date).
- Persist `chunks.text_tsv` with a trigger so lexical search doesn't rebuild vectors on the fly; hybrid retrieval should weight semantic vs lexical scores (~0.7/0.3) per Pinecone/Cohere guidance.
- `app/rag/llm_client.py`: generic chat/complete wrapper.
- `app/rag/pipeline.py`: `answer_query(query, session_id=None)` retrieving context, building prompt, calling LLM, persisting chat messages, returning `{answer, evidence[], analysis}`.

## Phase 4 – UI & Operational Tools
- `app/ui/streamlit_app.py`: chat interface, session list, evidence accordion, env-driven connectors; run inside docker app service.
- `scripts/debug_dump.py`: inspect DB rows, check document counts, verify storage connectivity.
- Logging/metrics: configure structured logging (JSON or simple text) to stdout for docker aggregation; optionally add lightweight health endpoint invoked via Streamlit sidebar diagnostics.

## Phase 5 – Testing, CI, and Docs
- Testing: add `tests/` mirroring modules with pytest fixtures mocking MinIO/Postgres (use Moto/localstack alternatives or temporary sqlite+fake storage); include integration test invoking crawler→processor→retriever on sample PDFs stored under `tests/fixtures`.
- CI: GitHub Actions workflow running `make up-detached`, `make test ARGS="-q"`, plus lint (optional `make lint`, `mypy`).
- Docs: keep `claude.md` authoritative spec, maintain `AGENTS.md` for quick contributor onboarding, and expand `README.md` with Makefile targets (`make crawl`, `make embeddings`, etc.), troubleshooting, and env var descriptions.

## Phase 6 – RAG Quality & Retrieval Enhancements

Industry best practices for production RAG systems, following guidance from Pinecone, Cohere, Anthropic, and academic research. All improvements are **optional** via config flags to maintain backward compatibility.

### 6.1 Chunking Improvements

**Table-Aware Boundaries** (`app/pdf/chunker.py`):
- Use existing `_contains_table_marker()` to detect tables near chunk boundaries
- Extend or contract boundaries by ±200 chars to avoid mid-table splits
- Prevents corrupted structured data in retrieval
- **Effort**: 2-3 hours

**Chunk Quality Scoring** (`app/pdf/chunker.py`):
- Score chunks by:
  - Sentence boundary ratio (complete sentences preferred)
  - Length variance (not too short/long)
  - Simple heuristics, no ML
- Filter chunks with quality score < 0.5 before embedding
- Reduces index noise and improves retrieval precision
- Config: `CHUNK_QUALITY_THRESHOLD=0.5`
- **Effort**: 2-3 hours

**Total 6.1 Effort**: 4-6 hours

### 6.2 Retrieval Diversity & Classification

**MMR (Maximal Marginal Relevance)** (`app/rag/retriever.py`):
- Implement `mmr_rerank(query_embedding, chunks, lambda_param=0.5)`
- Iteratively select chunks that are relevant AND diverse
- Penalize similarity to already-selected chunks
- Prevents redundant chunks from same document/section
- Config: `USE_MMR=1` (default: enabled), `MMR_LAMBDA=0.5`
- **Effort**: 3-4 hours

**Query Classification** (`app/rag/retriever.py`):
- Rule-based classifier: keyword / semantic / numerical
- Adjust hybrid weights dynamically:
  - Keyword queries (e.g., "RBA meeting 2024-05-07"): 80% lexical, 20% semantic
  - Semantic queries (e.g., "inflation trends"): 80% semantic, 20% lexical
  - Numerical queries (e.g., "GDP forecast 2025"): 70/30 + 50% table boost
- Simple pattern matching, no ML
- **Effort**: 3-4 hours

**Total 6.2 Effort**: 6-8 hours

### 6.3 Context & Ranking Enhancements

**Context Window Validation** (`app/rag/pipeline.py`):
- Add `_validate_context_budget(chunks, max_tokens=6000)`
- Count tokens using `tiktoken` library (GPT tokenizer as proxy)
- Truncate chunks from lowest-scoring to highest if exceeds limit
- Preserve at least top-3 chunks
- Log warnings when truncation occurs
- Config: `MAX_CONTEXT_TOKENS=6000`
- **Dependency**: Add `tiktoken = "^0.5.1"` to `pyproject.toml`
- **Effort**: 1-2 hours

**Reciprocal Rank Fusion (RRF)** (`app/rag/retriever.py`):
- Alternative to weighted score combination
- Formula: `RRF_score(chunk) = Σ 1/(k + rank_i(chunk))` across rankers
- More stable than weighted combination when scores have different scales
- Config: `USE_RRF=0` (default: disabled, keep weighted approach)
- **Effort**: 2-3 hours

**Total 6.3 Effort**: 3-5 hours

### 6.4 Multi-Page Table Merging

**Table Continuation Detection** (`scripts/extract_tables.py`):
- Pattern match "Table X.Y (continued)" in captions
- Merge consecutive tables with same base caption
- Combine `structured_data` JSONB arrays
- Update `page_number` ranges (store as `page_start` / `page_end`)
- Schema: Add `page_start`, `page_end` columns to `tables` table (migration required)
- **Effort**: 4-6 hours

**Total 6.4 Effort**: 4-6 hours

---

### Configuration Additions (`.env.example`)

```env
# ========================================
# Retrieval & Ranking Enhancements
# ========================================

# MMR Diversity
USE_MMR=1                          # Enable Maximal Marginal Relevance
MMR_LAMBDA=0.5                     # 0=max diversity, 1=max relevance

# Ranking Strategy
USE_RRF=0                          # Use Reciprocal Rank Fusion (alternative to weighted)
USE_RERANKING=0                    # Enable cross-encoder reranking (+25-40% accuracy, ~300ms)

# Hybrid Search Weights (used when USE_RRF=0)
SEMANTIC_WEIGHT=0.7                # Semantic search weight
LEXICAL_WEIGHT=0.3                 # Lexical (full-text) search weight
RECENCY_WEIGHT=0.25                # Recency bias weight
TABLE_BOOST_DATA_QUERIES=0.5      # Boost for table chunks when data query detected

# Reranking
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_MULTIPLIER=10               # Retrieve 10x candidates for reranking
RERANK_BATCH_SIZE=32

# Context Management
MAX_CONTEXT_TOKENS=6000            # Token budget for LLM context
CHUNK_QUALITY_THRESHOLD=0.5        # Min quality score for chunks (0.0-1.0)
```

### Dependency Updates (`pyproject.toml`)

```toml
[project.dependencies]
# ... existing dependencies ...
tiktoken = "^0.5.1"  # Token counting for context window validation
```

---

### Phase 6 Implementation Notes

**No over-engineering**:
- Simple rule-based approaches (no ML beyond existing cross-encoder)
- Optional flags for backward compatibility
- No new services, frameworks, or databases
- Graceful degradation if features disabled

**Testing priority**:
- Add unit tests for MMR, RRF, query classification
- Integration test: compare retrieval quality before/after improvements
- Validate merged tables render correctly in UI

**Rollout strategy**:
1. Implement all improvements with features disabled by default
2. Enable MMR + context validation (low-risk)
3. A/B test query classification and RRF
4. Enable reranking for accuracy-critical queries

**Total Phase 6 Effort**: 18-24 hours focused work

## Operational Considerations
- Schedule ingestion scripts via cron/Kubernetes Jobs invoking the relevant `make` targets (for example, `make refresh`).
- Back up Postgres and MinIO volumes; document retention policies for raw PDFs vs derived artifacts.
- Define alerting for crawler failures or embedding backlog (simple CLI exit codes consumed by external scheduler).

---

## Implementation Status

### Phases 0-5: ✅ Completed

All phases (0-5) have been completed and implemented via **Makefile targets**. The Makefile provides a centralized command interface for all operations:

- **Bootstrap**: `make bootstrap` (build + sync deps)
- **Services**: `make up`, `make up-models`, `make llm-pull`
- **Ingestion**: `make crawl`, `make process`, `make tables`, `make embeddings`
- **Development**: `make test`, `make lint`, `make format`
- **ML engineering**: `make export-feedback`, `make finetune`
- **UI**: `make streamlit`

For detailed line-by-line code explanations of the complete implementation, see **`LEARN.md`**.

### Phase 6: 📋 Planned - RAG Quality Improvements

**Status**: Ready for implementation (18-24 hours focused work)

**Alignment with current system**:
- All improvements build on existing architecture (no new services/frameworks)
- Backward compatible via config flags
- Follows industry best practices (Pinecone, Cohere, Anthropic)
- No over-engineering: simple, production-ready approaches

**Implementation priorities** (HIGH → MEDIUM):

1. **HIGH** (Correctness & Data Quality):
   - ✅ Table-aware chunk boundaries (prevents data corruption)
   - ✅ Multi-page table merging (complete table data)
   - ✅ Context window validation (prevents LLM errors)
   - ✅ MMR diversity (better coverage)

2. **MEDIUM** (Performance & Accuracy):
   - ✅ Query classification (adaptive weights, +10-15% relevance)
   - ✅ RRF alternative (more stable rankings)
   - ✅ Chunk quality scoring (cleaner index)

**Current RAG capabilities** (already implemented):
- ✅ Hybrid search (semantic + lexical)
- ✅ Optional cross-encoder reranking
- ✅ Recency bias
- ✅ Data query detection → table boosting
- ✅ Markdown table formatting
- ✅ Structured data + flattened text dual storage

**Next steps**:
1. Add `tiktoken` dependency to `pyproject.toml`
2. Update `.env.example` with Phase 6 config flags
3. Implement improvements in order: 6.1 → 6.4 → 6.3 → 6.2
4. Add unit tests for new functions (MMR, RRF, query classification)
5. Run `make embeddings` after chunking changes
6. Update `LEARN.md` with implementation details

### Maintenance Notes
- Added `docker/postgres/initdb.d/05_rebuild_chunk_index.sql` to drop/recreate `idx_chunks_document_id` without the bulky `text` column. Apply it on existing databases before re-running `make tables` so large table chunks no longer exceed the btree page limit.
- Phase 6 schema change: Add `page_start`, `page_end` columns to `tables` table for multi-page merging (requires migration).
