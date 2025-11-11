# Codebase Exploration Summary

## What Was Explored

Complete systematic exploration of the **RBA Document Intelligence Platform** project, including:

- **Directory structure** (18 directories, 55 files in project root)
- **Python modules** (33 application + script files)
- **Configuration files** (pyproject.toml, docker-compose.yml, .env.example)
- **Database schema** (10 tables in 3 PostgreSQL init scripts)
- **Docker services** (5 services: postgres, minio, embedding, llm, app)
- **Scripts** (8 operational CLI tools)
- **Tests** (5 test modules)
- **Documentation** (existing: CLAUDE.md, LEARN.md, README.md, PLAN.md, AGENTS.md)

## Documentation Generated

Two comprehensive guides have been created and saved to the project:

### 1. **CODEBASE_STRUCTURE.md** (901 lines)
Detailed reference covering:
- Project overview and tech stack
- Complete directory structure with descriptions
- Core components deep-dive:
  - Configuration system
  - Database layer (10 tables, status flows, relationships)
  - Storage layer (MinIO/S3 abstraction)
  - PDF processing pipeline (extract → clean → chunk)
  - Embeddings generation and indexing
  - RAG pipeline (hybrid retrieval, reranking, hooks)
  - Streamlit UI with feedback loop
- Script documentation (8 operational scripts)
- PostgreSQL schema (extensions, tables, indexes, triggers)
- Docker Compose services (5 containerized components)
- Dependencies reference (25+ core packages)
- Testing structure and commands
- Development workflow (7-step process)
- Key design decisions (5 architectural choices)
- Monitoring and observability patterns
- Future extensions (out of scope)

### 2. **QUICK_REFERENCE.md** (342 lines)
Quick lookup guide for developers:
- Project at a glance
- File & directory quick map (18 items)
- Database schema diagram (10 tables)
- Operations checklist (setup, start, ingest, debug, test, fine-tune)
- Pipeline flows (ingest phase diagram, query phase diagram)
- Key classes & functions (6 module groups)
- Configuration reference (17 environment variables)
- Performance notes (4 categories)
- Common workflows (5 practical examples)
- Troubleshooting table (7 common issues)
- Architecture decisions (5 key choices)
- File sizes & complexity (8 major files)
- External resources reference

## Key Findings

### Architecture Highlights
1. **Layered Design** - Clean separation: config → storage → PDF → embeddings → RAG → UI
2. **Parallel Processing** - 4x speedup on PDF processing and embedding generation
3. **Hybrid Retrieval** - Combines semantic (pgvector HNSW) + lexical (Postgres FTS) search
4. **Event Bus** - Lightweight hooks for decoupled instrumentation
5. **Production-Grade** - Proper error handling, streaming, batch processing, indexing

### Technology Choices
- **pgvector + Postgres** - Single database for vectors + metadata + chat (no extra services)
- **768-token chunks** - Aligns with embedding model pre-training window
- **Ollama local LLM** - No cloud dependencies, configurable models
- **Streamlit UI** - Minimal but functional chat interface
- **uv package manager** - Fast, reproducible Python environment

### Data Flow
```
RBA Websites
    ↓ (crawler_rba.py)
MinIO (raw PDFs)
    ↓ (process_pdfs.py)
PostgreSQL (pages + chunks)
    ↓ (build_embeddings.py)
pgvector HNSW + tsvector FTS
    ↓ (pipeline.py)
Ollama LLM ← User Question
    ↓
Streamlit UI ← Feedback ← chat_messages table
```

### Database Maturity
- **10 tables** covering documents, chunks, chat, feedback, evaluation
- **8 production-grade indexes** (HNSW vector, GIN full-text, composite)
- **Automatic triggers** for tsvector maintenance
- **Status flow tracking** for pipeline state (NEW → EMBEDDED)
- **Evaluation framework** for RAG quality metrics

### Operational Maturity
- **8 CLI scripts** covering the full lifecycle (crawl → process → embed → fine-tune)
- **Parallel workers** for performance (PDF processing 4x, embedding 12x on M4)
- **Reset/retry logic** for idempotent operations
- **Configuration flexibility** (year filters, model choices, batch sizes)
- **Health checks** and startup probes

## Stats

| Category | Count |
|----------|-------|
| Python source files | 33 |
| Database tables | 10 |
| Docker services | 5 |
| Scripts/CLI tools | 8 |
| Test modules | 5 |
| Dependencies | 25+ |
| Configuration variables | 17 |
| PostgreSQL indexes | 8 |
| API endpoints | Multiple (crawler, embedding, LLM, UI) |
| **Total lines of documentation generated** | **1,243** |

## Usage of Generated Documentation

### For New Team Members
- Start with **QUICK_REFERENCE.md** for overview
- Deep-dive with **CODEBASE_STRUCTURE.md** for details
- Reference existing docs (CLAUDE.md for spec, LEARN.md for decisions)

### For Feature Development
- Check **Quick Reference → Common Workflows** for patterns
- Review relevant module in **Codebase Structure** for implementation
- Look at tests for examples and edge cases

### For Troubleshooting
- Use **Quick Reference → Troubleshooting** table first
- Check **Codebase Structure → Key Design Decisions** for architectural context
- Review existing tests for expected behavior

### For Operations
- Follow **Quick Reference → Operations Checklist**
- Use **Codebase Structure → Scripts** for detailed script documentation
- Check performance notes for optimization guidance

## Key Insights

### What Makes This Project Special
1. **Production-Ready Architecture** - Not a prototype; implements real RAG system concerns
2. **Hybrid Search** - Combines multiple retrieval methods (rare in hobby projects)
3. **Feedback Loop** - Built-in thumbs up/down → fine-tuning pipeline
4. **Local-First** - No cloud dependencies; runs on M-series Macs
5. **Instrumentation** - Event bus for monitoring without coupling

### Well-Designed Patterns
- **Config as Code** - Environment-driven, lazy-loaded, immutable
- **Storage Adapter** - Abstract interface, pluggable backends
- **Context Managers** - Database sessions with auto-cleanup
- **Batch Processing** - Idempotent operations with retry logic
- **Hook Bus** - Decoupled event subscription without modifying core

### Documentation Completeness
- **CLAUDE.md** - Exhaustive spec with constraints (19,757 bytes)
- **LEARN.md** - Technical decisions (13,502 bytes)
- **README.md** - Quick start and features (8,025 bytes)
- **CODEBASE_STRUCTURE.md** - Comprehensive reference (GENERATED)
- **QUICK_REFERENCE.md** - Developer cheat sheet (GENERATED)

## Recommendations

### For Documentation
1. Pin **CODEBASE_STRUCTURE.md** as primary reference
2. Use **QUICK_REFERENCE.md** for onboarding
3. Keep **CLAUDE.md** as source of truth for constraints
4. Update docs when architecture changes

### For Development
1. Follow layered design pattern when adding features
2. Use hooks for observability (don't add logging to core)
3. Maintain test coverage for PDF, RAG, and UI modules
4. Profile embedding & retrieval performance regularly

### For Operations
1. Monitor PostgreSQL index health (`ANALYZE chunks` regularly)
2. Set `LOG_LEVEL=DEBUG` for hook tracing
3. Track embedding throughput with `debug_dump.py`
4. Export feedback monthly for fine-tuning opportunities

## File Locations

All documentation files have been saved to the project root:
- `/Users/leizheng/Library/Mobile Documents/com~apple~CloudDocs/Devs/rba-doc-intel/CODEBASE_STRUCTURE.md`
- `/Users/leizheng/Library/Mobile Documents/com~apple~CloudDocs/Devs/rba-doc-intel/QUICK_REFERENCE.md`

These are now part of the project and should be committed to version control.

---

## Next Steps

1. **Review** - Project maintainers should review generated docs for accuracy
2. **Maintain** - Update when architecture changes occur
3. **Distribute** - Share with team members for onboarding
4. **Link** - Reference in README and contributing guidelines
5. **Evolve** - Add sections as new features are implemented

---

Generated: November 11, 2025
Explorer: Claude Code (Anthropic)
Exploration Time: ~30 minutes
Total Codebase Files: 55 (core project files)
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
- `scripts/` hosts operational entry points (`crawler_rba.py`, `process_pdfs.py`, `build_embeddings.py`, `debug_dump.py`, `export_feedback_pairs.py`, `finetune_lora_dpo.py`).
Never rename top-level directories or move modules without explicit approval.

## Environment, Build & Run Commands
Use the Makefile wrappers (see `make help`) so everyone shares the same Compose incantations. Pass extra script flags via `ARGS="..."`.

- `make bootstrap` — prepare the containerized dev env.
- `make up` — launch Postgres, MinIO, the embedding service, and the Streamlit app; Postgres applies `/docker/postgres/initdb.d/*.sql` on first boot to create the schema and indexes automatically.
- `make up-embedding` — start the local embedding API (FastAPI + sentence-transformers). It must be healthy before `scripts/build_embeddings.py` succeeds.
- `make crawl` — crawl RBA sites, push PDFs to MinIO, insert metadata.
- `make process` — pull pending docs, produce cleaned chunks.
- `make embeddings` — fill missing pgvector embeddings (respects `EMBEDDING_BATCH_SIZE`/`EMBEDDING_API_TIMEOUT`). Append `ARGS="--reset"` to null out existing vectors (after chunk strategy changes) or `ARGS="--document-id <uuid>"` to target specific documents.
- `make export-feedback ARGS="--output data/feedback_pairs.jsonl"` — convert stored thumbs-up/down into preference pairs.
- `make finetune ARGS="--dataset data/feedback_pairs.jsonl"` — train/update the LoRA adapter via DPO.
- `make streamlit` — start the UI manually if needed.
- `make test` — execute the core test suite (lint optional via `make lint`).
Configure endpoints via `.env` (mirrors `.env.example`). Never hard-code credentials or paths; everything flows through `app/config.py`.
During local debugging you can set `CRAWLER_YEAR_FILTER` to limit ingestion to recent years, and tune `EMBEDDING_BATCH_SIZE`/`EMBEDDING_API_TIMEOUT` to trade off speed vs stability.

## Coding Style & Data Contracts
Use four-space indentation, PEP 8 naming, and thorough type hints. Keep SQLAlchemy models in `app/db/models.py` (documents, pages, chunks, chat tables) matching the schema in `CLAUDE.md`. Expose pure functions/classes in library modules and reserve side effects for scripts. Structure RAG responses as `{answer, evidence[], analysis}` and persist chat interactions per the spec. Documents must record `source_url`, `content_hash`, and `content_length`; chunk embeddings are fixed at 768 dims to match `nomic-ai/nomic-embed-text-v1.5`, and every chunk should populate `section_hint` + `text_tsv` so the UI and hybrid retriever stay in sync.

## Testing & Data Quality
Place tests under `tests/` mirroring module paths (`tests/rag/test_pipeline.py`, etc.). Prefer `pytest` fixtures to mock MinIO/Postgres; add lightweight integration tests that process sample PDFs end-to-end. Every change should keep `make test ARGS="-q"` green and cover regression points (crawler edge cases, chunking boundaries, retrieval filters, embedding client timeouts). New helpers (e.g. cleaners, analysis formatters) should ship with unit coverage before touching ingestion/RAG code.

## Contribution Workflow & Security
Commit messages follow `<area>: <imperative summary>` and PRs must describe the change, list new env vars/commands, reference issues, and attach evidence (logs, UI screenshots). Flag schema or data migrations prominently. Secrets live in `.env` (gitignored) and flow through `app/config.py`. Validate storage/database connectivity before merging, and avoid uploading proprietary PDFs or sensitive excerpts to public channels.
# RBA Document Intelligence Platform – Technical Spec
_File: `claude.md`_

## 0. Purpose of this document

This document defines **hard constraints** and **implementation guidelines** for an AI-assisted Python project that:

- Crawls and ingests **RBA PDF reports** into a local simulated cloud environment (MinIO + PostgreSQL).
- Processes PDFs in a **production-style pipeline** (batch + streaming friendly).
- Builds a **RAG system** over cleaned text & metadata.
- Exposes a **simple Streamlit UI** for chat-style questions and answers over the RBA knowledge base.

**Important:**  
LLM / AI tools that generate code for this project MUST:

- Follow this spec strictly.
- **Not introduce new services, directories, frameworks, or dependencies** beyond what is defined here, unless explicitly instructed later.
- **Not rename** top-level directories or tables.
- **Not change** the core architecture (MinIO + PostgreSQL + Streamlit + Python + uv) without explicit human instructions.

---

## 1. High-Level Overview

### 1.1 Project name

**RBA Document Intelligence Platform**

### 1.2 Core idea

- Offline: crawl & ingest **Reserve Bank of Australia (RBA)** PDF reports (e.g. SMP, FSR, economic indicators snapshots).
- Process PDFs into:
  - Cleaned text chunks
  - Document-level metadata
  - Optional structured fields (forecasts, risk tags, etc.)
- Store everything in:
  - **MinIO** (object storage, S3-compatible) for raw PDFs & optional derived artifacts.
  - **PostgreSQL** for metadata, text chunks, embeddings, evaluations, and chat logs.
- Online: expose a **Streamlit chat UI** where the user asks natural language questions, the backend runs **RAG over the RBA corpus**, and returns:
  - Answer
  - Evidence excerpts
  - Basic analytic commentary (e.g., macro / risk perspective).

No file upload is required in the UI. The knowledge base is pre-built from crawled PDFs.

---

## 2. Non-Negotiable Tech Stack

AI tools must use exactly these:

- **Language**: Python 3.x
- **Package & environment management**: `uv`  
  - Use `pyproject.toml` for dependency definitions.
  - No `conda`, no `poetry`, no `pipenv`.
- **Storage**: MinIO (S3-compatible object storage)
  - Accessed via standard S3 Python client (e.g. `boto3` or equivalent).
- **Embeddings runtime**: HTTP service (containerized FastAPI or Hugging Face text-embeddings-inference) hosting open-source embedding models such as `nomic-ai/nomic-embed-text-v1.5`. Must expose a POST `/embeddings` API compatible with `app/embeddings/client.py`.
- **Database**: PostgreSQL
  - With **pgvector** extension for embeddings.
- **UI**: Streamlit
  - Single-page or small multi-tab app, as simple as possible.
- **RAG**:
  - Embeddings stored in Postgres using pgvector.
  - The LLM provider must be abstracted (e.g. local Ollama, cloud model, etc.), configured via environment variables.
- **PDF parsing**: `pymupdf` (`fitz`) or `pdfplumber` (choose one and stick to it). No random switching per file.

**Do NOT:**

- Do **not** introduce Kafka/Redis/Elasticsearch/extra databases.
- Do **not** introduce extra web frameworks (no FastAPI/Flask unless explicitly asked later).
- Do **not** introduce other UI frameworks (no React/Vue/etc.).
- Do **not** add additional vector databases (no Qdrant, Milvus, Weaviate, etc.) – use Postgres + pgvector only.

---

## 3. System Architecture

### 3.1 Components (logical)

1. **Crawler & Ingestion (scripts)**
   - Scripts to crawl RBA websites, discover PDF URLs, download them into MinIO, and register them in Postgres.
   - Each document row must capture `source_url`, byte length, and a deterministic `content_hash` (e.g., SHA-256) so rerunning the crawler simply skips already ingested PDFs. Never fabricate PDFs from HTML—always download the original binary.
   - Support optional environment filters (e.g., `CRAWLER_YEAR_FILTER`) so engineers can limit ingestion to specific publication years while debugging without modifying code.

2. **PDF Processing Pipeline (batch / worker-style)**
   - A Python module that:
     - Reads pending documents from Postgres.
     - Streams PDF from MinIO.
     - Extracts text per page; cleans it; splits into chunks.
     - Writes text chunks + metadata to Postgres.
     - Optionally writes page-level text into MinIO (e.g. `text/` prefix).

3. **Embedding & Indexing Pipeline**
   - A Python module that:
     - Detects text chunks without embeddings.
     - Calls an embedding model (configurable; can be local or remote) through the embedding service container/API.
     - Stores embeddings in a `pgvector` column (current dimension: 768 to match `nomic-ai/nomic-embed-text-v1.5`).

4. **RAG Query Engine**
   - Library code (not a separate service) that:
     - Given a user question:
       - Computes question embedding.
       - Does similarity search over chunk embeddings in Postgres.
       - Optionally filters by doc_type, date range, etc.
       - Assembles a RAG prompt (context + question).
       - Calls the LLM to generate an answer.
     - Returns structured answer object: `answer`, `evidence`, `analysis`. The `analysis` field must summarize which documents/pages grounded the response so downstream UIs can show reasoning breadcrumbs.

5. **Streamlit UI**
   - A simple chat-like frontend that:
     - Accepts user queries.
     - Calls the RAG engine via Python function calls.
     - Displays answer, evidence snippets, and basic metadata.

### 3.2 Execution Modes

- **Batch mode**:
  - Crawl & ingest PDFs in bulk (e.g. initial corpus).
  - Process them through the pipeline (PDF → text → chunks → embeddings).
- **Incremental mode**:
  - Scripts can be re-run periodically to:
    - Detect new PDFs.
    - Insert only missing documents & chunks.
    - Generate embeddings only for new chunks.

No actual scheduling/orchestration tool is required in v1. Cron or manual runs are enough.

---

## 4. Project Structure

All code in Python, with this top-level structure:

```text
.
├── claude.md              # this spec
├── pyproject.toml         # uv-managed dependencies
├── uv.lock                # generated by uv (do not hand-edit)
├── app/
│   ├── config.py          # environment/config loading
│   ├── db/
│   │   ├── models.py      # SQLAlchemy models & pgvector types
│   │   └── session.py     # Postgres session/connection management
│   ├── storage/
│   │   ├── base.py        # Storage interface (save/get)
│   │   └── minio_s3.py    # MinIO implementation
│   ├── pdf/
│   │   ├── parser.py      # PDF → raw text/page model
│   │   ├── cleaner.py     # header/footer removal, normalization
│   │   └── chunker.py     # text chunking logic
│   ├── embeddings/
│   │   ├── client.py      # embedding model client (LLM-agnostic)
│   │   └── indexer.py     # batch embedding generation & storage
│   ├── rag/
│   │   ├── retriever.py   # similarity search & filtering
│   │   ├── llm_client.py  # LLM call wrapper (e.g. Ollama/OpenAI)
│   │   └── pipeline.py    # end-to-end RAG pipeline for a query
│   └── ui/
│       └── streamlit_app.py  # main Streamlit entrypoint
├── scripts/
│   ├── crawler_rba.py     # RBA PDF discovery & ingestion
│   ├── process_pdfs.py    # run PDF → text → chunks for pending docs
│   ├── build_embeddings.py# generate embeddings for unprocessed chunks
│   └── debug_dump.py      # optional: inspect DB contents for debugging
└── storage/               # local temp files if needed (not required)
```

**Strict rule:**  
AI tools must **not** add new top-level directories or rename existing ones without explicit human instruction.

---

## 5. Configuration & Environment

### 5.1 Configuration mechanism

Use a central `app/config.py` to load configuration via environment variables, e.g.:

- `DATABASE_URL` (Postgres)
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_SECURE`
- `MINIO_BUCKET_RAW_PDF` (e.g. `rba-raw-pdf`)
- `MINIO_BUCKET_DERIVED` (e.g. `rba-derived`)
- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_API_BASE_URL`
- `EMBEDDING_BATCH_SIZE`
- `EMBEDDING_API_TIMEOUT`
- `LLM_MODEL_NAME`
- `LLM_API_BASE_URL`
- `LLM_API_KEY` (if needed, or path to local Ollama)
- `CRAWLER_YEAR_FILTER` (optional, comma-separated years to ingest)

**Do not** hard-code credentials or local paths in code.

### 5.2 Make/uv usage

- Dependencies are defined in `pyproject.toml`.
- Basic workflow (always go through the Makefile wrappers so `uv run` executes inside the container with consistent flags):
  - `make bootstrap` – install dependencies (`uv sync`).
  - `make crawl`
  - `make process`
  - `make embeddings`
  - `make refresh`
  - `make streamlit`
  - (handled automatically) Postgres initializes schema/indexes from `/docker/postgres/initdb.d/*.sql` the first time the volume is created.
  - `make test ARGS="-q"` before every commit/PR; add targeted unit tests beside new helpers.
  - `make up-embedding` to ensure the embedding API is online before running backfills.

**Do not** introduce other environment managers or direct `pip install` instructions.

---

## 6. PostgreSQL Schema

Use SQLAlchemy (or SQLModel) to define models in `app/db/models.py`. Minimum required tables:

### 6.1 `documents`

Represents a high-level PDF document (e.g. one SMP / FSR release).

Fields:

- `id: UUID` (primary key)
- `source_system: TEXT` (e.g. `RBA_SMP`, `RBA_FSR`, `RBA_SNAPSHOT`)
- `s3_key: TEXT` (path in MinIO, e.g. `raw/smp/2025-02.pdf`)
- `doc_type: TEXT` (e.g. `SMP`, `FSR`, `SNAPSHOT`)
- `title: TEXT`
- `publication_date: DATE`
- `status: TEXT` (`NEW`, `TEXT_EXTRACTED`, `CHUNKS_BUILT`, `EMBEDDED`, `FAILED`)
- `created_at: TIMESTAMPTZ`
- `updated_at: TIMESTAMPTZ`

### 6.2 `pages` (optional but recommended)

Represents page-level extracted text.

- `id: BIGSERIAL`
- `document_id: UUID` (foreign key to `documents`)
- `page_number: INT`
- `raw_text: TEXT` (before heavy cleaning)
- `clean_text: TEXT` (after cleaning)
- `created_at: TIMESTAMPTZ`

### 6.3 `chunks`

Represents text chunks used for RAG.

- `id: BIGSERIAL`
- `document_id: UUID`
- `page_start: INT` (first page covered by this chunk)
- `page_end: INT` (last page covered)
- `chunk_index: INT` (0-based index per document)
- `text: TEXT`
- `embedding: VECTOR` (pgvector column, nullable until generated)
- `section_hint: TEXT` (e.g. `Inflation`, `Housing`, optional)
- `text_tsv: TSVECTOR` (materialized full-text index maintained via trigger for lexical search)
- `created_at: TIMESTAMPTZ`
- `updated_at: TIMESTAMPTZ`

### 6.4 `chat_sessions`

For UI sessions.

- `id: UUID`
- `created_at: TIMESTAMPTZ`
- `metadata: JSONB` (optional)

### 6.5 `chat_messages`

Per message in a chat session.

- `id: BIGSERIAL`
- `session_id: UUID` (FK to `chat_sessions`)
- `role: TEXT` (`user`, `assistant`, `system`)
- `content: TEXT`
- `created_at: TIMESTAMPTZ`
- `metadata: JSONB` (e.g., RAG context IDs used for that response)

**Do not** create extra tables unless necessary and explicitly discussed.

---

## 7. PDF Processing: Production-style Requirements

PDF processing must simulate realistic production constraints and problems.

### 7.1 Ingestion & Storage

- **Raw PDFs** are stored in MinIO under a deterministic key scheme, e.g.:

  - `raw/smp/YYYY-MM.pdf`
  - `raw/fsr/YYYY-MM.pdf`
  - `raw/snapshot/YYYY-MM.pdf`

- `scripts/crawler_rba.py` must:
  - Discover PDF URLs from RBA pages.
  - Download **streamed** (using `stream=True`) to avoid loading entire PDF in memory at once.
  - Upload to MinIO with streaming friendly approach.
  - Insert a `documents` row with status `NEW`.

### 7.2 Extraction Strategy (streaming-aware)

- When processing PDFs:
  - Do **not** load the entire PDF into memory if avoidable.
  - Iterate page-by-page using the PDF library.
  - For each page:
    - Extract text.
    - Store page-level text in `pages` table (or at least buffer until commit).
  - Commit to DB in batches (e.g. every N pages) to avoid huge transactions.

### 7.3 Cleaning Requirements

PDF text cleaning must handle:

1. **Repeated Headers/Footers**
   - RBA PDFs typically repeat:
     - Report title
     - Date
     - Page number
   - Implement header/footer detection based on:
     - First page sample vs subsequent pages
     - Common repeated lines at top/bottom of each page
   - Remove or blank out these lines before chunking.

2. **Hyphenated words across line breaks**
   - Merge words split with `-` at end-of-line where appropriate.
   - Avoid merging legitimate hyphenated terms (heuristic based on dictionary or simple rules).

3. **Multiple line breaks**
   - Normalize multiple `\n` into paragraph-level breaks (e.g. single `\n` inside paragraph, double `\n\n` between paragraphs).

4. **Non-text content**
   - Tables and charts may not parse cleanly.
   - Initial version: treat them as plain text; do **not** implement OCR or structural table parsing.
   - Mark pages with low text density as `is_sparse` in `pages` metadata if you want to skip them during RAG.

5. **Unicode / Encoding**
   - Normalize to UTF-8.
   - Strip non-printable characters.

AI tools must **not** add OCR or complex table extraction unless explicitly requested.

### 7.4 Chunking Strategy

Chunking must be:

- **Size-aware**:
  - Aim for ~500–1500 tokens per chunk (approx. based on characters, e.g. 2k–4k chars).
- **Boundary-aware**:
  - Prefer breaking on paragraph boundaries.
  - Avoid splitting in the middle of a sentence if possible.

Chunk metadata must record:

- `document_id`
- `page_start`, `page_end`
- `chunk_index`
- `section_hint` (if simple heuristics can infer it from headings, otherwise leave null)

### 7.5 Batch vs Streaming Modes

**Batch mode** (e.g. `scripts/process_pdfs.py`):

- Select a batch of `documents` with `status='NEW'` or `TEXT_EXTRACTED`.
- For each document:
  - Process page-by-page.
  - Update status:
    - `NEW` → `TEXT_EXTRACTED` after page extraction
    - `TEXT_EXTRACTED` → `CHUNKS_BUILT` after chunking
- Use **small batches** and commit frequently:
  - E.g. process up to N documents per run, or stop after M pages, to simulate production jobs.

**Streaming-like behavior**:

- The pipeline should be safe to re-run:
  - If partially processed, it should resume without corrupting data.
  - Avoid double-inserting chunks; use idempotent checks (e.g. if chunks exist, skip or rebuild explicitly).

### 7.6 Error Handling & Logging

- If PDF parsing fails:
  - Mark `documents.status = 'FAILED'`
  - Store error details in logs (and optionally in a `metadata` JSONB column).
- Use Python `logging` module:
  - No random printing; logs must be used for debugging/monitoring style output.

---

## 8. Embeddings & Retrieval

### 8.1 Embedding Generation

- Embedding client defined in `app/embeddings/client.py`.
- It must:
  - Accept a list of chunk texts.
  - Return a list of vectors.
  - Use configuration to determine model & API endpoint (e.g. local Ollama, remote embedding API).
- `app/embeddings/indexer.py`:
  - Select `chunks` where `embedding IS NULL`.
  - Process in batches (configurable batch size).
  - Insert/update embeddings in Postgres.

### 8.2 pgvector Setup

- Use `VECTOR` column type in `chunks.embedding`.
- Create appropriate index:

  ```sql
  CREATE INDEX idx_chunks_embedding
  ON chunks
  USING ivfflat (embedding vector_l2_ops)
  WITH (lists = 100);
  ```

- AI tools must not create other vector stores.

### 8.3 Retrieval Logic

Defined in `app/rag/retriever.py`:

- Given:
  - `query_text`
  - Optional filters: `doc_type`, `date_from`, `date_to`, `limit`
- Steps:
  1. Compute query embedding.
  2. Perform `ORDER BY embedding <-> query_embedding LIMIT k` using pgvector.
  3. Apply filters where specified.
  4. Return:
     - chunk texts
     - associated `document` metadata
     - `page_start/page_end`

No BM25/keyword index in v1; only vector retrieval.

---

## 9. LLM Integration & RAG Pipeline

### 9.1 LLM Client

`app/rag/llm_client.py`:

- Wrap a generic HTTP API:
  - Accepts `prompt` (or messages).
  - Returns model text output.
- Must be configurable via env:
  - base URL, model name, API key, etc.
- Do not hardcode a specific vendor.

### 9.2 RAG Pipeline

`app/rag/pipeline.py` implements:

```python
def answer_query(query: str, session_id: Optional[UUID] = None) -> dict:
    ...
```

Steps:

1. Call retriever to get top-k relevant chunks.
2. Build a prompt with:
   - System instructions (e.g. "You are an assistant summarizing RBA reports")
   - Context: selected chunks + metadata (date, doc_type)
   - User query
3. Call `llm_client`.
4. Parse result into a structured dict:

```python
{
  "answer": "<string>",
  "evidence": [
    {
      "document_id": "...",
      "doc_type": "SMP",
      "publication_date": "2025-02-01",
      "snippet": "...",
      "pages": [12, 13]
    },
    ...
  ],
  "analysis": "<optional short investment/macro view>"
}
```

5. Persist the interaction to `chat_sessions` / `chat_messages`.

---

## 10. Streamlit UI Spec

`app/ui/streamlit_app.py`:

- Single page with:
  - Text input for the user question.
  - A “Send” button.
  - Chat history display:
    - User messages.
    - Assistant answers.
  - For each assistant answer:
    - Show the answer text.
    - Collapsible section “Evidence” listing:
      - Document type
      - Publication date
      - Short snippet
- UI should be minimal, functional, and not over-engineered:
  - No fancy routing, no theme overkill.
  - Just enough to demo the RAG system.

Streamlit app calls directly into `answer_query()` from `app/rag/pipeline.py`.  
No extra microservices or HTTP layer is needed.

---

## 11. RBA Crawling Guidelines

`scripts/crawler_rba.py` must:

- Focus on known, stable RBA PDF sources (SMP, FSR, snapshots).
- Implement:
  - **Discovery**:
    - Use `requests` + `BeautifulSoup` to parse RBA publication listing pages.
    - Extract PDF links; normalize to absolute URLs.
  - **Deduplication**:
    - Check if a document with the same `source_system` + `publication_date` already exists in `documents`.
  - **Download**:
    - Stream download PDFs (chunked) to avoid large memory spikes.
  - **Upload to MinIO**:
    - Store in `raw/<doc_type>/<filename>.pdf`
  - **Register in Postgres**:
    - Insert into `documents` with `status = 'NEW'`.

AI tools must **not** scrape random unrelated websites; stay focused on RBA sources.

---

## 12. Future Extensions (Do not implement unless asked)

The following are explicitly **future ideas**, NOT part of the initial implementation:

- OCR pipeline for scanned PDFs.
- Advanced table extraction for numeric datasets.
- Additional vector stores or search engines (Elasticsearch, etc.).
- External schedulers (Airflow, Prefect, etc.).
- Multi-user auth and access control in Streamlit.

AI tools should **not** implement these unless prompted explicitly.

---

## 13. Summary

If an AI is generating code for this project, it must:

- Use **Python + uv + PostgreSQL + MinIO + Streamlit** only.
- Follow the project structure and schema defined here.
- Treat MinIO as S3 storage for raw PDFs and Postgres as single source of truth for metadata, text chunks, embeddings, and chat.
- Handle PDF processing as if it were production:
  - streaming download
  - per-page processing
  - robust cleaning
  - incremental & resumable batch processing
- Implement a minimal but usable Streamlit chat UI that calls the RAG pipeline.

Any deviation from this spec must be explicitly approved by the human; the AI should not "invent" new architectures, services, or directories on its own.
# RBA Document Intelligence Platform - Complete Codebase Structure

## Project Overview

The **RBA Document Intelligence Platform** is a production-style Python application that crawls, processes, and provides RAG-based search over Reserve Bank of Australia PDF publications. It combines PDF processing pipelines, vector embeddings, hybrid retrieval, and an interactive Streamlit UI.

**Tech Stack:**
- Language: Python 3.11+
- Package Manager: `uv`
- Database: PostgreSQL + pgvector
- Storage: MinIO (S3-compatible)
- Embeddings: Hugging Face `nomic-embed-text-v1.5` (768-dim)
- LLM: Ollama (local) with configurable models (default: `qwen2.5:7b`)
- UI: Streamlit
- Instrumentation: Lightweight hook bus for events

---

## Directory Structure

```
rba-doc-intel/
├── app/                          # Main application package
│   ├── config.py                # Environment configuration & settings
│   ├── db/                       # Database layer
│   │   ├── models.py            # SQLAlchemy ORM models (Document, Chunk, etc.)
│   │   └── session.py           # PostgreSQL connection management
│   ├── storage/                  # Object storage layer
│   │   ├── base.py              # Abstract StorageAdapter interface
│   │   └── minio_s3.py          # MinIO implementation
│   ├── pdf/                      # PDF processing pipeline
│   │   ├── parser.py            # PyMuPDF text extraction per page
│   │   ├── cleaner.py           # Header/footer removal & normalization
│   │   ├── chunker.py           # Paragraph-aware recursive chunking (768-token)
│   │   └── table_extractor.py   # Optional table/chart detection (Camelot)
│   ├── embeddings/               # Embedding generation & indexing
│   │   ├── client.py            # HTTP client for embedding API
│   │   └── indexer.py           # Batch embedding backfill logic
│   ├── rag/                      # RAG pipeline & retrieval
│   │   ├── retriever.py         # Hybrid semantic+lexical search (pgvector + FTS)
│   │   ├── reranker.py          # Optional cross-encoder reranking
│   │   ├── llm_client.py        # LLM HTTP wrapper (Ollama)
│   │   ├── pipeline.py          # Main RAG query orchestration
│   │   ├── eval.py              # Evaluation metrics & example sets
│   │   ├── safety.py            # Query safety checks
│   │   └── hooks.py             # Lightweight pub/sub event bus
│   └── ui/
│       └── streamlit_app.py     # Chat UI with feedback loop
│
├── scripts/                      # Operational CLI scripts
│   ├── crawler_rba.py           # RBA PDF discovery & ingestion
│   ├── process_pdfs.py          # Extract text → chunks (parallel workers)
│   ├── build_embeddings.py      # Generate embeddings with backfill (parallel)
│   ├── refresh_pdfs.py          # End-to-end convenience wrapper
│   ├── debug_dump.py            # Quick stats (doc/chunk/session counts)
│   ├── wait_for_services.py     # Startup readiness probe
│   ├── export_feedback_pairs.py # Extract thumbs up/down for fine-tuning
│   └── finetune_lora_dpo.py     # LoRA + DPO trainer (M-series Mac friendly)
│
├── docker/                       # Container definitions
│   ├── embedding/
│   │   ├── app.py               # FastAPI embedding service (auto-GPU detect)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── postgres/
│       └── initdb.d/            # PostgreSQL init scripts
│           ├── 00_extensions.sql         # pgvector, pgcrypto
│           ├── 01_create_tables.sql     # Schema definition
│           └── 02_create_indexes.sql    # Vector/FTS/composite indexes
│
├── tests/                        # Unit & integration tests
│   ├── pdf/
│   │   ├── test_cleaner.py
│   │   └── test_chunker.py
│   ├── rag/
│   │   ├── test_hooks.py
│   │   └── test_pipeline.py
│   └── ui/
│       └── test_feedback.py
│
├── docs/                         # Documentation & guides
│   ├── COMPLETE_PDF_RAG_INTERVIEW_GUIDE.md
│   └── IMPROVEMENTS_SUMMARY.md
│
├── pyproject.toml               # Dependencies & project config (uv)
├── docker-compose.yml           # Full stack orchestration
├── Dockerfile                   # App container (Python 3.11 + uv)
├── .env.example                 # Environment template
├── README.md                     # Quick start & usage guide
├── CLAUDE.md                     # Hard constraints & spec
├── LEARN.md                      # Technical deep-dive
├── PLAN.md                       # Current & future roadmap
├── AGENTS.md                     # AI agent guidelines
└── uv.lock                       # Locked dependencies (generated)
```

---

## Core Components

### 1. **Configuration (`app/config.py`)**

Centralized environment-driven settings with validation:

```python
@dataclass(frozen=True)
class Settings:
    database_url: str
    minio_endpoint: str
    embedding_api_base_url: str
    llm_model_name: str
    use_reranking: bool = False
    reranker_model_name: str | None = None
```

**Environment Variables:**
- `DATABASE_URL` - PostgreSQL connection string
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` - S3 storage
- `EMBEDDING_API_BASE_URL` - Embedding service URL
- `LLM_MODEL_NAME` - Ollama model (default: `qwen2.5:7b`)
- `CRAWLER_YEAR_FILTER` - Optional year filter (e.g., `2024`)

### 2. **Database Layer (`app/db/`)**

#### Models (`models.py`):

| Model | Purpose | Key Fields |
|-------|---------|-----------|
| **Document** | High-level PDF record | id (UUID), source_system, s3_key, doc_type, publication_date, status, content_hash |
| **Page** | Extracted page text | document_id (FK), page_number, raw_text, clean_text |
| **Chunk** | RAG text segments | document_id (FK), text, embedding (VECTOR 768), section_hint, page_start/end |
| **ChatSession** | User conversation | id (UUID), created_at, metadata_json |
| **ChatMessage** | Turn in conversation | session_id (FK), role, content, metadata_json |
| **Feedback** | User ratings | chat_message_id (FK), score (1/-1), comment, tags |
| **EvalExample** | Test queries | query, gold_answer, difficulty, category |
| **EvalRun** | Eval session | config, status, summary_metrics |
| **EvalResult** | Result per query | eval_run_id (FK), eval_example_id (FK), llm_answer, scores |
| **Table** | Extracted tables | document_id (FK), page_number, structured_data (JSONB) |

**Status Flow:**
```
NEW → TEXT_EXTRACTED → CHUNKS_BUILT → EMBEDDED
```

#### Session Management (`session.py`):

Context manager for SQLAlchemy sessions with automatic commit/rollback.

### 3. **Storage Layer (`app/storage/`)**

**Abstract Base (`base.py`):**
- `ensure_bucket(name)`
- `upload_file(bucket, object_name, file_path)`
- `upload_fileobj(bucket, object_name, file_obj)` - Streaming upload
- `download_file(bucket, object_name, destination)`
- `object_exists(bucket, object_name)`

**MinIO Implementation (`minio_s3.py`):**
Wraps boto3/MinIO client for S3-compatible storage.

**Bucket Strategy:**
- `rba-raw-pdf/` - Raw PDF blobs (keyed as `raw/{doc_type}/{year}-{month}.pdf`)
- `rba-derived/` - Optional extracted text, charts, tables

### 4. **PDF Processing Pipeline (`app/pdf/`)**

#### Parser (`parser.py`)
- **Tool:** PyMuPDF (`fitz`)
- **Output:** List of page text strings
- **Streaming:** Page-by-page to avoid memory spike

```python
def extract_pages(pdf_path: Path) -> List[str]:
    """Extract raw text per page."""
```

#### Cleaner (`cleaner.py`)
Removes headers/footers and normalizes text:

**RBA-Specific Patterns:**
- Headers: Page numbers, report titles, spaced caps (e.g., "S T A T E M E N T")
- Footers: URLs, copyright notices, standalone page numbers

**Strategy:**
1. Pattern-based removal (regex)
2. Frequency-based detection (lines in 80%+ of pages)
3. Whitespace normalization

**Output:** Clean pages with paragraph structure preserved

#### Chunker (`chunker.py`)
Recursive splitting with overlap:

- **Strategy:** Split on paragraph → sentence → word boundaries
- **Max chunk size:** 768 tokens (~3,500 chars)
- **Overlap:** 15% (default) for context continuity
- **Section hints:** Extracts heading from first 200 chars of chunk

```python
def chunk_pages(
    clean_pages: List[str],
    max_tokens: int = 768,
    overlap_pct: float = 0.15
) -> List[Chunk]:
```

#### Table Extractor (`table_extractor.py`)
Optional Camelot-based table detection and JSONB storage.

### 5. **Embeddings (`app/embeddings/`)**

#### Client (`client.py`)
HTTP wrapper for embedding API:
- Calls `/embeddings` POST endpoint
- Default model: `nomic-ai/nomic-embed-text-v1.5` (768-dim)
- Batch support for efficiency

#### Indexer (`indexer.py`)
Backfill logic:
- Finds chunks where `embedding IS NULL`
- Generates vectors via client
- Updates `chunks.embedding` column
- Batch processing with parallel workers

### 6. **RAG Pipeline (`app/rag/`)**

#### Retriever (`retriever.py`)
**Hybrid Retrieval (Semantic + Lexical):**

1. **Semantic Component:** pgvector HNSW cosine similarity
2. **Lexical Component:** Postgres full-text search (tsvector)
3. **Fusion:** Weighted combination (70% semantic, 30% lexical by default)

```python
def retrieve_similar_chunks(
    session: Session,
    query_text: str,
    query_embedding: Sequence[float],
    limit: int = 5,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
    rerank: bool = False,
) -> List[RetrievedChunk]:
```

**Reranking (Optional):**
- Cross-encoder (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Trade-off: +200–500ms latency for +25–40% accuracy
- Disabled by default; enable with `USE_RERANKING=1`

#### LLM Client (`llm_client.py`)
HTTP wrapper for Ollama:
- Accepts prompt + context
- Configurable model (default: `qwen2.5:7b`)
- Streaming support for token-by-token UI

#### Pipeline (`pipeline.py`, ~260 lines)
Main RAG orchestration:

```python
async def answer_query(
    query: str,
    session_id: Optional[UUID] = None
) -> dict:
```

**Flow:**
1. Embed query
2. Retrieve top-k chunks (hybrid)
3. Optional rerank
4. Build prompt (system + context + query)
5. Call LLM (streaming)
6. Parse response
7. Emit hooks (`rag:answer_completed`, etc.)
8. Persist to `chat_messages` table
9. Return structured answer

**Output:**
```python
{
    "answer": "...",
    "evidence": [
        {
            "document_id": "...",
            "doc_type": "SMP",
            "publication_date": "2025-02-01",
            "snippet": "...",
            "pages": [12, 13]
        }
    ],
    "analysis": "..."
}
```

#### Hooks (`hooks.py`)
Lightweight pub/sub event bus:

```python
from app.rag.hooks import hooks

hooks.emit("rag:query_started", query="...", session_id="...")
hooks.subscribe("rag:answer_completed", handler_fn)
```

**Events:**
- `rag:query_started`
- `rag:retrieval_complete`
- `rag:stream_chunk`
- `rag:answer_completed`
- `ui:question_submitted`
- `ui:feedback_recorded`

#### Safety (`safety.py`)
Query validation & safety checks (future expansion).

#### Eval (`eval.py`)
Evaluation framework:
- Example sets (gold queries, expected keywords)
- Eval runs (batch evaluations with metrics)
- Results tracking (NDCG, recall, latency)

#### Reranker (`reranker.py`)
Cross-encoder reranking when enabled:
- Scores (query, candidate) pairs
- Filters top-k by score
- Optional batch processing

### 7. **UI (`app/ui/streamlit_app.py`)**

**Features:**
- Chat interface with message history
- Token-by-token streaming from LLM
- Evidence section with:
  - Document type, publication date
  - Snippet excerpt
  - Page numbers & section hints
- Thumbs up/down feedback on every response
- Session persistence

**Feedback Integration:**
- Stores in `feedback` table
- Emits `ui:feedback_recorded` hook
- Can be exported for fine-tuning (→ `export_feedback_pairs.py`)

---

## Scripts & Operational Workflows

### `crawler_rba.py`
**RBA PDF Discovery & Ingestion**

**Sources:**
- Statement on Monetary Policy (SMP)
- Financial Stability Review (FSR)
- Annual reports & snapshots

**Flow:**
1. Parse RBA publication pages (BeautifulSoup)
2. Download PDFs (streamed, no memory spike)
3. Compute SHA-256 content hash
4. Check for duplicates via `content_hash`
5. Upload to MinIO (`raw/{doc_type}/{filename}`)
6. Insert into `documents` table with `status='NEW'`

**Configuration:**
- `CRAWLER_YEAR_FILTER`: Limit to specific years (e.g., `2024`)
- `USER_AGENT`: Crawler identification

### `process_pdfs.py`
**PDF → Text → Chunks Pipeline**

**Parallel Processing:**
- ThreadPoolExecutor with configurable workers (default: 4)
- Sequential: ~1 doc/min; Parallel: ~4 docs/min

**Flow (per document):**
1. Fetch from MinIO
2. Extract pages (PyMuPDF)
3. Detect & remove headers/footers
4. Chunk with overlap
5. Persist pages & chunks to DB
6. Update `documents.status` → `CHUNKS_BUILT`

**Args:**
- `--limit`: Max documents to process (default: 10)
- `--workers`: Parallel workers (default: 4)

### `build_embeddings.py`
**Embedding Backfill with Parallel Batches**

**Performance:**
- Sequential: ~50 chunks/sec
- Parallel (4 batches): ~600 chunks/sec (12x speedup on M4)
- GPU (NVIDIA): ~2,500 chunks/sec

**Flow:**
1. Find chunks where `embedding IS NULL`
2. Submit multiple batch jobs in parallel
3. Each batch calls embedding API
4. Update `chunks.embedding` column
5. Mark document as `EMBEDDED`

**Args:**
- `--batch-size`: Chunks per batch (default: 24)
- `--parallel`: Concurrent batches (default: 2)
- `--reset`: Wipe embeddings & downgrade doc status
- `--document-id`: Target specific document

### `refresh_pdfs.py`
**Convenience End-to-End Wrapper**

Sequentially runs:
1. `crawler_rba.main()`
2. `process_pdfs.main()`
3. `build_embeddings.main()`

### `debug_dump.py`
**Quick Database Stats**

Prints counts:
- Documents
- Pages
- Chunks
- Chat sessions & messages

### `wait_for_services.py`
**Startup Readiness Probe**

Waits for Postgres & MinIO to be healthy before app starts. Used in Docker Compose.

### `export_feedback_pairs.py`
**Export Feedback for Fine-tuning**

Extracts thumbs up/down feedback as JSONL preference pairs:
```json
{
  "query": "...",
  "chosen": "...",
  "rejected": "..."
}
```

Output ready for DPO training.

### `finetune_lora_dpo.py`
**LoRA + DPO Fine-tuning**

**Features:**
- Lightweight LoRA adapters (parameter-efficient)
- DPO loss for preference learning
- M-series Mac friendly (torch MPS support)
- Default base model: `microsoft/phi-2`

**Output:**
- Trained adapter in `models/rba-lora-dpo/`
- Can be merged back with base model

---

## PostgreSQL Schema

### Database Initialization (`docker/postgres/initdb.d/`)

#### `00_extensions.sql`
Creates PostgreSQL extensions:
- `pgcrypto` - UUID generation
- `vector` - pgvector for embeddings

#### `01_create_tables.sql`
Full schema with 10 tables (see **Database Layer** section above).

**Constraints:**
- Foreign key cascades on delete
- Unique constraint on `documents.content_hash`
- Timestamps (UTC) with auto-updates

#### `02_create_indexes.sql`
Production-grade indexes:

| Index | Type | Purpose |
|-------|------|---------|
| `idx_chunks_embedding_hnsw` | HNSW (vector_cosine_ops) | Fast approximate NN search |
| `idx_documents_type_date` | Composite (doc_type, publication_date) | Filtering by type/date |
| `idx_documents_status` | Partial (status = active) | Pipeline state tracking |
| `idx_chunks_document_id` | Covering (includes text, pages) | Join optimization |
| `idx_chunks_null_embedding` | Partial (embedding IS NULL) | Batch backfill queries |
| `idx_chunks_text_fts` | GIN (tsvector) | Full-text search |
| `idx_chat_messages_session` | Composite (session_id, created_at DESC) | Conversation history |
| `idx_chat_sessions_created` | Descending (created_at) | Session listing |

**Triggers:**
- `chunks_text_tsv_trigger` - Auto-maintains `text_tsv` on INSERT/UPDATE

---

## Docker Compose Services

### Full Stack (`docker-compose.yml`)

| Service | Image | Purpose |
|---------|-------|---------|
| **postgres** | pgvector/pgvector:pg15 | Primary database |
| **minio** | minio/minio | S3-compatible object storage |
| **embedding** | Custom (docker/embedding/Dockerfile) | FastAPI embedding service |
| **llm** | ollama/ollama | Local LLM runtime |
| **app** | Custom (Dockerfile) | Streamlit UI + scripts |

**Key Features:**
- Health checks (postgres)
- Automatic restart (embedding: `unless-stopped`)
- Shared volumes for models & data
- Environment variable injection (`.env`)

### Embedding Service (`docker/embedding/`)

**Responsibilities:**
- Host embedding model (default: `nomic-ai/nomic-embed-text-v1.5`)
- Auto-detect GPU (CUDA, MPS, CPU fallback)
- Expose `/embeddings` POST endpoint
- Batch inference support

**Performance:**
- M-series Mac (MPS): ~120 chunks/sec @ batch_size=32
- NVIDIA GPU (CUDA): ~300+ chunks/sec
- CPU: ~10–20 chunks/sec

---

## Dependencies (`pyproject.toml`)

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| sqlalchemy | ≥2.0.25 | ORM & query builder |
| pgvector | ≥0.2.4 | Vector type for SQLAlchemy |
| psycopg[binary] | ≥3.1.17 | PostgreSQL driver |
| minio | ≥7.2.4 | MinIO/S3 client |
| boto3 | ≥1.34.40 | AWS SDK (for S3 compat) |
| requests | ≥2.31.0 | HTTP client |
| beautifulsoup4 | ≥4.12.3 | HTML parsing (crawler) |
| pymupdf | ≥1.23.9 | PDF text extraction |
| streamlit | ≥1.32.0 | UI framework |
| pytest | ≥7.4.4 | Testing |
| python-dateutil | ≥2.8.2 | Date utilities |
| tenacity | ≥8.2.3 | Retry logic |
| camelot-py[cv] | ≥0.11.0 | Table extraction |
| opencv-python | ≥4.10.0 | Image processing |
| sentence-transformers | ≥2.3.0 | Cross-encoder reranking |
| torch | ≥2.1.0 | PyTorch (MPS support) |
| transformers | ≥4.36.0 | Hugging Face models |
| peft | ≥0.7.0 | LoRA/QLoRA |
| trl | ≥0.7.0 | DPO trainer |
| datasets | ≥2.16.0 | Data loading |
| accelerate | ≥0.25.0 | Distributed training |

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "ruff>=0.3.0",          # Fast linter/formatter
    "mypy>=1.8.0",          # Static type checker
    "types-requests",       # Type stubs
    "types-beautifulsoup4"  # Type stubs
]
```

---

## Testing (`tests/`)

### Structure

```
tests/
├── pdf/
│   ├── test_chunker.py      # Chunking logic tests
│   └── test_cleaner.py      # Header/footer removal tests
├── rag/
│   ├── test_hooks.py        # Hook bus pub/sub
│   └── test_pipeline.py     # End-to-end RAG queries
└── ui/
    └── test_feedback.py     # Feedback persistence & events
```

### Running Tests

```bash
make test                                          # All tests
make test ARGS="tests/ui/"                        # Specific module
```

### Linting

```bash
make lint                                          # Lint check
make format                                        # Auto-format
```

---

## Configuration Files

### `.env.example`

Template for environment variables:

```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rba_dev
DATABASE_URL=postgresql+psycopg://...

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_RAW_PDF=rba-raw-pdf
MINIO_BUCKET_DERIVED=rba-derived

# Embeddings
EMBEDDING_API_BASE_URL=http://embedding:8000
EMBEDDING_MODEL_NAME=nomic-embed-text-v1.5
EMBEDDING_BATCH_SIZE=16
EMBEDDING_API_TIMEOUT=120

# LLM
LLM_MODEL_NAME=qwen2.5:7b
LLM_API_BASE_URL=http://llm:11434

# Reranking (optional)
USE_RERANKING=0

# UI
STREAMLIT_SERVER_PORT=8501

# Crawler
CRAWLER_YEAR_FILTER=2024
```

### `pyproject.toml`

Project metadata & dependencies managed by `uv`. Includes tool configs:
- `[tool.ruff]` - Linter settings
- `[tool.pytest.ini_options]` - Test config
- `[tool.mypy]` - Type checker

---

## Quick Reference: How Everything Works Together

### Ingest Phase

```
┌─────────────────┐
│ RBA Websites    │
└────────┬────────┘
         │
         │ crawler_rba.py
         ↓
    ┌─────────────┐
    │ Download    │  (stream, compute hash)
    │ PDFs        │
    └──────┬──────┘
           │
           ↓
    ┌──────────────┐
    │ MinIO        │
    │ (raw/)       │
    └──────┬───────┘
           │
           ↓
    ┌────────────────────┐
    │ documents table    │
    │ status='NEW'       │
    └────────┬───────────┘
             │
             │ process_pdfs.py
             ↓
    ┌────────────────────────┐
    │ PyMuPDF → Pages        │
    │ Cleaner → No headers   │
    │ Chunker → 768-token    │
    └────────┬───────────────┘
             │
             ↓
    ┌───────────────────────┐
    │ pages table           │
    │ chunks table          │
    │ status='CHUNKS_BUILT' │
    └────────┬──────────────┘
             │
             │ build_embeddings.py
             ↓
    ┌─────────────────────┐
    │ Embedding API       │
    │ (nomic-embed-text)  │
    └────────┬────────────┘
             │
             ↓
    ┌────────────────────┐
    │ chunks.embedding   │
    │ (VECTOR 768)       │
    │ status='EMBEDDED'  │
    └────────┬───────────┘
             │
             ├──────────┐
             ↓          ↓
        pgvector     tsvector
         (HNSW)      (FTS)
```

### Query Phase

```
┌──────────────────┐
│ User Question    │
│ (Streamlit UI)   │
└────────┬─────────┘
         │
         │ pipeline.py
         ↓
    ┌─────────────────┐
    │ Embed query     │
    │ (embedding API) │
    └────────┬────────┘
             │
             ↓
    ┌──────────────────────┐
    │ Retrieve chunks      │
    │ - pgvector HNSW      │
    │ - + ts_rank (FTS)    │
    │ - Fuse (70/30)       │
    │ retriever.py         │
    └────────┬─────────────┘
             │
             ↓ (optional)
    ┌──────────────────┐
    │ Rerank with      │
    │ cross-encoder    │
    │ reranker.py      │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────────────┐
    │ Build prompt             │
    │ + System instructions    │
    │ + Retrieved evidence     │
    │ + User query             │
    └────────┬─────────────────┘
             │
             ↓
    ┌──────────────────┐
    │ Call Ollama LLM  │
    │ Stream tokens    │
    │ llm_client.py    │
    └────────┬─────────┘
             │
             ↓
    ┌──────────────────────────┐
    │ Render in Streamlit      │
    │ - Answer text            │
    │ - Evidence snippets      │
    │ - Section hints          │
    └────────┬─────────────────┘
             │
             ↓
    ┌──────────────────────┐
    │ User feedback        │
    │ (thumbs up/down)     │
    │ feedback table       │
    └──────────────────────┘
```

---

## Development Workflow

### 1. **Setup**
```bash
make bootstrap
```

### 2. **Start Services**
```bash
make up-detached
make llm-pull MODEL=qwen2.5:7b
```

### 3. **Ingest Corpus**
```bash
make crawl
make process
make embeddings ARGS="--batch-size 24 --parallel 2"
```

### 4. **Run UI**
```bash
make ui
# Visit http://localhost:8501
```

### 5. **Debug & Optimize**
```bash
make debug
make embeddings-reset  # Wipe embeddings
```

### 6. **Test & Lint**
```bash
make test
make lint
```

### 7. **Export & Fine-tune**
```bash
make export-feedback
make finetune
```

---

## Key Design Decisions

### Why pgvector?
- No extra services (vs. Qdrant, Weaviate, Milvus)
- Production-grade (Postgres is battle-tested)
- Hybrid search (vector + FTS) in single DB
- Built-in HNSW index support

### Why 768 tokens for chunks?
- Aligns with `nomic-embed-text-v1.5` pre-training window
- Avoids truncation, better embeddings
- Ollama context windows (4k–32k) handle it comfortably
- Industry standard (Pinecone, Anthropic, Cohere recommend 600–1000)

### Why hybrid retrieval (semantic + lexical)?
- Semantic: Captures narrative meaning & context
- Lexical: Catches specific IDs, dates, technical terms
- Fusion: Pinecone, Weaviate, Cohere all recommend this pattern
- Trade-off: Slightly slower, significantly better recall

### Why reranking is optional?
- Adds 200–500ms latency
- Only +25–40% accuracy improvement
- Default disabled for fast iteration & cost
- Enable with `USE_RERANKING=1` for production queries

### Why LoRA + DPO for fine-tuning?
- LoRA: Lightweight, fits on single GPU/M-series Mac
- DPO: Preference learning from feedback (thumbs up/down)
- No massive labeled dataset required
- Keeps base model frozen (no distribution shift)

---

## Monitoring & Observability

### Hooks (`app/rag/hooks.py`)
Event bus for decoupled instrumentation:

```python
from app.rag.hooks import hooks

# Subscribe to events
hooks.subscribe("rag:answer_completed", log_handler)
hooks.subscribe_all(debug_handler)

# Emit events
hooks.emit("rag:query_started", query="...", session_id="...")
```

**Events:**
- `rag:query_started` - User initiates query
- `rag:retrieval_complete` - Chunks retrieved
- `rag:stream_chunk` - Token streamed from LLM
- `rag:answer_completed` - Full answer ready
- `ui:question_submitted` - Question in UI
- `ui:answer_rendered` - Answer displayed
- `ui:feedback_recorded` - User votes (thumbs up/down)

### Logging
Set `LOG_LEVEL=DEBUG` for detailed hook output and pipeline traces.

### Database Queries
Use `debug_dump.py` for quick stats; run `EXPLAIN ANALYZE` for slow queries.

---

## Future Extensions (Out of Scope v1)

- [ ] OCR for scanned PDFs
- [ ] Advanced table extraction & structured data models
- [ ] Multi-user authentication & RBAC
- [ ] Kafka/streaming for real-time ingestion
- [ ] Additional vector stores (Elasticsearch, etc.)
- [ ] Production scheduler (Airflow, Prefect, etc.)

# RBA Document Intelligence – Code Walkthrough

This doc explains the repository file-by-file and, for the core modules, line-by-line. Use it to quickly understand how data flows from PDFs to the Streamlit UI.

> **Notation** – References follow `path:line-start–line-end`. Use `nl -ba <file>` to display exact line numbers locally.

---

## 1. Configuration Bootstrap

### `app/config.py:1–74`

| Lines | Purpose |
|-------|---------|
| 1–11  | Imports + `lru_cache` so settings load once per process. |
| 14–47 | `Settings` dataclass declares every env var (DB URLs, MinIO, embedding + LLM endpoints, batch sizes). |
| 50–74 | `get_settings()` returns the cached instance; every other module calls this instead of touching `os.environ` directly. |

**Key idea:** All scripts/servers read configuration from one typed object which makes environment debugging straightforward.

## 1b. Database Bootstrap (`docker/postgres/initdb.d`)

| File | Lines | Purpose |
|------|-------|---------|
| `00_extensions.sql` | 1–2 | Installs `pgcrypto` (UUIDs) + `pgvector` so we can create vector columns without extra migrations. |
| `01_create_tables.sql` | 1–120 | Declares every table (`documents`, `pages`, `chunks`, `chat_*`, `feedback`, `tables`, `eval_*`). Note: `chunks.embedding` is `VECTOR(768)` and `section_hint`/`text_tsv` columns live here so ORM + SQL stay aligned. |
| `02_create_indexes.sql` | 1–130 | Builds HNSW vector index, composite document indexes, **materializes `text_tsv`**, adds a trigger to keep it updated, and analyzes stats. |

Because these run via the official Postgres entrypoint, `make up` on a clean volume always yields the same schema without invoking Python migrations.

---

## 2. Database Models

### `app/db/models.py`

- `Document` (lines ~23–70): stores source metadata and ingestion status.
- `Page` (70–97): raw + cleaned text per PDF page.
- `Chunk` (97–131): the heart of retrieval – includes `embedding VECTOR(768)` and the new `section_hint` string.
  - Lines 112–116 add `text_tsv` so lexical search reads the persisted tsvector instead of recomputing it. The trigger from `02_create_indexes.sql` keeps it fresh whenever chunk text changes.
- `ChatSession`/`ChatMessage` (131–168): conversation persistence.
- `Feedback` (not shown here but imported in UI) links thumbs-up/down to chat messages.

**Why this matters:** chunk metadata (page ranges, section hints) flows straight to the UI as evidence.

---

## 3. Text Extraction & Chunking

### Pipeline overview (`scripts/process_pdfs.py`)

1. `process_document()` downloads a PDF from MinIO, extracts pages (`parser.extract_pages`), and cleans each page with `clean_text()`.
2. `_persist_pages()` stores raw + cleaned versions for auditing.
3. `chunk_pages()` splits the concatenated text (details below).
4. `_persist_chunks()` saves each chunk + section hint to Postgres (`ChunkModel`).

### Cleaner (`app/pdf/cleaner.py:10–33`)

- Removes headers via `HEADER_RE` (line 7).
- Keeps paragraph breaks (`\n\n`) by grouping non-empty lines into paragraphs.
- Returns joined paragraphs separated by blank lines so chunker can detect them.

### Chunker (`app/pdf/chunker.py:1–164`)

| Lines | Explanation |
|-------|-------------|
| 1–23  | Config: `max_tokens=768`, `overlap_pct=0.15`, `Chunk` dataclass carries `section_hint`. |
| 42–55 | Walks pages, builds `page_boundaries` so we can map char offsets back to page numbers. |
| 56–95 | Sliding window over the entire document text: tries to end a chunk at paragraph break (`\n\n`), sentence break (`. ` / `.\n`), then fallback to spaces/characters. |
| 104–118 | `_extract_section_hint()` inspects the opening lines for enumerated headings (`3.2`, `2.3.1`), named sections (“Chapter 3”), or uppercase headings and boxes. |
| 119–135 | Implements overlap by reusing the trailing tokens from the previous chunk. |
| 140–164 | Additional heuristics convert uppercase headings to title case and catch colon-separated titles. |

**Why line 68 matters:** `target_chars = max_tokens * 4.5` approximates tokens→characters so we avoid expensive tokenizer passes during ingestion.

### Embedding Indexer (`app/embeddings/indexer.py:1–52`)

- Lines 15–31: `generate_missing_embeddings()` pulls up to `batch_size` chunks with `embedding IS NULL`, defaulting to the `EMBEDDING_BATCH_SIZE` env var.
- Lines 33–39: Calls the HTTP embedding service once per batch and writes vectors back to each `Chunk` row.
- Lines 41–50: For every document touched in the batch, re-counts remaining NULL embeddings; when zero, the document gets promoted to `DocumentStatus.EMBEDDED`.

### Embedding CLI (`scripts/build_embeddings.py`)

| Lines | Highlights |
|-------|-----------|
| 35–57 | `embed_single_batch()` is the worker function the thread pool executes; all logging includes the batch ID. |
| 60–82 | `reset_embeddings()` optionally nulls vectors (entire corpus or specific document IDs) and downgrades every affected document back to `CHUNKS_BUILT`. |
| 84–183 | `main()` wires up logging, optional reset, then spins up `parallel_batches` workers until no chunks remain. Each loop sleeps 0.2 s so Postgres + the embedding API aren’t hammered. |
| 185–205 | CLI arguments (`--reset`, repeated `--document-id`) map directly to the function parameters; use these instead of manual SQL when re-chunking. |

---

## 4. Hybrid Retrieval

### `app/rag/retriever.py`

#### Semantic + Lexical fusion (lines ~60–205)

1. Build a vector query: `Chunk.embedding.cosine_distance(query_embedding)` (line 85). Fetch `limit * 2` rows so deduplication has slack.
2. Build a lexical query: `ts_rank_cd(Chunk.text_tsv, websearch_to_tsquery(query_text))` (line 120). Same fetch size because `text_tsv` is precomputed by the trigger from `02_create_indexes.sql`.
3. Merge by `chunk_id`; each entry carries `semantic` and `lexical` scores (lines 140–170).
4. Normalize (`semantic_max`, `lexical_max`) and weight by constants (`SEMANTIC_WEIGHT = 0.7`, `LEXICAL_WEIGHT = 0.3`).
5. Convert to `RetrievedChunk` objects, keeping `section_hint` so evidence shows headings.

#### Deterministic ordering (lines 205–214)

```python
results.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
```

This guarantees stable output ordering even when multiple chunks share identical fused scores.

#### (Optional) Reranking (lines 214–310)

Stub that can re-score candidates with a cross-encoder when needed. Disabled by default but the structure is ready.

---

## 5. RAG Pipeline

### `app/rag/pipeline.py`

| Lines | Explanation |
|-------|-------------|
| 23–36 | `SYSTEM_PROMPT` defines analyst persona: cite docs, include numbers, state gaps. |
| 39–44 | `_format_context()` packages chunks as `[doc_type] Title (pages x-y)` + text. |
| 57–104 | `answer_query()` orchestrates everything:
  1. Embed question (`EmbeddingClient`).
  2. Retrieve chunks (`retrieve_similar_chunks`) with `top_k=2` to stay within LLM context limits.
  3. Compose user message (`Question + Context`).
  4. Call LLM client; if `stream_handler` is provided, uses streaming path (see next section).
  5. Save user + assistant messages to Postgres so the UI can display history/feedback.
  6. Return `AnswerResponse` containing text + evidence array.

---

## 6. LLM Client

### `app/rag/llm_client.py:12–66`

1. `_build_payload()` (lines 19–33) joins system prompt + messages into the format Ollama expects.
2. `complete()` (lines 34–44) performs a blocking HTTP POST with `stream=False` (used by scripts/tests).
3. `stream()` (lines 46–66) sends `stream=True` and iterates over `iter_lines()`. Each JSON chunk is parsed, tokens are passed to `on_token`, and appended to `final_text`.

**Why 240s timeout?** CPU-only Ollama can take a minute+ per answer; the generous timeout prevents spurious failures.

---

## 7. Streamlit UI & Feedback Loop

### `app/ui/streamlit_app.py`

1. **Session state** (lines 32–57): stores `chat_session_id`, `history` (`[{question, answer, evidence, pending, message_id, error}]`), and `feedback_state`.
2. **`render_history()`** (lines ~70–150):
   - Prints each Q/A pair; pending entries show a “(generating…)” suffix.
   - Evidence expander shows `doc_type`, title, section hint, and page range.
   - Thumbs buttons are disabled until the response has a `message_id`. Clicking a button calls `store_feedback()` and updates local state.
3. **`store_feedback()`** (lines 59–108): wraps DB access with `session_scope()`; updates existing feedback or inserts new `Feedback` row.
4. **`handle_submit()`** (lines 157–236):
   - Validates input and appends a “pending” entry to history.
   - Streams tokens via `answer_query(..., stream_handler=on_token)` into that entry.
   - On completion, fetches the latest assistant `ChatMessage` to get `message_id` for future feedback.
   - Errors are captured in `entry["error"]` so the UI can display them.
5. **`main()`** (lines 238–254): renders the textarea form and the history list.

**Feedback persistence path:** UI → `store_feedback()` → `Feedback` table (with optional comment, currently unused). Tests cover update vs insert.

---

## 8. Embedding & LLM Services

### `docker-compose.yml`

- `embedding`: builds from `docker/embedding/Dockerfile`. Runs FastAPI + SentenceTransformers, exposes `/embeddings`.
- `llm`: `ollama/ollama:latest` container; we pull `qwen2.5:1.5b` by default for streaming chat.
- `app`: Streamlit server that waits for Postgres/MinIO/embedding/llm before launching.

**Tip:** use `make up-models` to keep model containers warm during development.

---

## 9. Operational Scripts (remaining ones after cleanup)

| Script | Purpose |
|--------|---------|
| `crawler_rba.py`  | Scrapes RBA PDFs + metadata, pushes to MinIO + Postgres. |
| `process_pdfs.py` | Runs the clean → chunk workflow described above. |
| `build_embeddings.py` | Fills in missing `Chunk.embedding` values in batches. |
| `refresh_pdfs.py` | Convenience wrapper to run crawler + processor + embeddings sequentially. |
| `debug_dump.py`   | Prints counts (documents/pages/chunks) for health checks. |
| `wait_for_services.py` | Entrypoint for `app` container to ensure Postgres/MinIO ready before Streamlit starts. |
| `export_feedback_pairs.py` | Reads thumbs-up/down feedback from Postgres and emits JSONL (`prompt/chosen/rejected`) for preference tuning. |
| `finetune_lora_dpo.py` | Runs a LoRA + DPO trainer (TRL/PEFT) on the exported JSONL to produce adapters under `models/`. |

These scripts cover the supported workflows: ingestion, embeddings, diagnostics, feedback export, and LoRA fine-tuning.

## 10. Observability Hooks (`app/rag/hooks.py`)

- `HookBus` is a zero-dependency pub/sub helper. Use `hooks.subscribe(event, handler)` or `hooks.subscribe_all(handler)` to tap into lifecycle events.
- `answer_query()` emits: `rag:query_started`, `rag:retrieval_complete`, `rag:stream_chunk`, `rag:answer_completed`.
- Streamlit emits: `ui:question_submitted`, `ui:answer_rendered`, `ui:message_committed`, `ui:feedback_recorded`.
- Default debug logging subscriber is registered so setting `LOG_LEVEL=DEBUG` prints every event.

### Feedback Export (`scripts/export_feedback_pairs.py:1–120`)

- Lines 23–52: `_resolve_prompt()` walks backward from an assistant reply to find the most recent user prompt in the same session.
- Lines 55–86: `collect_feedback_pairs()` joins `chat_messages` and `feedback`, buckets positive vs negative answers per prompt.
- Lines 89–113: `export_jsonl()` writes capped pairs per prompt (`--max-pairs-per-prompt`) in the TRL-friendly `{prompt, chosen, rejected}` format.

### Lightweight Fine-tuning (`scripts/finetune_lora_dpo.py:1–130`)

- Lines 22–38: CLI arguments let you set dataset path, output dir, base model, and hyperparameters without editing code.
- Lines 40–74: `load_model_and_tokenizer()` loads the base LLM, applies LoRA adapters to attention matrices, and auto-detects CUDA vs MPS vs CPU.
- Lines 78–108: `training_args` + `DPOTrainer` hook up the preference dataset, streaming logs every five steps; defaults fit on a single M-series Mac.
- Lines 110–118: `trainer.train()` then persists both adapter weights and tokenizer under `models/<name>` so you can later merge/evaluate.

---

## 11. Testing Strategy

- `tests/pdf/test_cleaner.py`: ensures blank lines are preserved and headers removed.
- `tests/rag/test_pipeline.py`: verifies `_compose_analysis()` mentions the right title/pages.
- `tests/ui/test_feedback.py`: unit tests `store_feedback()` for both update and insert flows using a dummy session.

Run the full suite with:

```bash
make test
```

Under heavy development you can run targeted suites:

```bash
make test ARGS="tests/ui/test_feedback.py"
```

This keeps the feedback subsystem verified even if Postgres/MinIO aren’t seeded with data.

---

## Full Data Flow (Cheat Sheet)

1. **Ingestion:** `crawler_rba.py` → `process_pdfs.py` → `build_embeddings.py`.
2. **Query:** Streamlit form → `answer_query()` → hybrid retrieval → LLM streaming → DB persistence.
3. **Feedback:** User clicks thumb → `store_feedback()` → `feedback` table (used later for evaluation/fine-tuning).

Keep this doc open alongside the code when stepping through the system; each section points you to the exact file/line block that implements the described behavior.
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

## Operational Considerations
- Schedule ingestion scripts via cron/Kubernetes Jobs invoking the relevant `make` targets (for example, `make refresh`).
- Back up Postgres and MinIO volumes; document retention policies for raw PDFs vs derived artifacts.
- Define alerting for crawler failures or embedding backlog (simple CLI exit codes consumed by external scheduler).
# RBA Document Intelligence Platform - Quick Reference Guide

## Project at a Glance

**What is it?** A production-style RAG system that crawls RBA PDFs, processes them intelligently, and exposes a chat UI for searching the knowledge base.

**Tech Stack:** Python 3.11 + uv + PostgreSQL + MinIO + Ollama + Streamlit

**Key Features:**
- Parallel PDF processing (1 doc → 4 docs/min)
- Hybrid semantic+lexical retrieval (pgvector HNSW + Postgres FTS)
- Token-by-token LLM streaming
- User feedback loop (thumbs up/down)
- Optional cross-encoder reranking
- LoRA + DPO fine-tuning support

---

## File & Directory Quick Map

| Path | What It Does |
|------|-------------|
| `app/config.py` | Environment settings (DB, MinIO, embedding, LLM) |
| `app/db/` | SQLAlchemy models + DB session management |
| `app/storage/` | MinIO/S3 client wrapper |
| `app/pdf/` | PDF extract → clean → chunk pipeline |
| `app/embeddings/` | Embedding client + batch backfill |
| `app/rag/` | RAG pipeline, retrieval, reranking, hooks, eval |
| `app/ui/` | Streamlit chat interface |
| `scripts/` | Crawler, PDF processor, embedder, fine-tuner |
| `docker/` | Container configs (embedding service, Postgres init) |
| `tests/` | Unit tests (PDF, RAG, UI feedback) |
| `docs/` | Deep dives (interview guide, improvements) |
| `docker-compose.yml` | Full stack orchestration |
| `CLAUDE.md` | Hard spec & constraints |
| `LEARN.md` | Technical design decisions |
| `CODEBASE_STRUCTURE.md` | This detailed reference (generated) |

---

## Database Schema (10 Tables)

```
documents          → Documents (PDFs) with status flow: NEW → TEXT_EXTRACTED → CHUNKS_BUILT → EMBEDDED
├── pages          → Extracted page text (raw + clean)
├── chunks         → RAG segments (text + 768-dim embedding + section_hint)
├── tables         → Extracted table data (Camelot, JSONB)
├── eval_examples  → Test queries (gold answers, difficulty, category)
├── eval_runs      → Evaluation batches (config, status, metrics)
└── eval_results   → Per-query eval result (scores, latency, error)

chat_sessions      → User conversation sessions
├── chat_messages  → Turns in conversation (role: user/assistant/system)
└── feedback       → User ratings (score: 1/-1), comments, tags

Indexes: HNSW (vector), GIN (full-text), composite (type/date, status, session)
```

---

## Operations Checklist

### Setup
- [ ] Copy `.env.example` → `.env` (adjust if needed)
- [ ] `make bootstrap`

### Start Services
```bash
make up-detached
make llm-pull MODEL=qwen2.5:7b
```

### Ingest PDFs
```bash
make crawl
make process
make embeddings ARGS="--batch-size 24 --parallel 2"
```

### Run UI
```bash
make ui  # http://localhost:8501
```

### Debug
```bash
make debug     # Stats
make embeddings-reset  # Wipe embeddings
```

### Test & Lint
```bash
make test
make lint
```

### Fine-tune
```bash
make export-feedback
make finetune
```

---

## Pipeline Flows

### Ingest (PDF → Chunks → Embeddings)

1. **Crawler** (`crawler_rba.py`)
   - Discovers RBA PDF URLs
   - Downloads (streamed) to temp
   - Computes SHA-256 hash
   - Uploads to MinIO `raw/{doc_type}/{date}.pdf`
   - Inserts `documents` (status=NEW)

2. **Processor** (`process_pdfs.py`)
   - Fetches pending PDFs from MinIO
   - PyMuPDF → extract pages (raw text)
   - Cleaner → remove headers/footers (RBA-specific patterns)
   - Chunker → recursive split (768 tokens, 15% overlap)
   - Persist `pages` + `chunks` tables
   - Update `documents` (status=CHUNKS_BUILT)

3. **Embedder** (`build_embeddings.py`)
   - Find chunks where `embedding IS NULL`
   - Call `/embeddings` API (nomic-embed-text-v1.5, 768-dim)
   - Batch + parallel workers for speed
   - Update `chunks.embedding`
   - Mark document (status=EMBEDDED)

### Query (Question → Answer)

1. **User submits question** → Streamlit UI
2. **Embed question** → Embedding API
3. **Retrieve chunks**
   - Semantic: pgvector HNSW cosine similarity
   - Lexical: Postgres full-text search (tsvector)
   - Fuse: 70% semantic + 30% lexical
4. **Optional rerank** → Cross-encoder (if `USE_RERANKING=1`)
5. **Build prompt** → System instructions + evidence + question
6. **Call LLM** → Ollama (stream tokens)
7. **Emit hook** → `rag:answer_completed`
8. **Persist** → `chat_messages` table
9. **Return** → Structured answer (text + evidence + analysis)
10. **User feedback** → Thumbs up/down → `feedback` table

---

## Key Classes & Functions

### Models (`app/db/models.py`)
- `Document` - PDF record (UUID, source_url, doc_type, status, content_hash)
- `Page` - Raw & clean text per page
- `Chunk` - RAG segment (text, embedding VECTOR 768, section_hint)
- `ChatSession` / `ChatMessage` - Conversation history
- `Feedback` - User votes
- `EvalExample` / `EvalRun` / `EvalResult` - Eval framework

### Config (`app/config.py`)
- `Settings` - Frozen dataclass with all env vars
- `get_settings()` - Lazy-load once per process

### PDF (`app/pdf/`)
- `extract_pages(pdf_path)` - PyMuPDF → List[str] pages
- `detect_repeating_headers_footers(pages)` - Regex + frequency
- `chunk_pages(clean_pages, max_tokens=768, overlap_pct=0.15)` - Recursive split
- Optional: Camelot table extraction

### Embeddings (`app/embeddings/`)
- `EmbeddingClient` - HTTP POST `/embeddings` wrapper
- `generate_missing_embeddings(batch_size, limit)` - Backfill logic

### RAG (`app/rag/`)
- `retrieve_similar_chunks(session, query_text, query_embedding, limit=5)` - Hybrid search
- `rerank_chunks(chunks, query, reranker)` - Cross-encoder (optional)
- `answer_query(query, session_id=None)` - Main pipeline (async)
- `hooks.emit(event, **payload)` - Event pub/sub
- `EvalFramework` - NDCG/recall metrics

### UI (`app/ui/streamlit_app.py`)
- Chat interface with message history
- Evidence collapsible with snippets & page numbers
- Feedback buttons (thumbs up/down)

---

## Configuration Reference

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | Required | PostgreSQL connection |
| `MINIO_ENDPOINT` | Required | MinIO/S3 host:port |
| `MINIO_ACCESS_KEY` | minioadmin | S3 access key |
| `MINIO_SECRET_KEY` | minioadmin | S3 secret key |
| `MINIO_BUCKET_RAW_PDF` | rba-raw-pdf | Raw PDFs bucket |
| `MINIO_BUCKET_DERIVED` | rba-derived | Derived data bucket |
| `EMBEDDING_MODEL_NAME` | nomic-embed-text-v1.5 | HF model ID |
| `EMBEDDING_API_BASE_URL` | http://embedding:8000 | Embedding service |
| `EMBEDDING_BATCH_SIZE` | 16 | Chunks per batch |
| `EMBEDDING_API_TIMEOUT` | 120s | HTTP timeout |
| `LLM_MODEL_NAME` | qwen2.5:7b | Ollama model |
| `LLM_API_BASE_URL` | http://llm:11434 | Ollama host |
| `USE_RERANKING` | 0 (disabled) | Enable cross-encoder |
| `RERANKER_MODEL_NAME` | (default) | Cross-encoder model ID |
| `RERANKER_DEVICE` | auto | cpu / cuda / mps |
| `STREAMLIT_SERVER_PORT` | 8501 | UI port |
| `CRAWLER_YEAR_FILTER` | (none) | Limit to years (e.g., 2024) |

---

## Performance Notes

### Embedding Throughput
- M-series Mac (MPS): 120 chunks/sec
- NVIDIA GPU (CUDA): 300+ chunks/sec
- CPU: 10-20 chunks/sec

### PDF Processing
- Sequential: 1 doc/min
- Parallel (4 workers): 4 docs/min

### Chunk Sizes
- 768 tokens (~3,500 chars) with 15% overlap
- Industry standard: 600–1000 tokens
- Aligns with `nomic-embed-text-v1.5` pre-training

### Retrieval
- Hybrid (semantic + lexical): 70% + 30% weighting
- Reranking: +200–500ms latency, +25–40% accuracy

---

## Common Workflows

### Add New PDF Source
1. Update `crawler_rba.py` → `PUBLICATION_SOURCES`
2. Run `make crawl`
3. Process & embed as usual

### Re-embed After Chunking Changes
```bash
make embeddings-reset
```
(Wipes embeddings, downgrades doc status, re-fills)

### Debug Slow Queries
```bash
make debug
# Then: EXPLAIN ANALYZE <query> in Postgres
```

### Export Feedback for Fine-tuning
```bash
make export-feedback ARGS="--output data/feedback_pairs.jsonl"
make finetune ARGS="--dataset data/feedback_pairs.jsonl --output-dir models/rba-lora-dpo"
```

### Test a Single Module
```bash
make test ARGS="tests/pdf/test_chunker.py -v"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "DATABASE_URL not set" | Copy `.env.example` → `.env` and fill in values |
| Embedding API not ready | Wait ~30s for model download; check logs: `make logs SERVICE=embedding` |
| OOM during embedding | Reduce `EMBEDDING_BATCH_SIZE` in `.env` (e.g., 8) |
| Slow retrieval | Check indexes: `ANALYZE chunks; SELECT pg_size_pretty(pg_relation_size('chunks'));` |
| LLM model not found | Run `make llm-pull MODEL=qwen2.5:7b` |
| PDF parsing fails | Check doc status: `DEBUG_DUMP`, look for status=FAILED |
| Reranking slow | `USE_RERANKING=0` to disable, or increase `RERANKER_BATCH_SIZE` |

---

## Architecture Decisions

**Why pgvector + Postgres?**
- Single database for everything (metadata + vectors + chat)
- HNSW index for fast NN search
- Hybrid retrieval (semantic + lexical) in one system
- No extra services to maintain

**Why 768-token chunks?**
- Matches `nomic-embed-text-v1.5` pre-training window
- Avoids truncation loss in embeddings
- Ollama LLM context (4k–32k) handles it easily
- Industry consensus: 600–1000 tokens

**Why hybrid retrieval?**
- Semantic alone: Misses dates, identifiers, numbers
- Lexical alone: Poor for narrative/concept matching
- Combined: Industry best practice (Pinecone, Weaviate, Cohere)

**Why reranking optional?**
- +200–500ms latency overhead
- Only +25–40% accuracy gain
- Disabled by default for fast iteration
- Enable for production when answer quality is critical

**Why LoRA + DPO?**
- LoRA: Parameter-efficient, fits M-series Mac
- DPO: Preference learning from user feedback (no massive dataset)
- Keep base model frozen (avoids distribution shift)

---

## File Sizes & Complexity

| File | Lines | Purpose |
|------|-------|---------|
| `app/rag/pipeline.py` | 262 | Main RAG orchestration |
| `app/rag/retriever.py` | 180+ | Hybrid search logic |
| `scripts/crawler_rba.py` | 250+ | RBA PDF discovery |
| `scripts/process_pdfs.py` | 280+ | PDF → chunks pipeline |
| `scripts/build_embeddings.py` | 200+ | Embedding backfill |
| `app/pdf/cleaner.py` | 200+ | Header/footer removal |
| `app/ui/streamlit_app.py` | 400+ | Chat UI |
| `docker/embedding/app.py` | 150+ | Embedding service |

---

## External Resources

- **CLAUDE.md** - Hard spec & constraints (read first)
- **LEARN.md** - Technical deep dives
- **PLAN.md** - Roadmap
- **AGENTS.md** - AI agent guidelines
- **README.md** - Quick start

---

Generated: Nov 11, 2025
Version: RBA Document Intelligence Platform v0.1.0
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
