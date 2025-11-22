# RBA Document Intelligence Platform - Complete Codebase Structure

## Project Overview

The **RBA Document Intelligence Platform** is a production-style Python application that crawls, processes, and provides RAG-based search over Reserve Bank of Australia PDF publications. It combines PDF processing pipelines, vector embeddings, hybrid retrieval, and an interactive Streamlit UI.

**Tech Stack:**
- Language: Python 3.11+
- Package Manager: `uv`
- Database: PostgreSQL + pgvector
- Storage: MinIO (S3-compatible)
- Embeddings: Hugging Face `nomic-embed-text-v1.5` (768-dim)
- LLM: Ollama (local) with configurable models (default: `qwen2.5:1.5b`)
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
│   │   ├── table_extractor.py   # Camelot lattice+stream extraction + metadata
│   │   └── chart_extractor.py   # Heuristic chart/image detector (PyMuPDF)
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
│   ├── extract_tables.py        # Camelot table extraction + table chunks
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
│   ├── QUICK_REFERENCE.md       # Quick reference guide
│   ├── CODEBASE_STRUCTURE.md    # This file (complete structure)
│   ├── COMPLETE_PDF_RAG_INTERVIEW_GUIDE.md
│   ├── IMPROVEMENTS_SUMMARY.md
│   └── EXPLORATION_SUMMARY.md
│
├── pyproject.toml               # Dependencies & project config (uv)
├── docker-compose.yml           # Full stack orchestration
├── Dockerfile                   # App container (Python 3.11 + uv)
├── Makefile                     # **Primary command interface** (make help)
├── .env.example                 # Environment template
├── README.md                     # Quick start & usage guide
├── CLAUDE.md                     # Hard constraints & spec
├── LEARN.md                      # **Comprehensive line-by-line code learning guide (4,500+ lines)**
├── PLAN.md                       # Implementation phases & status
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
- `LLM_MODEL_NAME` - Ollama model (default: `qwen2.5:1.5b`)
- `CRAWLER_YEAR_FILTER` - Optional year filter (e.g., `2024`)

### 2. **Database Layer (`app/db/`)**

#### Models (`models.py`):

| Model | Purpose | Key Fields |
|-------|---------|-----------|
| **Document** | High-level PDF record | id (UUID), source_system, s3_key, doc_type, publication_date, status, content_hash |
| **Page** | Extracted page text | document_id (FK), page_number, raw_text, clean_text |
| **Chunk** | RAG text segments | document_id (FK), text, embedding (VECTOR 768), section_hint, page_start/end, table_id?, chart_id? |
| **ChatSession** | User conversation | id (UUID), created_at, metadata_json |
| **ChatMessage** | Turn in conversation | session_id (FK), role, content, metadata_json |
| **Feedback** | User ratings | chat_message_id (FK), score (1/-1), comment, tags |
| **EvalExample** | Test queries | query, gold_answer, difficulty, category |
| **EvalRun** | Eval session | config, status, summary_metrics |
| **EvalResult** | Result per query | eval_run_id (FK), eval_example_id (FK), llm_answer, scores |
| **Table** | Extracted tables | document_id (FK), page_number, structured_data (JSONB) — also flattened into retrieval chunks |
| **Chart** | Chart/large-image metadata | document_id (FK), page_number, image_metadata (JSONB), bbox?, s3_key? |

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
Paragraph-aware, section-aware splitting with sentence overlap:

- **Strategy:** Split on paragraph → sentence → word boundaries; avoid mid-paragraph breaks via `_find_paragraph_boundary`.
- **Max chunk size:** 768 tokens (~3,500 chars) with a 20% tolerance; hard splits tighten the boundary window.
- **Overlap:** Last two sentences are reused for continuity (better than word overlap).
- **Section hints:** Detects RBA headings (Inflation, Labour Market, numbered sections) before falling back to generic hints.
- **Table-aware marker:** Detects table-like text to avoid mid-table splits (boundary tuning can be extended).

```python
def chunk_pages(
    clean_pages: List[str],
    max_tokens: int = 768,
    overlap_pct: float = 0.15
) -> List[Chunk]:
```

#### Table Extractor (`table_extractor.py`)
Camelot-based table detection (lattice + stream) with quality filters:

- Tries lattice first, falls back to stream if no tables are found.
- Detects header rows automatically and rejects text-only false positives (requires numeric content).
- Returns accuracy + bbox + structured rows for each table.
- Used by `scripts/extract_tables.py` to persist structured rows to `tables` **and flatten table content into chunk text** so it can be embedded and retrieved alongside prose. Failures are logged and skipped (best-effort).

#### Chart Extractor (`chart_extractor.py`)
Heuristic detector for large images/charts:
- Flags images larger than 200x150px and records page number, bounding box, dimensions, and format.
- Optional image extraction for future MinIO storage.
- Backed by the `charts` table and `chunks.chart_id` for future multimodal retrieval.

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
- Configurable model (default: `qwen2.5:1.5b`)
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
2. Retrieve top-k chunks (hybrid) — includes flattened tables
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
- `--batch-size`: Documents per fetch (defaults to `PDF_BATCH_SIZE`, 16 in `.env.example`)
- `--workers`: Thread workers (defaults to `PDF_MAX_WORKERS`, 2 in `.env.example`)

### `extract_tables.py`
**Table Extraction as a Separate Stage**

- Runs Camelot (lattice + stream) on processed PDFs.
- Persists structured rows to `tables` plus flattened table chunks.
- Downgrades doc status from `EMBEDDED` to `CHUNKS_BUILT` when new chunks are added (so embeddings can be regenerated).
- Supports sequential mode or a ProcessPool (`--workers`, defaults to `TABLE_MAX_WORKERS`, 4 in `.env.example`).
- Respects `--batch-size` (defaults to `TABLE_BATCH_SIZE`, 16 in `.env.example`).
- Optional `--document-id` targeting and `--force` re-extraction.

### `build_embeddings.py`
**Embedding Backfill with Parallel Batches**

**Performance:**
- Sequential: ~50 chunks/sec (CPU baseline)
- Parallel batches configurable via `.env` (`EMBEDDING_PARALLEL_BATCHES`, default 2)
- GPU (NVIDIA): >2,000 chunks/sec depending on model/device

**Flow:**
1. Find chunks where `embedding IS NULL`
2. Submit multiple batch jobs in parallel
3. Each batch calls embedding API
4. Update `chunks.embedding` column
5. Mark document as `EMBEDDED`
- Exponential backoff with abort after consecutive failures (default 5) if batches keep erroring (e.g., embedding API down)

**Args:**
- `--batch-size`: Chunks per batch (defaults to `EMBEDDING_BATCH_SIZE`, currently 8 via `.env`)
- `--parallel`: Concurrent batches (defaults to `EMBEDDING_PARALLEL_BATCHES`, currently 2 via `.env`)
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
MINIO_ACCESS_KEY=YOUR_MINIO_ACCESS_KEY
MINIO_SECRET_KEY=YOUR_MINIO_SECRET_KEY
MINIO_BUCKET_RAW_PDF=YOUR_RAW_BUCKET
MINIO_BUCKET_DERIVED=YOUR_DERIVED_BUCKET

# Embeddings
EMBEDDING_API_BASE_URL=http://embedding:8000
EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1.5
EMBEDDING_BATCH_SIZE=8
EMBEDDING_PARALLEL_BATCHES=2
EMBEDDING_API_TIMEOUT=240

# PDF/Table processing
PDF_BATCH_SIZE=16
PDF_MAX_WORKERS=2
TABLE_BATCH_SIZE=16
TABLE_MAX_WORKERS=4

# LLM
LLM_MODEL_NAME=qwen2.5:1.5b
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
make llm-pull MODEL=qwen2.5:1.5b
```

### 3. **Ingest Corpus**
```bash
make crawl
make process
make tables
make embeddings ARGS="--batch-size 8 --parallel 2"
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

---

## For Detailed Code Explanations

This document provides the **structural overview** of the codebase. For comprehensive **line-by-line code explanations** with WHY/HOW/WHAT decisions, examples, and common pitfalls, see:

**`LEARN.md`** (4,500+ lines covering all 12 sections)

Sections include:
1. Project Overview & Architecture
2. Setup & Configuration (Makefile)
3. Database Layer
4. Storage Layer
5. PDF Processing Pipeline
6. Embedding Generation
7. RAG Retrieval System
8. LLM Integration
9. User Interface (Streamlit)
10. ML Engineering Features
11. Scripts & CLI Tools
12. Testing & Quality

---

Last Updated: 2024-11-11
