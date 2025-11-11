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

---

## 14. Additional Documentation

For comprehensive line-by-line code explanations and learning the complete project implementation, see **`LEARN.md`**. That document provides detailed walkthroughs of every module, explaining WHY each decision was made, HOW the code works, and WHAT alternatives were considered.
