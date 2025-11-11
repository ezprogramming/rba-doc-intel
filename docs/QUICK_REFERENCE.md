# RBA Document Intelligence Platform - Quick Reference Guide

## Project at a Glance

**What is it?** A production-style RAG system that crawls RBA PDFs, processes them intelligently, and exposes a chat UI for searching the knowledge base.

**Tech Stack:** Python 3.11 + uv + PostgreSQL + MinIO + Ollama + Streamlit

**Key Features:**
- Parallel PDF processing (1 doc → 4 docs/min)
- Hybrid semantic+lexical retrieval (pgvector HNSW + Postgres FTS)
- Token-by-token LLM streaming with live updates
- User feedback loop (thumbs up/down → DPO training)
- Optional cross-encoder reranking (+25-40% accuracy)
- Safety guardrails (PII, prompt injection, toxicity detection)
- Hook bus for observability (rag:*, ui:* events)
- Evaluation framework with golden examples
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
| `docs/` | Deep dives (interview guide, improvements, quick reference) |
| `docker-compose.yml` | Full stack orchestration |
| `CLAUDE.md` | Hard spec & constraints |
| `LEARN.md` | **Comprehensive line-by-line code explanations (4,500+ lines)** |
| `Makefile` | **Primary command interface for all operations** |

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
- **LEARN.md** - **Comprehensive line-by-line code learning guide** (4,500+ lines covering every module)
- **Makefile** - Primary command interface (`make help` for all targets)
- **PLAN.md** - Implementation phases & status
- **AGENTS.md** - AI agent guidelines
- **README.md** - Quick start guide

---

**For detailed code explanations:** See `LEARN.md` - every module explained with WHY/HOW/WHAT, examples, and common pitfalls.

---

Last Updated: 2024-11-11
Version: RBA Document Intelligence Platform v0.1.0
