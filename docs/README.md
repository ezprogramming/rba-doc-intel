# Documentation Index

Quick reference for all project documentation.

---

## Core Documentation (Start Here)

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [../README.md](../README.md) | Project overview, setup instructions, quick start | First time setup |
| [../CLAUDE.md](../CLAUDE.md) | Technical specification (constraints, architecture) | Understanding requirements |
| [../LEARN.md](../LEARN.md) | **Comprehensive code walkthrough** (4,500+ lines) | Learning the codebase in detail |

---

## Deep Dive Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [ARCHITECTURE_DECISIONS.md](./ARCHITECTURE_DECISIONS.md) | Why we chose each technology (transformers, PyMuPDF, etc.) | Developers, architects |
| [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md) | **Complete embedding guide**: transformers vs sentence-transformers, performance, scaling | ML engineers, DevOps |
| [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) | Multi-level parallelism, batching strategies, production tuning | Production optimization |
| [TABLE_VERIFICATION.md](./TABLE_VERIFICATION.md) | Table extraction quality, verification, and markdown formatting for RAG | Data engineers, QA |
| [TESTING.md](./TESTING.md) | Testing strategy, test structure, how to run tests | QA, developers |

---

## Quick Answers

### "How do I get started?"

**Read**: [../README.md](../README.md)

**TL;DR**: `make bootstrap && make up && make up-models && make refresh`

---

### "Why transformers instead of sentence-transformers?"

**Read**: [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md) → Quick Start

**TL;DR**: 2.5x faster, no tensor shape errors, production-grade batching.

---

### "How do I optimize embedding performance?"

**Read**: [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md)

**TL;DR**: Adjust `EMBEDDING_BATCH_SIZE` and `EMBEDDING_PARALLEL_BATCHES` in `.env` (CPU: 4/2, GPU: 32/8).

---

### "Why not use transformers for PDF parsing?"

**Read**: [ARCHITECTURE_DECISIONS.md](./ARCHITECTURE_DECISIONS.md) → Section 2

**TL;DR**: Transformers are for ML (embeddings, text generation). PDFs need format parsers (PyMuPDF).

---

### "How are tables formatted for RAG?"

**Read**: [TABLE_VERIFICATION.md](./TABLE_VERIFICATION.md) → LLM prompt representation

**TL;DR**: Tables auto-convert to markdown in LLM prompts for 25-40% better accuracy. Stored as JSONB, embedded as text, formatted as markdown at query time.

---

### "What changed from the original design?"

**Changed**:
- Embedding service now uses `transformers` directly (not `sentence-transformers`)
- Tables formatted as markdown in LLM prompts (not flattened text)

**Unchanged**:
- PDF processing (PyMuPDF)
- Text chunking (string operations)
- Table extraction (Camelot)
- Database (PostgreSQL + pgvector)
- LLM (Ollama/qwen2.5)
- UI (Streamlit)

**Why**:
- Embeddings: Faster (2.5x), more reliable, production-ready
- Table formatting: Better LLM reasoning (25-40% accuracy improvement on numerical queries)

---

## Learning Path

### 1. Beginner: First Time

1. Read [../README.md](../README.md) - Setup
2. Run `make bootstrap && make up`
3. Browse [../LEARN.md](../LEARN.md) sections 1-3
4. Run `make refresh` and observe logs

### 2. Intermediate: Understanding

1. Read [../LEARN.md](../LEARN.md) completely (comprehensive!)
2. Read [ARCHITECTURE_DECISIONS.md](./ARCHITECTURE_DECISIONS.md)
3. Experiment with `.env` parameters
4. Run `make embeddings` and watch performance

### 3. Advanced: Optimization

1. Read [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md)
2. Read [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md)
3. Tune parallelism for your hardware
4. Consider GPU deployment
5. Review [TESTING.md](./TESTING.md) for quality assurance

---

## Common Tasks

### Add a new data source
1. Duplicate `scripts/crawler_rba.py`
2. Update parser for new HTML structure
3. Add new `source_system` in `app/db/models.py`

### Change embedding model
1. Update `MODEL_ID` in `docker/embedding/app.py`
2. Update `embedding_model_name` in `.env`
3. Rebuild: `docker compose up -d --build embedding`
4. **Important**: Dimension must stay 768 or update DB schema

### Improve retrieval quality
1. Read [../LEARN.md](../LEARN.md) section 7.1 (Retriever)
2. Tune `alpha` (vector vs full-text weight)
3. Enable reranking (`USE_RERANKING=1` in `.env`)
4. Adjust `top_k` and `rerank_k` parameters

### Add GPU support
1. Update `docker-compose.yml`:
   ```yaml
   embedding:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```
2. Rebuild: `docker compose up -d --build embedding`
3. Increase batch sizes in `.env`:
   ```bash
   EMBEDDING_BATCH_SIZE=32
   EMBEDDING_PARALLEL_BATCHES=8
   ```

---

## Troubleshooting

### Tensor shape mismatch errors
**Solution**: Already fixed! Using `transformers` with proper padding.

**Details**: [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md) → Why transformers Over sentence-transformers

### Timeout errors during embedding
**Solution**: Reduce `EMBEDDING_BATCH_SIZE` in `.env`.

**Details**: [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) → Pitfalls section

### Low embedding throughput
**Solution**: Increase parallelism or add GPU.

**Details**: [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) → Performance Tuning

---

## Historical Documentation (Archive)

For reference only - content has been integrated into current docs:

- `archive/EXPLORATION_SUMMARY.md` - Original codebase exploration notes
- `archive/IMPROVEMENTS_SUMMARY.md` - Historical changelog
- `archive/COMPLETE_PDF_RAG_INTERVIEW_GUIDE.md` - Interview preparation guide

---

## Contributing

When adding new features:

1. **Update LEARN.md** with code explanations (primary learning resource)
2. **Document architectural decisions** in ARCHITECTURE_DECISIONS.md (if introducing new tech)
3. **Add performance notes** if relevant (PARALLEL_PROCESSING.md)
4. **Test thoroughly** (`make test`)
5. **Update this README** if adding new docs

---

## Questions?

Can't find what you're looking for?

1. Search all docs: `grep -r "your query" docs/`
2. Check [../LEARN.md](../LEARN.md) table of contents (most comprehensive!)
3. Read inline code comments in relevant files
4. Open an issue with `[docs]` tag

---

## Document Hierarchy (Simplified!)

```
docs/
├── README.md (this file) ← Start here for navigation
├── ARCHITECTURE_DECISIONS.md ← WHY each technology
├── EMBEDDING_SERVICE_ARCHITECTURE.md ← HOW embeddings work (complete guide)
├── PARALLEL_PROCESSING.md ← Production tuning
├── TESTING.md ← Testing guide
└── archive/ ← Historical docs (reference only)
    ├── EXPLORATION_SUMMARY.md
    ├── IMPROVEMENTS_SUMMARY.md
    └── COMPLETE_PDF_RAG_INTERVIEW_GUIDE.md

../
├── README.md ← Quick start & setup
├── CLAUDE.md ← Technical spec (constraints)
└── LEARN.md ← **PRIMARY LEARNING RESOURCE** (4,500+ lines, comprehensive!)
```

**Rule of thumb**:
- **Setup**: Read ../README.md
- **Learning**: Read ../LEARN.md (this is THE comprehensive guide!)
- **Why**: Read ARCHITECTURE_DECISIONS.md
- **How (Embeddings)**: Read EMBEDDING_SERVICE_ARCHITECTURE.md
- **Optimize**: Read PARALLEL_PROCESSING.md
- **Test**: Read TESTING.md

---

## What Changed (Documentation Cleanup - 2024-11-23)

**Consolidated & Removed**:
- ❌ `QUICK_REFERENCE.md` → Info merged into README.md and LEARN.md
- ❌ `CODEBASE_STRUCTURE.md` → Redundant with comprehensive LEARN.md
- ❌ `FAST_EMBEDDING_SOLUTION.md` → Merged into EMBEDDING_SERVICE_ARCHITECTURE.md

**Archived** (moved to `archive/`):
- 📦 `EXPLORATION_SUMMARY.md` → Historical exploration notes
- 📦 `IMPROVEMENTS_SUMMARY.md` → Historical changelog
- 📦 `COMPLETE_PDF_RAG_INTERVIEW_GUIDE.md` → Interview prep guide

**Result**: **11 docs → 5 core docs** + 3 archived. Much cleaner!

**Core Docs (Keep)**:
1. README.md (this file) - Navigation
2. ARCHITECTURE_DECISIONS.md - Why decisions
3. EMBEDDING_SERVICE_ARCHITECTURE.md - Complete embedding guide
4. PARALLEL_PROCESSING.md - Production tuning
5. TESTING.md - Testing guide

All information preserved, just better organized!
