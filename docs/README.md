# Documentation Index

Quick reference for all project documentation.

---

## Getting Started

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [../README.md](../README.md) | Project overview, setup instructions | First time setup |
| [../CLAUDE.md](../CLAUDE.md) | Technical specification (constraints, architecture) | Understanding requirements |
| [../LEARN.md](../LEARN.md) | Complete code walkthrough, line-by-line explanations | Learning the codebase |

---

## Architecture & Design

| Document | Purpose | Audience |
|----------|---------|----------|
| [ARCHITECTURE_DECISIONS.md](./ARCHITECTURE_DECISIONS.md) | Why we chose each technology (transformers, PyMuPDF, etc.) | Developers, architects |
| [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md) | Deep dive: embedding service design with transformers | ML engineers |
| [FAST_EMBEDDING_SOLUTION.md](./FAST_EMBEDDING_SOLUTION.md) | Performance comparison: sentence-transformers vs transformers | Performance tuning |
| [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) | Multi-level parallelism, batching strategies | Production optimization |

---

## Quick Answers

### "Why transformers instead of sentence-transformers?"

**Read**: [FAST_EMBEDDING_SOLUTION.md](./FAST_EMBEDDING_SOLUTION.md)

**TL;DR**: 2.5x faster, no tensor shape errors, production-grade.

---

### "How do I optimize embedding performance?"

**Read**: [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md)

**TL;DR**: Adjust `EMBEDDING_BATCH_SIZE` and `EMBEDDING_PARALLEL_BATCHES` in `.env`.

---

### "Why not use transformers for PDF parsing?"

**Read**: [ARCHITECTURE_DECISIONS.md](./ARCHITECTURE_DECISIONS.md) → Section 2

**TL;DR**: Transformers are for ML (embeddings, text generation). PDFs need format parsers (PyMuPDF).

---

### "How does mean pooling work?"

**Read**: [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md) → Mean Pooling section

**TL;DR**: Averages token embeddings to get fixed-size sentence vectors.

---

### "What changed from the original design?"

**Changed**: Embedding service now uses `transformers` directly (not `sentence-transformers`)

**Unchanged**:
- PDF processing (still PyMuPDF)
- Text chunking (still string operations)
- Table extraction (still Camelot)
- Database (still PostgreSQL + pgvector)
- LLM (still Ollama/qwen2.5)
- UI (still Streamlit)

**Why**: Faster (2.5x), more reliable, production-ready.

---

## Learning Path

### 1. Beginner: First Time

1. Read [../README.md](../README.md) - Setup
2. Run `make bootstrap && make up`
3. Browse [../LEARN.md](../LEARN.md) sections 1-3
4. Run `make crawl` and observe logs

### 2. Intermediate: Understanding

1. Read [../LEARN.md](../LEARN.md) completely
2. Read [ARCHITECTURE_DECISIONS.md](./ARCHITECTURE_DECISIONS.md)
3. Experiment with `.env` parameters
4. Run `make embeddings` and watch performance

### 3. Advanced: Optimization

1. Read [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md)
2. Read [FAST_EMBEDDING_SOLUTION.md](./FAST_EMBEDDING_SOLUTION.md)
3. Read [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md)
4. Tune parallelism for your hardware
5. Consider GPU deployment

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

**Details**: [FAST_EMBEDDING_SOLUTION.md](./FAST_EMBEDDING_SOLUTION.md)

### Timeout errors during embedding
**Solution**: Reduce `EMBEDDING_BATCH_SIZE` in `.env`.

**Details**: [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) → Pitfall 4

### Low embedding throughput
**Solution**: Increase parallelism or add GPU.

**Details**: [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) → Performance Tuning

---

## Contributing

When adding new features:

1. **Update LEARN.md** with code explanations
2. **Document architectural decisions** in ARCHITECTURE_DECISIONS.md
3. **Add performance notes** if relevant (PARALLEL_PROCESSING.md)
4. **Test thoroughly** (`make test`)
5. **Update this README** if adding new docs

---

## Questions?

Can't find what you're looking for?

1. Search all docs: `grep -r "your query" docs/`
2. Check [../LEARN.md](../LEARN.md) table of contents
3. Read inline code comments in relevant files
4. Open an issue with `[docs]` tag

---

## Document Hierarchy

```
docs/
├── README.md (this file) ← Start here
├── ARCHITECTURE_DECISIONS.md ← Why each technology?
├── EMBEDDING_SERVICE_ARCHITECTURE.md ← Deep dive: embeddings
├── FAST_EMBEDDING_SOLUTION.md ← Performance comparison
└── PARALLEL_PROCESSING.md ← Optimization guide

../
├── README.md ← Setup instructions
├── CLAUDE.md ← Technical spec (constraints)
└── LEARN.md ← Complete code walkthrough
```

**Rule of thumb**:
- **Setup**: Read ../README.md
- **Learning**: Read ../LEARN.md
- **Why**: Read docs/ARCHITECTURE_DECISIONS.md
- **How**: Read docs/EMBEDDING_SERVICE_ARCHITECTURE.md
- **Optimize**: Read docs/PARALLEL_PROCESSING.md
