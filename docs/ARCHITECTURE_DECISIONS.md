# Architecture Decisions & Rationale

This document explains key architectural choices in the RBA Document Intelligence Platform, with focus on **why** certain technologies were chosen and **where** they apply.

---

## 1. Embedding Service: transformers vs sentence-transformers

### Decision
Use **`transformers` library directly** instead of `sentence-transformers` for the embedding service.

### Where It Applies
**ONLY** the embedding service (`docker/embedding/app.py`).

**Does NOT apply to**:
- ❌ PDF processing
- ❌ Text chunking
- ❌ Table extraction
- ❌ Any other part of the pipeline

### Why transformers?

| Aspect | sentence-transformers | transformers |
|--------|----------------------|--------------|
| **Performance** | 9s per chunk (one-by-one) | 3.6s per chunk (batched) ✅ |
| **Variable-length batching** | ❌ Tensor shape mismatch | ✅ Proper padding |
| **Control** | Black box | Explicit control ✅ |
| **Production readiness** | Research/prototype | Production APIs ✅ |

**Technical reason**: `sentence-transformers` doesn't properly pad variable-length sequences in batches, causing tensor shape mismatches like:
```
The size of tensor a (860) must match the size of tensor b (846)
```

**Solution**: `transformers` gives explicit control over tokenization:
```python
tokenizer(texts, padding='longest', truncation=True, max_length=8192)
```

This ensures all tensors have uniform shape within a batch.

### Performance Impact
- **2.5x faster** embedding generation
- CPU: 0.28 chunks/sec (vs 0.11 chunks/sec)
- Total corpus time: 2.6 hours (vs 6.5 hours)

### References
- [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md) - Complete technical details
- [FAST_EMBEDDING_SOLUTION.md](./FAST_EMBEDDING_SOLUTION.md) - Performance comparison

---

## 2. PDF Processing: PyMuPDF (not transformers)

### Decision
Use **PyMuPDF (`fitz`)** for PDF text extraction.

### Why NOT transformers?
Transformers are ML models for:
- Text embeddings
- Text generation
- Classification
- NER, etc.

They **cannot**:
- ❌ Parse binary PDF files
- ❌ Extract text from PDF pages
- ❌ Handle PDF structure (pages, fonts, images)

### Why PyMuPDF?
- ✅ Fast C++ implementation
- ✅ Reliable Unicode support
- ✅ Handles complex PDFs (multi-column, tables, images)
- ✅ Good API for page-by-page processing
- ✅ Memory-efficient streaming

### Alternative Considered
**pdfplumber**: Better for table extraction but slower for text.

**Our choice**: PyMuPDF for text, Camelot for tables (specialized tool).

---

## 3. Text Chunking: Simple String Operations (not transformers)

### Decision
Use **paragraph-aware recursive splitting** with Python string operations.

### Why NOT transformers?
Chunking is about:
- Splitting text at paragraph/sentence boundaries
- Controlling chunk size (tokens/chars)
- Creating overlaps for context

This is **pure string manipulation**, not ML.

### Our Implementation (`app/pdf/chunker.py`)
```python
def chunk_pages(clean_pages, max_tokens=768, overlap_pct=0.15):
    # 1. Concatenate pages
    full_text = " ".join(clean_pages)

    # 2. Find paragraph boundaries
    boundary = text.find('\n\n', target_pos)

    # 3. Split at boundaries
    chunk_text = full_text[start:end]

    # 4. Add sentence-based overlap
    overlap = get_sentence_overlap(chunk_text, num_sentences=2)
```

**Key features**:
- ✅ Paragraph-aware (preserves semantic units)
- ✅ Sentence-based overlap (maintains context)
- ✅ Table-aware (detects table markers)
- ✅ Fast (no ML inference needed)

---

## 4. Table Extraction: Camelot (not transformers)

### Decision
Use **Camelot** for structured table extraction from PDFs.

### Why NOT transformers?
Table extraction from PDFs requires:
- PDF rendering (detect lines, borders)
- Geometric analysis (find table boundaries)
- Cell detection (intersection of lines)
- Text alignment (assign text to cells)

This is **computer vision on PDF graphics**, not NLP.

### Why Camelot?
- ✅ Specialized for PDF table extraction
- ✅ Two methods: lattice (bordered tables) + stream (borderless)
- ✅ Returns structured data (rows, columns)
- ✅ Accuracy scores (confidence metrics)

### Our Implementation
```python
# Try lattice first (bordered tables)
tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')

# Fallback to stream (borderless tables)
if not tables:
    tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')

# Extract structured data
for table in tables:
    structured_data = table.df.to_dict('records')  # List of row dicts
```

**Why this approach?**
- Lattice: Fast, accurate for bordered tables
- Stream: Slower but handles borderless tables
- Both return structured data (not just text)

---

## 5. When to Use ML vs Traditional Methods

### Use ML (transformers, LLMs) When:
- ✅ Need semantic understanding (embeddings, similarity)
- ✅ Generate natural language (LLM responses)
- ✅ Classification/NER (document type, entities)
- ✅ Complex reasoning (RAG query answering)

### Use Traditional Methods When:
- ✅ Parsing structured formats (PDF, JSON, XML)
- ✅ String manipulation (chunking, cleaning)
- ✅ Deterministic operations (regex, boundary detection)
- ✅ Performance-critical paths (no inference overhead)

### Our Choices

| Task | Method | Reason |
|------|--------|--------|
| **Embed text** | transformers (ML) | Need semantic vectors |
| **Parse PDF** | PyMuPDF (traditional) | Structured format parsing |
| **Chunk text** | String ops (traditional) | Deterministic splitting |
| **Extract tables** | Camelot (computer vision) | Geometric analysis |
| **Search chunks** | pgvector (ML + traditional) | Hybrid: vectors + full-text |
| **Generate answers** | LLM (ML) | Natural language generation |

---

## 6. Database: PostgreSQL + pgvector (not specialized vector DB)

### Decision
Use **PostgreSQL with pgvector extension** instead of specialized vector databases (Qdrant, Milvus, Weaviate).

### Why?
- ✅ **Single source of truth**: Metadata + vectors + chat logs in one place
- ✅ **ACID transactions**: Consistency between chunks and embeddings
- ✅ **Mature ecosystem**: Backups, replication, monitoring tools
- ✅ **Cost-effective**: No additional service to maintain
- ✅ **Good enough performance**: HNSW index provides 10-100x speedup

### When to Consider Specialized Vector DB?
- 🔴 **Scale**: >10M vectors (Postgres starts to struggle)
- 🔴 **Complex queries**: Multi-vector search, ANN with filters
- 🔴 **Real-time updates**: High-frequency vector insertions

**Our scale**: ~3K chunks → PostgreSQL is perfect.

---

## 7. LLM: Local (Ollama) vs Cloud (OpenAI)

### Decision
Use **local LLM via Ollama** (qwen2.5:1.5b) for RAG answers.

### Why Local?
- ✅ **Privacy**: No data sent to external APIs
- ✅ **Cost**: No per-token charges
- ✅ **Latency**: No network overhead
- ✅ **Control**: Can fine-tune on feedback

### Tradeoffs
- ❌ **Quality**: Cloud models (GPT-4) are better
- ❌ **Hardware**: CPU inference slower than GPU (optimized for 1.5B model)
- ❌ **Maintenance**: Model updates require manual download

### When to Use Cloud?
- Production deployment with high quality requirements
- No GPU available
- Budget for API costs ($0.01-0.03 per 1K tokens)

**Our choice**: Local for learning/development, easy to swap for production.

---

## 8. Parallelism: Client-side vs Server-side

### Decision
Use **client-side parallelism** (concurrent HTTP requests) instead of server-side model batching.

### Why?
With `transformers` and proper padding, we can now use **both**:

**Client-side** (4 parallel requests):
- ✅ Better CPU utilization (multi-core)
- ✅ Fault isolation (one failure doesn't block others)
- ✅ Load distribution (with multiple servers)

**Server-side** (batch_size=16):
- ✅ GPU efficiency (parallel tensor ops)
- ✅ Reduced HTTP overhead
- ✅ Better throughput per server

**Our config**:
```bash
EMBEDDING_BATCH_SIZE=16          # Server processes 16 at once
EMBEDDING_PARALLEL_BATCHES=4     # Client sends 4 concurrent requests
Total: 16 × 4 = 64 chunks in flight
```

### References
- [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) - Complete parallelism guide

---

## Summary

| Component | Technology | ML or Traditional? | Reason |
|-----------|-----------|-------------------|--------|
| **Embedding** | transformers | 🤖 ML | Need semantic vectors |
| **PDF parsing** | PyMuPDF | 📄 Traditional | Structured format |
| **Chunking** | String ops | 📄 Traditional | Deterministic |
| **Tables** | Camelot | 👁️ Computer Vision | Geometric analysis |
| **Search** | pgvector | 🤖 ML + 📄 Traditional | Hybrid approach |
| **LLM** | Ollama | 🤖 ML | Text generation |

**Key insight**: Use the **simplest tool for the job**. Don't use ML when traditional methods work better.

---

## Further Reading

- [EMBEDDING_SERVICE_ARCHITECTURE.md](./EMBEDDING_SERVICE_ARCHITECTURE.md) - Why transformers
- [FAST_EMBEDDING_SOLUTION.md](./FAST_EMBEDDING_SOLUTION.md) - Performance comparison
- [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) - Parallelism strategies
- [LEARN.md](../LEARN.md) - Complete code walkthrough
