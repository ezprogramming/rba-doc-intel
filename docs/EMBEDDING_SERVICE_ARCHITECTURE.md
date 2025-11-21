# Embedding Service Architecture

## Overview

The embedding service converts text chunks into 768-dimensional vectors for semantic search. This document explains the architecture, why we use `transformers` directly instead of `sentence-transformers`, and the performance implications.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Client (scripts/build_embeddings.py)                        │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │ Worker Thread 1│  │ Worker Thread 2│  ... (4 threads)   │
│  │ Batch: 16 cks  │  │ Batch: 16 cks  │                    │
│  └────────┬───────┘  └────────┬───────┘                    │
│           │ HTTP POST          │ HTTP POST                  │
└───────────┼────────────────────┼────────────────────────────┘
            │                    │
            ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Embedding Service (docker/embedding/app.py)                │
│                                                              │
│  FastAPI receives batch: ["text1", "text2", ..., "text16"] │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Tokenization (with padding)                       │  │
│  │    tokenizer(texts, padding='longest', ...)          │  │
│  │    → uniform tensor: [16, max_len]                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. Model forward pass                                │  │
│  │    model(**encoded_input)                            │  │
│  │    → token embeddings: [16, max_len, 768]           │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. Mean pooling                                      │  │
│  │    Average token embeddings (ignore padding)         │  │
│  │    → sentence embeddings: [16, 768]                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. L2 Normalization                                  │  │
│  │    F.normalize(embeddings, p=2, dim=1)               │  │
│  │    → normalized vectors for cosine similarity        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Return: 16 vectors (768-dim each)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Why `transformers` Over `sentence-transformers`?

### The Problem with sentence-transformers

```python
# sentence-transformers approach (DOESN'T WORK RELIABLY)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
vectors = model.encode(texts, batch_size=16)

# ❌ Issues:
# 1. Tensor shape mismatch with variable-length texts
# 2. Black box - can't control padding/truncation
# 3. Slower with workarounds (process one-by-one)
```

**Error we encountered:**
```
The size of tensor a (860) must match the size of tensor b (846) at non-singleton dimension 1
```

This happens because `sentence-transformers` doesn't properly pad variable-length sequences in batches.

### The Solution: transformers Library

```python
# transformers approach (WORKS PERFECTLY)
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 1. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5")

# 2. Tokenize with explicit padding
encoded_input = tokenizer(
    texts,
    padding='longest',      # Pad to longest in batch
    truncation=True,        # Truncate if > max_length
    max_length=8192,
    return_tensors='pt'
)

# 3. Generate embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# 4. Mean pooling
embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# 5. Normalize
embeddings = F.normalize(embeddings, p=2, dim=1)
```

**Benefits:**
- ✅ **2.5x faster** (3.6s vs 9s per chunk)
- ✅ **No tensor shape errors** (proper padding)
- ✅ **Explicit control** (can tune every parameter)
- ✅ **Production-ready** (used by major APIs)

---

## Mean Pooling Explained

**Problem**: Model outputs token-level embeddings (one vector per token).

**Solution**: Average all token embeddings to get one vector per text.

```python
def mean_pooling(model_output, attention_mask):
    """Average token embeddings, ignoring padding tokens.

    Args:
        model_output: Model output with shape [batch, seq_len, hidden_dim]
        attention_mask: Binary mask [batch, seq_len] (1=real token, 0=padding)

    Returns:
        Averaged embeddings with shape [batch, hidden_dim]
    """
    # Extract token embeddings (first element of output tuple)
    token_embeddings = model_output[0]  # Shape: [batch, seq_len, 768]

    # Expand mask to match embedding dimensions
    # From [batch, seq_len] to [batch, seq_len, 768]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Sum embeddings where mask=1 (real tokens only)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)  # [batch, 768]

    # Count real tokens per sequence
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # [batch, 768]

    # Average: sum / count
    return sum_embeddings / sum_mask
```

**Example**:
```python
# Text: "The quick brown fox"
# Tokens: [CLS] The quick brown fox [SEP] [PAD] [PAD]
# Token embeddings: 8 vectors (including padding)
# Mean pooling: Average first 6, ignore last 2 → 1 vector
```

---

## Padding Strategies

### padding='longest' (What We Use)

Pads all sequences in batch to the length of the longest sequence.

```python
# Batch: ["Short", "Medium text here", "Very long text goes here"]
# Token counts: [2, 4, 6]
# After padding: [2+4PAD, 4+2PAD, 6+0PAD] → All length 6

# Efficient! Only pads as much as needed for THIS batch
```

### padding='max_length' (DON'T Use)

Pads all sequences to the maximum model length (8192 tokens).

```python
# Batch: ["Short", "Medium text here", "Very long text goes here"]
# Token counts: [2, 4, 6]
# After padding: [2+8190PAD, 4+8188PAD, 6+8186PAD] → All length 8192

# Wasteful! Adds 8000+ padding tokens for short texts
```

### padding=False (BREAKS)

No padding - causes tensor shape mismatch.

```python
# Batch: ["Short", "Medium text here"]
# Token counts: [2, 4]
# Tensors: [batch, 2] and [batch, 4] → ❌ CAN'T STACK!
```

---

## L2 Normalization

**Purpose**: Enable cosine similarity for vector search.

```python
# Before normalization
vec1 = [0.5, 0.3, 0.8]  # Length: √(0.5² + 0.3² + 0.8²) = 1.02
vec2 = [1.0, 0.6, 1.6]  # Length: √(1.0² + 0.6² + 1.6²) = 2.03

# After L2 normalization (divide by length)
vec1_norm = [0.49, 0.29, 0.78]  # Length = 1.0
vec2_norm = [0.49, 0.29, 0.78]  # Length = 1.0

# Cosine similarity = dot product (when normalized)
similarity = vec1_norm · vec2_norm = 1.0  # These are actually the same!
```

**Why normalize?**
- Cosine similarity = dot product of normalized vectors
- PostgreSQL `<->` operator (L2 distance) becomes equivalent to cosine with normalization
- Faster computation (no need to calculate vector lengths)

---

## Performance Comparison

### Before (sentence-transformers, one-by-one)
```
Server Code:
  for text in texts:
      vec = model.encode([text])  # One at a time
      vectors.append(vec)

Performance:
  - Time per chunk: 9.0s
  - Throughput: 0.11 chunks/sec
  - 2,587 chunks: ~6.5 hours
```

### After (transformers, batched)
```
Server Code:
  # Process all texts in one batch
  encoded = tokenizer(texts, padding='longest', ...)
  embeddings = model(**encoded)

Performance:
  - Time per chunk: 3.6s
  - Throughput: 0.28 chunks/sec
  - 2,587 chunks: ~2.6 hours

✅ 2.5x speedup
```

---

## Configuration

### Environment Variables (.env)

```bash
# Client-side batching
EMBEDDING_BATCH_SIZE=16          # Chunks per HTTP request
EMBEDDING_PARALLEL_BATCHES=4     # Concurrent requests

# Total parallelism: 16 × 4 = 64 chunks in flight
```

### Docker Configuration (docker-compose.yml)

```yaml
embedding:
  environment:
    MODEL_ID: nomic-ai/nomic-embed-text-v1.5
    EMBEDDING_BATCH_SIZE: 16  # (Ignored, kept for documentation)
```

Note: The `EMBEDDING_BATCH_SIZE` in docker-compose.yml is overridden by the actual batch size received from HTTP requests.

---

## When to Use sentence-transformers vs transformers

### Use sentence-transformers When:
- ✅ Simple use case (single text at a time)
- ✅ Fixed-length inputs
- ✅ Prototyping/research
- ✅ You want high-level abstractions

### Use transformers When:
- ✅ Production system with batching
- ✅ Variable-length inputs
- ✅ Need explicit control (padding, truncation)
- ✅ Performance is critical
- ✅ Building an API service

**Our case**: Production RAG system with variable-length chunks → **transformers**

---

## Scaling Options

### Vertical Scaling (GPU)

```yaml
# docker-compose.yml
embedding:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Expected speedup:
- CPU: 0.28 chunks/sec
- GPU (T4): 5-10 chunks/sec (20-35x faster)
- GPU (A100): 20-50 chunks/sec (70-180x faster)

### Horizontal Scaling (Multiple Servers)

```yaml
# docker-compose.yml
embedding:
  replicas: 4  # 4 containers behind load balancer
```

Expected throughput:
- 1 server: 0.28 chunks/sec
- 4 servers: 1.12 chunks/sec (linear scaling)
- 8 servers: 2.24 chunks/sec

---

## Summary

| Aspect | sentence-transformers | transformers |
|--------|----------------------|--------------|
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Performance | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Control | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Production-ready | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Variable-length batching | ❌ | ✅ |

**Decision**: Use **transformers** for production embedding services.

---

## References

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Nomic Embed v1.5 Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Mean Pooling Strategy](https://www.sbert.net/examples/applications/computing-embeddings/README.html)
- [FAST_EMBEDDING_SOLUTION.md](./FAST_EMBEDDING_SOLUTION.md) - Implementation details
- [PARALLEL_PROCESSING.md](./PARALLEL_PROCESSING.md) - Parallelism architecture
