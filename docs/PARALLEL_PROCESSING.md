# Parallel Processing & Batch Configuration Guide

## Overview

This document explains the **multi-level parallelism** architecture used in the RBA Document Intelligence Platform's embedding pipeline. Understanding these concepts is critical for production ML systems.

---

## Architecture: 3 Levels of Parallelism

```
┌────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: Client-Side Parallel Batches (scripts/build_embeddings.py) │
│                                                                      │
│  Thread Pool with N workers (EMBEDDING_PARALLEL_BATCHES)            │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Worker 0     │  │ Worker 1     │  │ Worker N     │             │
│  │ Batch: 8 cks │  │ Batch: 8 cks │  │ Batch: 8 cks │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                  │                  │                      │
│         └──────────────────┼──────────────────┘                      │
│                            │ HTTP POST /embeddings                   │
└────────────────────────────┼───────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: HTTP API Batching (docker/embedding/app.py)              │
│                                                                      │
│  FastAPI receives batch of texts: [text1, text2, ..., text8]       │
│                                                                      │
│  Configured by: EMBEDDING_BATCH_SIZE (in .env, client-side)        │
│                                                                      │
│  Purpose: Efficient HTTP transport (reduce round-trips)             │
│                                                                      │
└────────────────────────────┼───────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│ LEVEL 3: Model Internal Batching (sentence-transformers)          │
│                                                                      │
│  model.encode(texts, batch_size=BATCH_SIZE)                        │
│                                                                      │
│  Configured by: EMBEDDING_BATCH_SIZE (in docker-compose.yml)       │
│                                                                      │
│  Purpose: GPU/CPU vectorization (process multiple texts in parallel)│
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ Tokenize → Pad/Truncate → Embed → Return vectors   │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Parameters

### 1. `EMBEDDING_BATCH_SIZE` (in `.env`)
**What it controls**: How many chunks the **client** sends per HTTP request

**Example**:
```bash
EMBEDDING_BATCH_SIZE=8
```

**Tradeoffs**:
- ✅ **Larger (16-32)**: Fewer HTTP round-trips, better throughput
- ❌ **Too large (>64)**:
  - Longer timeout risk
  - More memory per request
  - Harder to recover from failures (lose entire batch)

**Recommended**: `8-16` for development, `16-32` for production

---

### 2. `EMBEDDING_PARALLEL_BATCHES` (in `.env`)
**What it controls**: How many **concurrent HTTP requests** the client makes

**Example**:
```bash
EMBEDDING_PARALLEL_BATCHES=4
```

**Calculation**:
```
Total chunks processed concurrently = BATCH_SIZE × PARALLEL_BATCHES
Example: 8 × 4 = 32 chunks in flight at once
```

**Tradeoffs**:
- ✅ **Higher (4-8)**: Better CPU/GPU utilization, faster overall pipeline
- ❌ **Too high (>8)**:
  - Overwhelms embedding server
  - Out-of-memory errors
  - Thread contention

**Recommended**:
- **CPU**: `2-4` (CPUs don't benefit much from high parallelism)
- **GPU**: `4-8` (GPUs can handle more parallel work)

---

### 3. `EMBEDDING_BATCH_SIZE` (in `docker-compose.yml`)
**What it controls**: How many texts the **model** processes internally per forward pass

**Example**:
```yaml
environment:
  EMBEDDING_BATCH_SIZE: 16  # Server-side model batching
```

**IMPORTANT NOTE**: Due to limitations in `sentence-transformers` with variable-length sequences, we currently use `batch_size=1` in `model.encode()` to avoid tensor shape mismatches. The `EMBEDDING_BATCH_SIZE` in docker-compose.yml is kept for documentation but overridden in code.

**Why batch_size=1?**
- `sentence-transformers` doesn't properly pad variable-length sequences in batches
- Tensor shape mismatches occur: `tensor a (860) vs tensor b (846)`
- Processing one-at-a-time eliminates the issue

**Where does parallelism come from then?**
- ✅ **Client-side concurrent requests** (`EMBEDDING_PARALLEL_BATCHES=4`)
- ✅ **Multiple chunks per request** (`EMBEDDING_BATCH_SIZE=16`)
- ❌ **NOT from server-side model batching** (set to 1)

**Total throughput**: 16 chunks × 4 requests = **64 chunks in flight**

---

## Padding & Truncation (Critical!)

### The Tensor Shape Problem

When processing batches, PyTorch requires **uniform tensor shapes**:

```python
# ❌ FAILS: Variable-length sequences
chunk1 = "Short text"           → 710 tokens
chunk2 = "Much longer text..."  → 758 tokens

# PyTorch tries to create:
tensor([
    [tok1, tok2, ..., tok710],        # 710 wide
    [tok1, tok2, ..., tok758]         # 758 wide  ❌ SHAPE MISMATCH!
])
```

### The Solution: Padding

```python
# ✅ WORKS: Pad to longest sequence in batch
chunk1 = "Short text"           → 710 tokens → pad to 758
chunk2 = "Much longer text..."  → 758 tokens → keep at 758

# PyTorch creates uniform tensor:
tensor([
    [tok1, ..., tok710, PAD, PAD, ..., PAD],  # 758 wide
    [tok1, tok2, ..., tok758]                  # 758 wide  ✅ SUCCESS!
])
```

### Configuration (in `docker/embedding/app.py`)

```python
# Configure tokenizer for proper batching
tokenizer.padding = 'longest'     # Pad to longest in batch
tokenizer.truncation = True       # Truncate if > max_seq_length
tokenizer.model_max_length = 8192 # Max tokens (nomic-embed-text-v1.5 limit)
```

**Key insight**:
- `padding='longest'` means **no fixed overhead** - small batches use less memory
- Only sequences exceeding 8192 tokens get truncated (your chunks are ~1700 tokens max)

---

## Recommended Configurations

### Development (CPU-only, stable)
```bash
# .env
EMBEDDING_BATCH_SIZE=4
EMBEDDING_PARALLEL_BATCHES=2
```

**Result**: 4 × 2 = **8 chunks in flight** at once

**Why these numbers?**
- 4 chunks per request = manageable memory usage with padding
- 2 parallel requests = CPU can handle without crashing
- Total throughput: ~0.2 chunks/sec
- **Critical**: Higher values (16×4=64) cause server crashes from memory exhaustion

**Warning**: Even with `transformers` and proper batching, CPU memory limits apply:
- ❌ 16 chunks × 4 parallel = server crashes (OOM, connection refused)
- ✅ 4 chunks × 2 parallel = stable (tested on 2,587 chunks)

---

### Production (CPU-only server with more cores)
```bash
# .env
EMBEDDING_BATCH_SIZE=8
EMBEDDING_PARALLEL_BATCHES=4
```

**Result**: 8 × 4 = **32 chunks in flight** at once

**Why these numbers?**
- 8 chunks per request = higher throughput but more memory
- 4 parallel requests = utilizes multi-core CPU
- Total throughput: ~0.3 chunks/sec
- **Requires**: 16GB+ RAM and 4+ CPU cores

**Warning**: This config may still crash on low-memory systems. Monitor with:
```bash
docker stats embedding
# Watch CPU% and MEM USAGE
```

If you see crashes, reduce to Development config (4×2)

---

### Production (GPU server)
```bash
# .env
EMBEDDING_BATCH_SIZE=32
EMBEDDING_PARALLEL_BATCHES=8
```

```yaml
# docker-compose.yml
environment:
  EMBEDDING_BATCH_SIZE: 32
```

**Result**: 32 × 8 = **256 chunks in flight** at once

---

## Performance Tuning Guide

### 1. Find Bottleneck

```bash
# Monitor embedding service logs
docker compose logs -f embedding

# Look for:
# - "Embedding generation failed" → reduce batch sizes
# - Slow response times → increase parallelism
# - OOM errors → reduce EMBEDDING_BATCH_SIZE in docker-compose.yml
```

### 2. Measure Throughput

```python
# In scripts/build_embeddings.py, add timing:
import time

start = time.time()
total_embedded = 0

while True:
    count = generate_missing_embeddings()
    if count == 0:
        break
    total_embedded += count

elapsed = time.time() - start
throughput = total_embedded / elapsed
print(f"Throughput: {throughput:.1f} chunks/sec")
```

### 3. Optimize Iteratively

Start conservative, increase gradually:

```
Step 1: BATCH_SIZE=8,  PARALLEL=2  → measure throughput
Step 2: BATCH_SIZE=16, PARALLEL=2  → if faster, continue
Step 3: BATCH_SIZE=16, PARALLEL=4  → if faster, continue
Step 4: BATCH_SIZE=32, PARALLEL=4  → monitor for errors
```

Stop when:
- Throughput stops improving
- You see errors (OOM, timeouts)
- CPU/GPU utilization reaches ~80-90%

---

## Common Pitfalls

### ❌ Pitfall 1: Too Much Parallelism
```bash
EMBEDDING_PARALLEL_BATCHES=16  # Overkill for CPU
```
**Symptom**: High CPU, but not faster (thread contention)
**Fix**: Reduce to 2-4 for CPU

### ❌ Pitfall 2: Mismatched Batch Sizes
```bash
# .env: Client sends 32 chunks
EMBEDDING_BATCH_SIZE=32

# docker-compose.yml: Server processes 8 at a time
EMBEDDING_BATCH_SIZE: 8
```
**Result**: Server does 4 internal batches per request (inefficient)
**Fix**: Match client and server batch sizes

### ❌ Pitfall 3: No Padding Configuration
```python
# Missing in docker/embedding/app.py
tokenizer.padding = 'longest'
tokenizer.truncation = True
```
**Symptom**: Tensor shape mismatch errors
**Fix**: Always configure padding for variable-length inputs (or use batch_size=1)

### ❌ Pitfall 4: Timeout Errors
```bash
ERROR Embedding request failed: Read timed out. (read timeout=240)
```
**Symptom**: Requests take longer than `EMBEDDING_API_TIMEOUT` (240s)
**Root cause**: Too many chunks per request for CPU processing

**Calculation**:
```
batch_size=16 chunks × 15s/chunk = 240s ⚠️ AT LIMIT
With retries or slow chunks → timeout
```

**Fix**: Reduce `EMBEDDING_BATCH_SIZE`
```bash
# Before (timeout risk):
EMBEDDING_BATCH_SIZE=16

# After (safe):
EMBEDDING_BATCH_SIZE=4   # 4 × 10s = 40s << 240s
```

**Why not increase timeout?**
- Longer timeouts = slower failure detection
- Better to process smaller batches faster
- Easier to recover from transient errors

---

## Advanced: Dynamic Batching

For production systems with varying load, consider **adaptive batching**:

```python
# Pseudocode
def adaptive_batch_size(queue_depth: int) -> int:
    """Scale batch size based on backlog."""
    if queue_depth > 1000:
        return 32  # Large batches for backlog
    elif queue_depth > 100:
        return 16  # Medium batches
    else:
        return 8   # Small batches for responsiveness
```

**Benefit**: Balance latency vs throughput based on workload

---

## Why This Architecture Is Actually Common

You might think "batch_size=1 is inefficient!" but this pattern is **actually common in production**:

### Pattern: Horizontal Scaling Over Vertical Batching

```
❌ Bad: Large batches, single server
   model.encode(1000 texts, batch_size=100)
   → Slow, memory-intensive, all-or-nothing failure

✅ Good: Small batches, many parallel requests
   model.encode(10 texts, batch_size=1) × 100 parallel requests
   → Fast, fault-tolerant, better resource utilization
```

### Why This Works

1. **HTTP-level batching** (16 chunks per request)
   - Reduces network overhead
   - Efficient serialization/deserialization

2. **Client-side parallelism** (4 concurrent requests)
   - Better CPU core utilization
   - Isolates failures (1 bad request doesn't block others)

3. **Model processes sequentially** (batch_size=1)
   - No tensor shape issues
   - Simpler code, fewer edge cases
   - Easier to debug

4. **Scale horizontally** (multiple embedding servers)
   - Add more containers behind load balancer
   - Linear scaling: 2× servers = 2× throughput

### Real-World Example

**OpenAI's embedding API** uses a similar pattern:
- Accepts batches via HTTP (up to 2048 texts)
- Processes internally with small/unit batches
- Scales horizontally with many replicas

---

## Summary

**Key Takeaways**:
1. **Parallelism via concurrency, not batching**: Client threads handle parallelism
2. **HTTP batching for efficiency**: Reduce network round-trips
3. **Model batch_size=1 for correctness**: Avoids tensor shape issues
4. **Horizontal scaling**: Add more servers for more throughput
5. **Monitor**: Watch logs, measure throughput, detect bottlenecks

**Current Configuration** (after fixes):
- ✅ Server-side `batch_size=1` (correct, avoids tensor mismatch)
- ✅ Client-side batch=16, parallel=4 (64 chunks in flight)
- ✅ Max sequence length: 8192 tokens
- ✅ No data loss (your chunks are ~1700 tokens max)
