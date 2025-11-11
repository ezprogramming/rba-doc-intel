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

Because these run via the official Postgres entrypoint, `docker compose up` on a clean volume always yields the same schema without invoking Python migrations.

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

**Tip:** use `docker compose up embedding llm` to keep model containers warm during development.

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
docker compose run --rm app uv run pytest
```

Under heavy development you can run targeted suites:

```bash
docker compose run --rm app uv run pytest tests/ui/test_feedback.py
```

This keeps the feedback subsystem verified even if Postgres/MinIO aren’t seeded with data.

---

## Full Data Flow (Cheat Sheet)

1. **Ingestion:** `crawler_rba.py` → `process_pdfs.py` → `build_embeddings.py`.
2. **Query:** Streamlit form → `answer_query()` → hybrid retrieval → LLM streaming → DB persistence.
3. **Feedback:** User clicks thumb → `store_feedback()` → `feedback` table (used later for evaluation/fine-tuning).

Keep this doc open alongside the code when stepping through the system; each section points you to the exact file/line block that implements the described behavior.
