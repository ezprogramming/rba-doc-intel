# Testing Guide

## End-to-End Workflow Test

The `test_workflow.py` script validates the complete RAG pipeline using **real RBA PDFs** from the crawler.

**Note:** Tests use production tables but with clearly marked test data (`source_system='TEST'`). A separate test schema exists in `test.*` tables for future isolated testing, but the current test script uses production tables for simplicity.

### What It Tests

1. **PDF Crawling** - Uses the real crawler to fetch recent RBA PDFs:
   - Discovers and downloads 1-2 Statement on Monetary Policy (SMP) documents from 2024+
   - Uploads to MinIO and registers in database
   - Uses real RBA data with tables

2. **Ingestion Pipeline** - Verifies `make ingest`:
   - Single-pass text + table extraction
   - Camelot table extraction from real PDFs
   - Chunk creation with proper metadata
   - Table-chunk linking (via `table_id`)

3. **Embedding Generation** - Verifies `make embeddings`:
   - All chunks get embeddings
   - Correct vector dimensions (768 for nomic-embed-text)

4. **RAG Retrieval** - Verifies end-to-end search:
   - Queries like "What is the inflation outlook?"
   - Semantic search returns relevant chunks
   - Table content is searchable and linked
   - Evidence includes structured data when available

### Running the Test

**Prerequisites:**
- Docker services must be running (`make up`, `make up-models`)
- Embedding service must be available
- Database must be initialized

**Run the test:**

```bash
# Simple way
make test-workflow

# Or directly
uv run python scripts/test_workflow.py
```

### Expected Output

```
======================================================================
=== End-to-End Workflow Test with Real RBA PDFs ===
======================================================================

Step 1: Crawling recent RBA PDFs...
Crawling 2 recent RBA PDFs...
Crawled 1 PDFs from Statement on Monetary Policy
✓ Found 2 documents to test

Step 2: Running ingestion (text + tables in one pass)...
  Processing document 12345678-1234-1234-1234-123456789abc...
✓ Document: Statement on Monetary Policy - February 2025 (status: CHUNKS_BUILT)
  Chunks: 127 total, 8 from tables
  Tables: 8 extracted
    Chunk 15 → table 1 (5 rows, accuracy: 95%)
    Chunk 23 → table 2 (7 rows, accuracy: 88%)
    Chunk 45 → table 3 (4 rows, accuracy: 92%)
✓ Ingestion complete for all documents

Step 3: Generating embeddings...
✓ Generated embeddings for 254 chunks

Step 4: Verifying embeddings...
✓ All 127 chunks have embeddings
  Embedding dimension: 768

Step 5: Testing RAG retrieval...

=== Testing RAG Retrieval ===

Query: What is the inflation outlook?
  [1] Score: 0.842, Doc: SMP, Section: 3. Economic Outlook
      Text preview: The Bank's central forecast is for CPI inflation to decline to around 3¼ per cent by mid-2025...
      ✓ Linked to table with 5 rows (accuracy: 95%)
✓ Found expected content for query

Query: What are the GDP forecasts?
  [1] Score: 0.835, Doc: SMP, Section: N/A
      Text preview: GDP Growth — 2024: 2.1%, 2025: 2.5%, 2026: 2.8%...
      ✓ Linked to table with 7 rows (accuracy: 88%)
✓ Found expected content for query

======================================================================
✓✓✓ ALL TESTS PASSED ✓✓✓
======================================================================

Tested 2 real RBA documents:
  • Statement on Monetary Policy - February 2025
    - 127 chunks (8 from tables)
    - 8 tables extracted
  • Statement on Monetary Policy - November 2024
    - 132 chunks (7 from tables)
    - 7 tables extracted

Workflow validation:
  ✓ PDF ingestion (text + tables)
  ✓ Table extraction and linking
  ✓ Embedding generation
  ✓ RAG retrieval with table content

Note: Test documents remain in database for manual inspection.
      Use 'make ingest-reset' to reprocess if needed.
```

### What Success Looks Like

✅ **All 5 steps pass without errors**
✅ **Real RBA PDFs are crawled** (1-2 SMP documents from 2024+)
✅ **Tables are extracted** (typically 5-10 tables per SMP document)
✅ **Table chunks are created** (chunks with `table_id` properly linked)
✅ **Embeddings are generated** (all chunks have 768-dim vectors)
✅ **RAG retrieval works** (queries return relevant chunks with table data)
✅ **Table links work** (can access structured rows via `table_id`)

### Troubleshooting

**Test fails at Step 3 (Ingestion):**
- Check if Camelot dependencies are installed (`ghostscript`, `opencv`)
- Verify PDF is valid (check MinIO upload)

**Test fails at Step 5 (Embeddings):**
- Ensure embedding service is running (`make up-embedding`)
- Check `EMBEDDING_API_BASE_URL` in `.env`

**Test fails at Step 7 (RAG Retrieval):**
- Verify embeddings were generated (Step 6)
- Check if queries are too specific (adjust test queries)

**Table extraction finds 0 tables:**
- Camelot may need different settings
- Check if PDF has actual table structure (not just formatted text)

### Adding More Tests

To test with your own PDF:

```python
# In test_workflow.py, replace generate_test_pdf() with:
def generate_test_pdf() -> bytes:
    with open("path/to/your/test.pdf", "rb") as f:
        return f.read()
```

Or create a separate test script using the same pattern:
1. Upload PDF to MinIO
2. Register in database
3. Run `ingest_document()`
4. Generate embeddings
5. Test retrieval

### Integration with CI/CD

This test can be run in CI pipelines:

```yaml
# .github/workflows/test.yml
- name: Run end-to-end workflow test
  run: make test-workflow
```

**Note:** Requires Docker services to be available in CI environment.
