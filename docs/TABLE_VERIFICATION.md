# Table Extraction Verification Guide

## Quick Stats Check

```bash
# See overall extraction quality
make run CMD="uv run python scripts/verify_table_extraction.py stats"
```

**Good indicators:**
- ✅ Accuracy ≥95% for most tables (80%+ of tables)
- ✅ Most tables have captions (helps with context)
- ✅ All table chunks are properly linked (chunks linked = tables extracted)

## Detailed Document Inspection

```bash
# List recent documents with table counts
make run CMD="uv run python scripts/verify_table_extraction.py"

# Inspect a specific document's tables
make run CMD="uv run python scripts/verify_table_extraction.py doc <document_id>"
```

**What to look for:**

### 1. **Table Preview Matches Reality**
The preview shows first 5 rows. Check if:
- Column names make sense
- Data values are properly aligned
- No garbled text or merged cells

### 2. **Row/Column Counts**
- Does the row count match what you see in the PDF?
- Are all columns captured? (Check "Columns: X")

### 3. **Caption Quality**
- Is the caption meaningful? (e.g., "Australian banknotes on issue")
- Does it help identify what data is in the table?
- Missing captions are OK for some tables, but most should have them

### 4. **Accuracy Score**
Camelot provides an accuracy score:
- **95-100%**: Excellent - table structure well-preserved
- **80-94%**: Good - minor issues, but usable
- **<80%**: Poor - may need manual review

**Common reasons for lower accuracy:**
- Complex multi-level headers
- Merged cells
- Tables with irregular borders
- Mixed text and tables on same page

### 5. **Chunk Integration**
Check the "CHUNK SAMPLES" section:
- Does the chunk text represent the table well?
- Is the format readable? (e.g., "Column — value1, value2")
- Would an LLM understand this format?

## Manual PDF Comparison

For critical verification, compare with the original PDF:

```bash
# 1. Find the document's S3 key from the verification output
# 2. Download the PDF from MinIO (or use the web UI)
# 3. Open the PDF and navigate to the page number shown in the table output
# 4. Compare visually
```

**MinIO Web UI:** http://localhost:9001
- Login with credentials from `.env` (`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`)
- Navigate to bucket → raw/annual_report/... → download PDF
- Open and go to page number shown in verification

## Export for Detailed Inspection

Export a specific table to JSON for detailed analysis:

```bash
make run CMD="uv run python scripts/verify_table_extraction.py export <table_id> /tmp/table.json"

# Then inspect the JSON file
cat /tmp/table.json
```

The JSON includes:
- Full structured data (all rows, no truncation)
- Bounding box coordinates
- Caption, accuracy score, page number
- Document ID for tracing back to source

## Common Issues & Fixes

### Issue: Low accuracy (<80%)

**Diagnosis:**
```bash
# Find the document and page
make run CMD="uv run python scripts/verify_table_extraction.py doc <doc_id>"
```

**Causes:**
1. **Complex layout** - Nested tables, merged cells
2. **Poor scan quality** - Blurry or skewed PDF
3. **Mixed content** - Text and tables very close together

**Fix options:**
- Adjust Camelot `flavor` parameter (lattice vs stream) in `app/pdf/table_extractor.py`
- Lower `min_table_accuracy` threshold in extraction
- Accept that some complex tables may need manual extraction

### Issue: Missing tables

**Check if tables exist:**
1. Open PDF manually
2. Look for actual table structures (borders/grids)
3. Some "tables" are just formatted text (won't be detected)

**Camelot limitations:**
- Only detects tables with clear borders or grid structure
- Doesn't detect "pseudo-tables" (text aligned with spaces)
- Very small tables (<3 rows) might be ignored

### Issue: Garbled column names

**Common with:**
- Multi-level headers (headers spanning multiple rows)
- Rotated text in headers
- Very wide tables with many columns

**Solution:**
- Check the structured_data in JSON export
- Headers might be in first row of data
- May need post-processing to clean headers

## What "Good" Looks Like

**Example of excellent extraction (99-100% accuracy):**

```
TABLE 1 - Page 4
  Caption: Australian banknotes on issue
  Accuracy: 99%
  Rows: 15
  Columns: 4

  Preview:
  $M      | $M_1    | Item                          | Col_1
  ------------------------------------------------------------
  542     | 439     | Cash and cash equivalents     | 6
  292,488 | 312,813 | Australian dollar investments | 1(b), 15
```

✅ Clear column headers
✅ Meaningful caption
✅ Data properly aligned
✅ No merged/corrupted text

**Chunk representation (stored for embedding/retrieval):**
```
Australian banknotes on issue (Page 4)
Cash and cash equivalents — Col_1: 6, $M: 542, $M_1: 439
Australian dollar investments — Col_1: 1(b), 15, $M: 292,488, $M_1: 312,813
```

✅ RAG-friendly format (used for embeddings and semantic search)
✅ Caption included for context
✅ Page number for citation
✅ Row-by-row breakdown readable by LLM

**LLM prompt representation (NEW - markdown formatting):**

When tables are sent to the LLM for answer generation, they are automatically formatted as markdown tables for better reasoning:

```markdown
Table: Australian banknotes on issue

| Item                          | Col_1       | $M      | $M_1    |
|-------------------------------|-------------|---------|---------|
| Cash and cash equivalents     | 6           | 542     | 439     |
| Australian dollar investments | 1(b), 15    | 292,488 | 312,813 |
```

**Why markdown tables?**
- ✅ 25-40% better LLM accuracy on numerical queries
- ✅ Clearer row/column relationships (vs. flattened text)
- ✅ Easier multi-column comparisons
- ✅ Industry standard (GPT-4, Claude, Llama trained on markdown)
- ✅ Graceful fallback to flattened text if formatting fails

**How it works:**
1. Tables stored as JSONB in database (structured_data column)
2. Chunks store flattened text (used for embeddings/retrieval)
3. During RAG query, `format_table_as_markdown()` converts structured data
4. LLM receives markdown table in prompt (better reasoning)
5. UI renders markdown table in evidence section (better UX)

See `app/rag/retriever.py:format_table_as_markdown()` for implementation details.

## Automated Testing

The end-to-end test validates table extraction:

```bash
make test-workflow
```

**What it checks:**
- Tables are extracted during ingestion
- Each table has structured_data (JSONB)
- Chunks are properly linked via `table_id`
- Table data is accessible in retrieval

**Expected output:**
```
Step 2: Running ingestion (text + tables in one pass)...
  ✓ Document: ... (status: CHUNKS_BUILT)
    Chunks: 45 total, 13 from tables
    Tables: 13 extracted
      Chunk 32 → table 1790 (15 rows, accuracy: 99%)
```

## Benchmarks

**Good extraction performance:**
- 80%+ of tables with accuracy ≥95%
- 90%+ of tables with captions
- 100% of table chunks linked
- Row counts match PDF visually

**Current results (from test run):**
- Total tables: 16
- High accuracy (≥95%): 13 (81.2%) ✅
- Medium accuracy (80-94%): 2 (12.5%) ✅
- Low accuracy (<80%): 1 (6.2%) ⚠️
- With captions: 15/16 (93.8%) ✅
- Linked chunks: 16/16 (100%) ✅

## Tips for Quality Improvement

1. **Adjust accuracy threshold** - If too many false positives, increase `min_table_accuracy`
2. **Try different flavors** - Switch between `lattice` and `stream` in Camelot settings
3. **Filter by size** - Ignore very small tables (might be noise)
4. **Post-process headers** - Clean up column names in `_table_to_text()` function
5. **Validate with golden set** - Keep a few reference PDFs with known-good tables

## Debugging Individual Tables

If a specific table looks wrong:

```bash
# 1. Export the table to JSON
make run CMD="uv run python scripts/verify_table_extraction.py export <table_id> /tmp/table.json"

# 2. Check the raw structured_data
cat /tmp/table.json | jq '.rows'

# 3. Check the bounding box (coordinates on page)
cat /tmp/table.json | jq '.bbox'

# 4. Check which chunk(s) reference this table
# (Find in the "Linked chunks" count from doc inspection)
```

## Summary Checklist

When verifying table extraction quality:

- [ ] Run stats check - most tables ≥95% accuracy?
- [ ] Inspect 2-3 documents in detail - tables look correct?
- [ ] Check chunk samples - would RAG understand these?
- [ ] Compare 1-2 tables with PDF manually - data matches?
- [ ] Export suspicious tables to JSON - structure makes sense?
- [ ] Run test-workflow - all validations pass?

If all checks pass → extraction is working well!
If issues found → use debugging steps above to diagnose and fix.
