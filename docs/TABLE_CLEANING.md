# Table Data Cleaning - Simple Best Practices

## Why Simple is Better

**Problem:** Tables extracted from PDFs often have:
- Null/NaN values
- Extra whitespace
- PDF artifacts (null bytes, replacement characters)
- Multiple spaces/newlines

**Bad approach:** Write complex custom logic to handle every edge case
**Good approach:** Use pandas built-in methods (battle-tested, fast, reliable)

## Current Simple Approach

Located in `app/pdf/table_extractor.py:_clean_dataframe()`

```python
def _clean_dataframe(self, df):
    """Clean DataFrame using simple pandas operations."""
    import re

    # 1. Fill NaN/None with empty string
    df = df.fillna('')

    # 2. Convert everything to string and strip whitespace
    df = df.applymap(lambda x: str(x).strip() if x else '')

    # 3. Collapse multiple whitespaces to single space
    df = df.applymap(lambda x: re.sub(r'\s+', ' ', x) if x else '')

    # 4. Remove null bytes and replacement characters (PDF artifacts)
    df = df.applymap(lambda x: x.replace('\x00', '').replace('\ufffd', '') if x else '')

    return df
```

**Applied in extraction pipeline:**
```python
# Clean BEFORE header detection (so headers are clean too)
df_cleaned = self._clean_dataframe(table.df)
df_with_headers = self._detect_headers(df_cleaned)
```

## What This Handles

✅ **Null values** - `NaN`, `None`, `null` → empty string
✅ **Whitespace** - Leading/trailing spaces removed
✅ **Multiple spaces** - `"GDP    growth"` → `"GDP growth"`
✅ **Newlines** - `"Line1\n\nLine2"` → `"Line1 Line2"`
✅ **PDF artifacts** - Null bytes (`\x00`), replacement chars (`�`)

## What This Does NOT Try to "Fix"

❌ **Number formats** - Keep as-is: `"1,234.56"`, `"5.2%"`, `"(123)"`
❌ **OCR errors** - Don't blindly replace O→0 or l→1 (corrupts text)
❌ **Data validation** - Don't check if numbers are "valid"
❌ **Unit conversion** - Don't parse or convert units

**Why not fix these?**
1. **Risk of corruption** - Aggressive fixing can damage legitimate data
2. **LLM can handle it** - Modern LLMs understand "1,234" vs "1234"
3. **Keep it simple** - Less code = fewer bugs

## When to Add More Cleaning

Only add complexity if you see **repeated, systematic problems**:

### Example: Many tables have "n.a." for missing values

**Before adding code, check:**
1. How common is this? (1 table? 10 tables? 100 tables?)
2. Does it break anything? (Usually no - LLM understands "n.a.")
3. Can you fix at the source? (Contact data provider)

**If truly needed:**
```python
# Add to _clean_dataframe() only if problem is widespread
df = df.replace('n.a.', '')
df = df.replace('n/a', '')
```

### Example: Numbers have inconsistent formats

**RBA tables might have:**
- `"5.2"` (decimal)
- `"5.2%"` (percentage)
- `"1,234"` (thousands separator)
- `"(123)"` (negative in parentheses)

**Don't normalize these!**
- Keep original format (preserves meaning)
- LLM understands all formats
- RAG retrieval doesn't need perfect numbers

## Debugging Cleaning Issues

If tables look wrong after cleaning:

```bash
# 1. Export a problematic table
make run CMD="uv run python scripts/verify_table_extraction.py export <table_id> /tmp/table.json"

# 2. Check the raw data
cat /tmp/table.json | jq '.rows[0:3]'

# 3. Compare with PDF manually
# Open PDF at the page number shown in verification
```

**Common issues:**
1. **Headers still look wrong** → Check `_detect_headers()` logic
2. **Data looks corrupt** → Check if cleaning is too aggressive
3. **Nulls still present** → Check if new null pattern exists

## Testing Cleaning

Run the workflow test to verify cleaning works:

```bash
make test-workflow
```

**Expected output:**
```
✓ Document: ... (status: CHUNKS_BUILT)
  Chunks: 45 total, 13 from tables
  Tables: 13 extracted
    Chunk 32 → table 1790 (15 rows, accuracy: 99%)
```

**Check specific table:**
```bash
make verify-tables-doc ARGS="doc <doc_id>"
```

Look at the preview - data should be clean and readable.

## Performance Impact

**Cleaning cost:** ~10-50ms per table (negligible)
- pandas operations are vectorized (fast)
- Regex is only applied to strings, not all cells
- Runs once during ingestion, not during retrieval

**Alternative (slower):**
```python
# ❌ Slow: Cell-by-cell iteration
for row in rows:
    for col, value in row.items():
        cleaned = custom_clean_function(value)  # Calls regex, string ops per cell
```

**Our approach (faster):**
```python
# ✅ Fast: Vectorized pandas operations
df = df.fillna('')  # Single operation on entire DataFrame
df = df.applymap(str.strip)  # Vectorized, optimized in C
```

## Summary: Best Simple Practices

1. **Use pandas built-in methods** - Don't reinvent the wheel
2. **Clean early** - Before header detection
3. **Be conservative** - Only fix obvious problems (null, whitespace)
4. **Don't over-engineer** - Resist urge to "perfect" the data
5. **Test with real PDFs** - Verify cleaning doesn't break anything
6. **Monitor, don't preempt** - Add cleaning only when you see actual problems

**Golden rule:** If the LLM can understand it, don't "fix" it!

## Related Files

- Implementation: `app/pdf/table_extractor.py:_clean_dataframe()`
- Testing: `scripts/test_workflow.py`
- Verification: `scripts/verify_table_extraction.py`
- Documentation: `docs/TABLE_VERIFICATION.md`
