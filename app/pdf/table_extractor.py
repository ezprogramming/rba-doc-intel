"""Table and chart extraction from RBA PDF documents.

RBA reports (SMP, FSR, Annual Reports) contain critical numerical data in tables:
- Economic forecasts (GDP, inflation, unemployment)
- Historical data series
- Risk assessments
- International comparisons

Why extract tables separately?
1. Preserves structure (rows/columns) lost in plain text extraction
2. Enables structured queries ("What is GDP forecast for 2025?")
3. Better grounding for numerical questions
4. Can be stored in separate database table for efficient querying

Extraction approach:
- Camelot library: detects grid-based tables via image processing
- Returns DataFrame-like structures with accuracy scores
- Works best with clear gridlines (RBA tables are well-formatted)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# KNOWN ISSUE: PyMuPDF warnings about invalid graphics states
# ============================================================
# Many RBA PDFs contain invalid graphics states (e.g., /'P0' instead of valid float values)
# These trigger warnings like:
# "Cannot set gray stroke color because /'P0' is an invalid float value"
#
# Root cause: The source PDF files have malformed graphics state dictionaries
# Impact: Warnings clutter logs, but extraction still works correctly (warnings are recoverable)
# Future fix options:
#   1. Pre-process PDFs to repair graphics states (requires PDF manipulation tools)
#   2. Contact RBA to fix their PDF generation process
#   3. Add logging filter to suppress these specific recoverable warnings (production)
#
# For now, warnings are left visible for transparency and future investigation.

# Try to import camelot (optional dependency)
try:
    import camelot

    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning(
        "Camelot not available. Table extraction will be disabled. "
        "Install with: uv pip install 'camelot-py[cv]' opencv-python"
    )


class TableExtractor:
    """Extract structured tables and detect charts from PDF pages.

    Features:
    - Table extraction: Uses Camelot to find grid-based tables
    - Chart detection: Identifies large images likely to be charts/graphs
    - Quality filtering: Only returns tables with >70% accuracy

    Limitations:
    - Works best with gridded tables (RBA documents are good candidates)
    - May miss borderless tables or complex multi-level headers
    - Chart detection is heuristic-based (not ML-based)
    """

    def __init__(self, min_table_accuracy: float = 70.0):
        """Initialize table extractor.

        Args:
            min_table_accuracy: Minimum Camelot accuracy score (0-100) to accept table.
                                Higher = fewer false positives, but may miss some tables.
                                70.0 is a good balance for RBA documents.
        """
        self.min_accuracy = min_table_accuracy

        if not CAMELOT_AVAILABLE:
            logger.warning("TableExtractor initialized but camelot is not available")

    def _is_numeric_data(self, value: str) -> bool:
        """Check if value is numeric data (not a text header).

        Returns True for:
        - Pure integers with decimal points: "0.7", "1.5", "-3.2"
        - Integers < 100 (likely data values, not years)
        - Percentages: "5.2%"

        Returns False for:
        - Years: "2022", "2020", "2024"
        - Words: "October", "GDP", "Total"
        - Mixed text and numbers: "Q1 2024"

        This helps distinguish header rows from data rows.
        """
        if not value:
            return False

        # Remove common data symbols
        cleaned = (
            value.replace(".", "")
            .replace("-", "")
            .replace("%", "")
            .replace(",", "")
            .replace("+", "")
            .strip()
        )

        # Check if it's a pure number
        if not cleaned.isdigit():
            return False

        # If it has a decimal point, it's likely data (0.7, 1.5)
        if "." in value:
            return True

        # If it's a small integer (< 100), it's likely data
        # Years (2020, 2024) are > 1900, so not data
        try:
            num = int(cleaned)
            if num < 100:
                return True
            # Large numbers (years) are headers
            return False
        except ValueError:
            return False

    def _detect_headers(self, df):
        """Detect and extract headers from DataFrame (handles multi-row headers).

        Args:
            df: pandas DataFrame from Camelot extraction

        Returns:
            DataFrame with proper column names (headers as column names if detected)

        Improvements:
        - Detects multi-row headers (common in RBA tables)
        - Merges header rows intelligently
        - Skips rows with mostly empty cells or units/metadata
        - Falls back to last non-empty row if multi-row headers detected

        How multi-row headers work in RBA tables:
        - Row 0: Units or labels (e.g., "Per cent, seasonally adjusted")
        - Row 1: Period groupings (e.g., "Since April")
        - Row 2: Actual column headers (e.g., "October", "September", "August")
        - Row 3+: Data rows (e.g., "Sydney", "0.7", "0.9", ...)
        """

        if len(df) == 0:
            return df

        # Check first 4 rows for header patterns
        max_header_rows = min(4, len(df))
        header_candidates = []

        for row_idx in range(max_header_rows):
            row = df.iloc[row_idx].astype(str)

            # Count non-empty cells
            non_empty = sum(1 for val in row if val.strip())

            # Count text cells (not pure decimal numbers)
            # Years like "2022", "2020" are treated as text (headers), not data
            # Only pure decimals like "0.7", "1.5", "-3" are treated as numbers
            text_cells = sum(
                1 for val in row if val.strip() and not self._is_numeric_data(val.strip())
            )

            # Count empty or whitespace cells
            empty_cells = sum(1 for val in row if not val.strip())

            # Special case: Check if first cell is empty but rest have content
            # This is common in RBA tables (row label column has no header)
            first_empty = not row.iloc[0].strip()
            rest_non_empty = sum(1 for val in row.iloc[1:] if val.strip())

            # This row looks like headers if EITHER:
            # 1. Standard: Has some non-empty cells (>30%) and most are text (>70%)
            # 2. RBA pattern: First cell empty, but >50% of remaining cells are text
            is_header_row = (
                non_empty >= len(row) * 0.3 and text_cells >= max(1, non_empty * 0.7)
            ) or (
                first_empty
                and rest_non_empty >= len(row[1:]) * 0.5
                and text_cells >= max(1, rest_non_empty * 0.7)
            )

            if is_header_row:
                header_candidates.append(
                    {
                        "index": row_idx,
                        "row": row,
                        "non_empty": non_empty,
                        "empty_cells": empty_cells,
                        "text_cells": text_cells,
                    }
                )

        # No header rows detected
        if not header_candidates:
            return df

        # Strategy: Use the LAST header candidate row (most specific headers)
        # For RBA tables: Row 2 usually has actual column names
        # Row 0/1 often have grouping labels or units
        best_header = header_candidates[-1]
        header_row_idx = best_header["index"]
        headers = best_header["row"].values.tolist()

        # Clean and make headers unique
        cleaned_headers = []
        seen = {}

        for i, header in enumerate(headers):
            # Clean whitespace
            header = header.strip()

            # Replace empty headers with meaningful names
            if not header:
                # First column without header is usually row labels
                if i == 0:
                    header = "Item"
                else:
                    header = f"Col_{i}"

            # Handle duplicates
            if header in seen:
                seen[header] += 1
                cleaned_headers.append(f"{header}_{seen[header]}")
            else:
                seen[header] = 0
                cleaned_headers.append(header)

        # Apply headers and skip all header rows
        df.columns = cleaned_headers
        return df[header_row_idx + 1 :].reset_index(drop=True)

    def _clean_dataframe(self, df):
        """Clean DataFrame using simple pandas operations.

        Handles:
        - NaN/None → empty string
        - Whitespace normalization (strip, collapse multiple spaces)
        - Null bytes and replacement characters

        This is SIMPLER and more reliable than custom cell-by-cell cleaning.

        Args:
            df: Raw pandas DataFrame from Camelot

        Returns:
            Cleaned DataFrame
        """
        import re

        # 1. Fill NaN/None with empty string
        df = df.fillna("")

        # 2. Convert everything to string and strip whitespace
        df = df.map(lambda x: str(x).strip() if x else "")

        # 3. Collapse multiple whitespaces to single space
        df = df.map(lambda x: re.sub(r"\s+", " ", x) if x else "")

        # 4. Remove null bytes and replacement characters (PDF artifacts)
        df = df.map(lambda x: x.replace("\x00", "").replace("\ufffd", "") if x else "")

        return df

    def _has_numeric_content(self, rows: List[Dict[str, Any]], threshold: float = 0.3) -> bool:
        """Check if table contains numerical data (not just text layout).

        Args:
            rows: List of row dictionaries from table
            threshold: Minimum fraction of numeric cells required (default 30%)

        Returns:
            True if table has enough numeric content, False if text-only

        Why this matters:
        - Camelot sometimes detects text columns as "tables"
        - These are layout artifacts, not real data tables
        - Real RBA tables have numbers (forecasts, statistics, metrics)
        """
        if not rows:
            return False

        # Check first 10 rows (or all if fewer)
        sample_rows = rows[:10]

        numeric_cells = 0
        total_cells = 0

        for row in sample_rows:
            for value in row.values():
                total_cells += 1
                str_val = str(value).strip()

                # Check if cell contains a number (allow %, -, .)
                cleaned = str_val.replace(".", "").replace("-", "")
                cleaned = cleaned.replace("%", "").replace(",", "")
                if str_val and cleaned.isdigit():
                    numeric_cells += 1

        if total_cells == 0:
            return False

        numeric_ratio = numeric_cells / total_cells
        return numeric_ratio >= threshold

    def _extract_caption_from_text(self, page: fitz.Page, bbox: List[float] | None) -> str | None:
        """Extract table caption from text above the table.

        Args:
            page: PyMuPDF page object
            bbox: Table bounding box [x0, y0, x1, y1] or None

        Returns:
            Caption text if detected, None otherwise

        Caption patterns in RBA PDFs:
        - "Table 1: Economic Forecasts"
        - "Table 3.2 – GDP Growth Projections"
        - Line of text immediately above table bbox
        """
        if not bbox:
            return None

        try:
            # Get text blocks from page (lighter than "dict", no rendering warnings)
            # Format: list of tuples (x0, y0, x1, y1, "text", block_no, block_type)
            # block_type 0 = text, 1 = image
            text_blocks = page.get_text("blocks")

            # Table bbox coordinates (y-axis: larger = lower on page)
            table_top = bbox[1]

            # Look for text above table (within 100 points)
            candidates = []
            for block in text_blocks:
                # Block format: (x0, y0, x1, y1, text, block_no, block_type)
                if len(block) < 7:
                    continue

                block_type = block[6]
                if block_type != 0:  # Only text blocks
                    continue

                # Block bounding box
                block_bbox = block[:4]  # x0, y0, x1, y1
                block_bottom = block_bbox[3]

                # Check if block is above table and close to it
                if block_bottom < table_top and (table_top - block_bottom) < 100:
                    # Extract text (5th element)
                    block_text = block[4].strip()
                    if block_text:
                        candidates.append((block_bottom, block_text))

            # Sort by proximity (closest to table = most likely caption)
            candidates.sort(key=lambda x: x[0], reverse=True)

            # Check candidates for table caption patterns
            for _, text in candidates:
                # Split by newlines - caption is usually on its own line
                lines = [line.strip() for line in text.split("\n") if line.strip()]

                for line in lines:
                    line_lower = line.lower()

                    # Pattern 1: "Table N" or "Table N.M"
                    if line_lower.startswith("table "):
                        return line

                    # Pattern 2: Short descriptive line (2-15 words)
                    # RBA captions range from short ("Business investment") to longer phrases
                    words = line.split()
                    if 2 <= len(words) <= 15:
                        # Filter out lines with too many numbers (likely table data)
                        # Count numeric tokens (integers or decimals)
                        numeric_tokens = sum(
                            1
                            for word in words
                            if word.replace(".", "")
                            .replace(",", "")
                            .replace("-", "")
                            .replace("%", "")
                            .isdigit()
                        )

                        # Accept if less than 30% of words are numbers
                        if numeric_tokens / len(words) < 0.3:
                            return line

            return None

        except Exception as e:
            logger.debug(f"Caption extraction failed: {e}")
            return None

    def extract_tables(self, pdf_path: Path, page_num: int) -> List[Dict[str, Any]]:
        """Extract structured tables from a specific PDF page.

        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number to extract from

        Returns:
            List of table dictionaries, each containing:
            - accuracy: Camelot confidence score (0-100)
            - data: List of row dictionaries (column_name -> value)
            - bbox: Bounding box coordinates [x1, y1, x2, y2]
            - caption: Table caption if detected (NEW)

        How Camelot works:
        1. Converts PDF page to image
        2. Detects gridlines and text regions
        3. Infers table structure from spatial layout
        4. Returns DataFrame representation
        5. Assigns accuracy score based on detection confidence

        Example output:
        [
            {
                "accuracy": 85.3,
                "data": [
                    {"Year": "2024", "GDP": "2.1%", "Inflation": "3.5%"},
                    {"Year": "2025", "GDP": "2.5%", "Inflation": "2.8%"}
                ],
                "bbox": [100, 200, 500, 400],
                "caption": "Table 3.1 – GDP Growth Projections"
            }
        ]
        """
        if not CAMELOT_AVAILABLE:
            logger.debug("Camelot not available, skipping table extraction")
            return []

        def _try_extract(flavor: str) -> List[Dict[str, Any]]:
            return camelot.read_pdf(
                str(pdf_path),
                pages=str(page_num),
                flavor=flavor,
                suppress_stdout=True,  # Suppress Camelot's verbose output
            )

        extracted: List[Dict[str, Any]] = []
        errors: List[str] = []

        # Open PDF once for caption extraction
        try:
            pdf_doc = fitz.open(pdf_path)
            page = pdf_doc[page_num - 1]  # 0-based indexing
        except Exception as e:
            logger.warning(f"Failed to open PDF for caption extraction: {e}")
            pdf_doc = None
            page = None

        for flavor in ("lattice", "stream"):
            try:
                tables = _try_extract(flavor)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{flavor}: {exc}")
                continue

            if tables.n == 0:
                # Try stream if lattice returns no tables
                continue

            for table in tables:
                # Filter by accuracy threshold
                if table.accuracy < self.min_accuracy:
                    logger.debug(
                        f"Skipping table on page {page_num} "
                        f"(accuracy {table.accuracy:.1f} < {self.min_accuracy})"
                    )
                    continue

                # Clean the DataFrame first (handles nulls, whitespace, artifacts)
                df_cleaned = self._clean_dataframe(table.df)

                # Apply header detection to get meaningful column names
                df_with_headers = self._detect_headers(df_cleaned)
                rows = df_with_headers.to_dict("records")

                # Filter out text-only tables (false positives)
                if not self._has_numeric_content(rows):
                    logger.debug(
                        f"Skipping text-only table on page {page_num} (no numeric content detected)"
                    )
                    continue

                # Extract bbox
                bbox = list(table._bbox) if hasattr(table, "_bbox") and table._bbox else None

                # Extract caption from surrounding text
                caption = None
                if page and bbox:
                    caption = self._extract_caption_from_text(page, bbox)

                extracted.append(
                    {
                        "accuracy": float(table.accuracy),
                        "data": rows,
                        "bbox": bbox,
                        "caption": caption,
                    }
                )

                logger.info(
                    f"Extracted table from page {page_num} "
                    f"({len(rows)} rows, accuracy: {table.accuracy:.1f}%, "
                    f"caption: {caption or 'none'}, flavor={flavor})"
                )

            # Stop after first flavor that yields acceptable tables
            if extracted:
                break

        # Clean up PDF document
        if pdf_doc:
            pdf_doc.close()

        if not extracted and errors:
            logger.warning(f"Table extraction failed for page {page_num}: {'; '.join(errors)}")

        return extracted

    def detect_charts(self, page: fitz.Page) -> int:
        """Detect charts and graphs on a PDF page.

        Args:
            page: PyMuPDF Page object

        Returns:
            Number of charts/graphs detected

        Detection heuristic:
        - Extract all images from page
        - Filter by size (width > 200px and height > 150px)
        - Assumption: Large images in RBA reports are usually charts

        Limitations:
        - Heuristic-based (not ML-based classification)
        - May count photos/logos as charts if large enough
        - Won't detect charts rendered as vector graphics (rare in RBA PDFs)

        Why count charts?
        - Can add metadata flag: "contains_chart": True
        - Helps retriever prioritize pages with visual data
        - Future: could extract chart images for multimodal RAG
        """
        try:
            images = page.get_images()

            # Count large images (likely charts/graphs, not small icons/logos)
            charts = 0
            for img_index, img in enumerate(images):
                # img[2] = width, img[3] = height (in pixels)
                # Why 200x150 threshold? Typical RBA chart is 400x300+
                # Filters out small icons, logos, decorative elements
                width = img[2] if len(img) > 2 else 0
                height = img[3] if len(img) > 3 else 0

                if width > 200 and height > 150:
                    charts += 1
                    logger.debug(f"Detected chart on page {page.number + 1}: {width}x{height}px")

            return charts

        except Exception as e:
            logger.warning(f"Chart detection failed for page {page.number + 1}: {e}")
            return 0

    def extract_page_metadata(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        """Extract comprehensive metadata from a page (tables + charts).

        Args:
            pdf_path: Path to PDF file
            page_num: 1-based page number

        Returns:
            Dictionary with:
            - has_tables: bool
            - table_count: int
            - tables: List[Dict] (table data)
            - has_charts: bool
            - chart_count: int

        Use case:
        - Enrich chunk metadata: {"contains_table": True, "contains_chart": True}
        - Improve retrieval: boost chunks with tables for numerical queries
        - Analytics: "What % of SMP pages contain forecasts tables?"
        """
        metadata = {
            "has_tables": False,
            "table_count": 0,
            "tables": [],
            "has_charts": False,
            "chart_count": 0,
        }

        # Extract tables
        tables = self.extract_tables(pdf_path, page_num)
        if tables:
            metadata["has_tables"] = True
            metadata["table_count"] = len(tables)
            metadata["tables"] = tables

        # Detect charts
        try:
            # Open PDF to get PyMuPDF page object
            # Why with statement? Ensures PDF is closed after use
            with fitz.open(pdf_path) as doc:
                # PyMuPDF uses 0-based indexing
                page = doc[page_num - 1]
                chart_count = self.detect_charts(page)

                if chart_count > 0:
                    metadata["has_charts"] = True
                    metadata["chart_count"] = chart_count

        except Exception as e:
            logger.warning(f"Failed to detect charts on page {page_num}: {e}")

        return metadata
