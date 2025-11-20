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

    def _detect_headers(self, df):
        """Detect and extract header row from DataFrame.

        Args:
            df: pandas DataFrame from Camelot extraction

        Returns:
            DataFrame with proper column names (headers as column names if detected)

        How it works:
        - Check if first row looks like headers (mostly text, not numbers)
        - If yes: Use first row as column names, drop it from data
        - If no: Keep numeric column names ("0", "1", "2"...)
        - Handles duplicate column names by adding suffixes (_1, _2, etc.)
        """

        if len(df) == 0:
            return df

        first_row = df.iloc[0].astype(str)

        # Count how many cells in first row look like text headers (not numbers)
        text_cells = sum(
            1 for val in first_row
            if val and not val.replace('.', '').replace('-', '').replace('%', '').isdigit()
        )

        # If >70% of first row is text, treat it as headers
        if text_cells >= len(first_row) * 0.7:
            # Make column names unique to avoid pandas warning
            headers = first_row.values.tolist()
            unique_headers = []
            seen = {}

            for header in headers:
                if header in seen:
                    seen[header] += 1
                    unique_headers.append(f"{header}_{seen[header]}")
                else:
                    seen[header] = 0
                    unique_headers.append(header)

            df.columns = unique_headers
            return df[1:].reset_index(drop=True)  # Skip header row from data

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
                cleaned = str_val.replace('.', '').replace('-', '')
                cleaned = cleaned.replace('%', '').replace(',', '')
                if str_val and cleaned.isdigit():
                    numeric_cells += 1

        if total_cells == 0:
            return False

        numeric_ratio = numeric_cells / total_cells
        return numeric_ratio >= threshold

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
                "bbox": [100, 200, 500, 400]
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
                suppress_stdout=True  # Suppress Camelot's verbose output
            )

        extracted: List[Dict[str, Any]] = []
        errors: List[str] = []

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

                # Apply header detection to get meaningful column names
                df_with_headers = self._detect_headers(table.df)
                rows = df_with_headers.to_dict('records')

                # Filter out text-only tables (false positives)
                if not self._has_numeric_content(rows):
                    logger.debug(
                        f"Skipping text-only table on page {page_num} "
                        f"(no numeric content detected)"
                    )
                    continue

                extracted.append({
                    "accuracy": float(table.accuracy),
                    "data": rows,
                    "bbox": list(table._bbox) if hasattr(table, '_bbox') and table._bbox else None,
                })

                logger.info(
                    f"Extracted table from page {page_num} "
                    f"({len(rows)} rows, accuracy: {table.accuracy:.1f}%, flavor={flavor})"
                )

            # Stop after first flavor that yields acceptable tables
            if extracted:
                break

        if not extracted and errors:
            logger.warning(
                f"Table extraction failed for page {page_num}: {'; '.join(errors)}"
            )

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
                    logger.debug(
                        f"Detected chart on page {page.number + 1}: "
                        f"{width}x{height}px"
                    )

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
