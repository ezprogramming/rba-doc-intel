"""Text cleaning utilities for RBA PDF documents.

This module removes headers, footers, and normalizes text from RBA publications
(Statement on Monetary Policy, Financial Stability Review, Annual Reports).

Common RBA patterns:
- Headers: Page numbers, report titles, publication dates
- Footers: URLs (www.rba.gov.au), copyright notices
- Repeating lines across 80%+ of pages

Cleaning strategy:
1. Pattern-based removal (regex for known formats)
2. Frequency-based detection (lines repeating in most pages)
3. Whitespace normalization while preserving paragraph structure
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Set, Tuple

# ============================================================================
# RBA-SPECIFIC HEADER PATTERNS
# ============================================================================
# These patterns match lines that appear at the top of RBA PDF pages

HEADER_PATTERNS = [
    # Pattern: "34    Reserve Bank of Australia" (page number + institution name)
    re.compile(r"^\s*\d+\s+Reserve Bank of Australia\s*$", re.IGNORECASE),
    # Pattern: "Page 12" or "Page 12 of 45"
    re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    # Pattern: "S T A T E M E N T   O N   M O N E T A R Y   P O L I C Y"
    # (spaced caps title style used in SMP headers)
    re.compile(r"^\s*S\s*T\s*A\s*T\s*E\s*M\s*E\s*N\s*T.*P\s*O\s*L\s*I\s*C\s*Y\s*$", re.I),
    # Pattern: "Financial Stability Review" (FSR header)
    re.compile(r"^\s*Financial\s+Stability\s+Review\s*$", re.IGNORECASE),
    # Pattern: "February 2025" (publication month/year in headers)
    re.compile(r"^\s*[A-Z][a-z]+\s+20\d{2}\s*$"),
    # Pattern: "ANNUAL REPORT 2024" (annual report headers)
    re.compile(r"^\s*ANNUAL\s+REPORT\s+\d{4}\s*$", re.IGNORECASE),
]

# ============================================================================
# RBA-SPECIFIC FOOTER PATTERNS
# ============================================================================
# These patterns match lines that appear at the bottom of RBA PDF pages

FOOTER_PATTERNS = [
    # Pattern: "www.rba.gov.au" (website URL in footers)
    re.compile(r"^\s*www\.rba\.gov\.au\s*$", re.IGNORECASE),
    # Pattern: "© Reserve Bank of Australia 2025"
    re.compile(r"^\s*©\s*Reserve Bank.*20\d{2}\s*$", re.IGNORECASE),
    # Pattern: "2024 Reserve Bank" (short copyright in footers)
    re.compile(r"^\s*\d{4}\s+Reserve\s+Bank\s*$", re.IGNORECASE),
    # Pattern: Standalone page numbers (just "12" or "- 12 -")
    re.compile(r"^\s*-?\s*\d+\s*-?\s*$"),
]


def detect_repeating_headers_footers(pages: List[str]) -> Tuple[Set[str], Set[str]]:
    """Detect headers and footers that repeat across 80%+ of pages.

    Args:
        pages: List of page texts (one string per page)

    Returns:
        Tuple of (header_lines, footer_lines) sets

    Why this works:
    - True content varies across pages
    - Headers/footers are consistent (page numbers change but format stays same)
    - We look at first 3 lines (headers) and last 3 lines (footers)
    - Lines appearing in 80%+ of pages are likely headers/footers

    Example:
        If "Reserve Bank of Australia" appears in first 3 lines of 45/50 pages,
        it's detected as a header and removed from all pages.
    """
    if len(pages) < 3:
        # Too few pages to reliably detect patterns
        return set(), set()

    line_counts = defaultdict(int)
    total_pages = len(pages)

    for page in pages:
        lines = [line.strip() for line in page.strip().split("\n") if line.strip()]

        if not lines:
            continue

        # Check first 3 lines for headers
        for line in lines[:3]:
            line_counts[("header", line)] += 1

        # Check last 3 lines for footers
        for line in lines[-3:]:
            line_counts[("footer", line)] += 1

    # Lines appearing in 80%+ of pages are repeating headers/footers
    threshold = total_pages * 0.8
    headers = {
        line for (typ, line), count in line_counts.items() if typ == "header" and count >= threshold
    }
    footers = {
        line for (typ, line), count in line_counts.items() if typ == "footer" and count >= threshold
    }

    return headers, footers


def is_header_or_footer(
    line: str, repeating_headers: Set[str], repeating_footers: Set[str]
) -> bool:
    """Check if a line matches header/footer patterns.

    Args:
        line: Text line to check
        repeating_headers: Set of lines detected as repeating headers
        repeating_footers: Set of lines detected as repeating footers

    Returns:
        True if line should be removed (is a header/footer)

    Checks:
    1. Pattern-based: matches regex in HEADER_PATTERNS or FOOTER_PATTERNS
    2. Frequency-based: in repeating_headers or repeating_footers sets
    """
    stripped = line.strip()

    if not stripped:
        return False

    # Check pattern-based header/footer detection
    for pattern in HEADER_PATTERNS + FOOTER_PATTERNS:
        if pattern.match(stripped):
            return True

    # Check frequency-based detection
    if stripped in repeating_headers or stripped in repeating_footers:
        return True

    return False


def clean_text(
    text: str, repeating_headers: Set[str] | None = None, repeating_footers: Set[str] | None = None
) -> str:
    """Normalize whitespace and remove headers/footers while preserving paragraph structure.

    Args:
        text: Raw text extracted from PDF page
        repeating_headers: Optional set of lines to remove (from detect_repeating_headers_footers)
        repeating_footers: Optional set of lines to remove (from detect_repeating_headers_footers)

    Returns:
        Cleaned text with paragraphs separated by double newlines

    Cleaning steps:
    1. Split text into lines
    2. Remove headers/footers (pattern-based + frequency-based)
    3. Normalize whitespace within lines
    4. Group lines into paragraphs (separated by blank lines)
    5. Join lines within paragraphs with single spaces
    6. Join paragraphs with double newlines

    Why preserve paragraph structure?
    - Better for chunking (semantic boundaries)
    - Better for LLM comprehension
    - More natural reading experience
    """
    if repeating_headers is None:
        repeating_headers = set()
    if repeating_footers is None:
        repeating_footers = set()

    paragraphs: list[list[str]] = []
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        # Skip headers/footers entirely
        if is_header_or_footer(stripped, repeating_headers, repeating_footers):
            continue

        # Empty line = paragraph break
        if not stripped:
            if current:
                paragraphs.append(current)
                current = []
            continue

        # Add line to current paragraph
        current.append(stripped)

    # Don't forget the last paragraph
    if current:
        paragraphs.append(current)

    # Join lines within paragraphs with spaces
    # Join paragraphs with double newlines
    return "\n\n".join(" ".join(parts) for parts in paragraphs)
