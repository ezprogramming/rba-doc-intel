"""PDF parsing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF


def extract_pages(pdf_path: Path) -> List[str]:
    """Extract raw text per page from a PDF."""
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages
