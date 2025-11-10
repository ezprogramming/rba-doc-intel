"\"\"\"Text cleaning utilities.\"\"\""

from __future__ import annotations

import re

HEADER_RE = re.compile(r"^\s*\d+\s+Reserve Bank of Australia\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize whitespace and drop simple headers/footers."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or HEADER_RE.match(stripped):
            continue
        lines.append(stripped)
    return " ".join(lines)

