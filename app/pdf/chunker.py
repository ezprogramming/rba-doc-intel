"""Chunking logic for cleaned text with paragraph-aware recursive splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    section_hint: str | None = None


def chunk_pages(
    clean_pages: List[str],
    max_tokens: int = 768,
    overlap_pct: float = 0.15
) -> List[Chunk]:
    """
    Group cleaned page text into overlapping chunks using recursive splitting.

    Tries to split on natural boundaries in this order:
    1. Paragraph breaks (\\n\\n)
    2. Sentence breaks (. )
    3. Word breaks ( )

    Args:
        clean_pages: List of cleaned page texts
        max_tokens: Maximum tokens per chunk (default: 768)
        overlap_pct: Percentage overlap between chunks (default: 0.15 = 15%)
    """
    chunks: List[Chunk] = []
    buffer_text = ""
    page_start = 0
    chunk_index = 0

    # Track page boundaries for accurate page_start/page_end
    page_boundaries = []
    current_pos = 0

    for page_num, page_text in enumerate(clean_pages):
        page_boundaries.append((current_pos, current_pos + len(page_text), page_num))
        current_pos += len(page_text) + 1  # +1 for space between pages

    # Concatenate all pages with space separator
    full_text = " ".join(clean_pages)

    # Split into chunks
    start_idx = 0

    while start_idx < len(full_text):
        # Calculate end position for this chunk
        end_idx = min(start_idx + max_tokens * 5, len(full_text))  # Rough estimate: 1 token ≈ 5 chars

        chunk_text = full_text[start_idx:end_idx]

        # Count tokens (rough: split on whitespace)
        tokens = chunk_text.split()

        # If we're over max_tokens, try to split at paragraph or sentence boundary
        if len(tokens) > max_tokens:
            # Try to split at paragraph boundary
            target_chars = int(max_tokens * 4.5)  # Rough char estimate
            chunk_candidate = full_text[start_idx:start_idx + target_chars]

            # Try paragraph break first
            last_para = chunk_candidate.rfind("\n\n")
            if last_para > target_chars * 0.5:  # At least 50% of target
                end_idx = start_idx + last_para + 2
            else:
                # Try sentence break
                last_sent = max(
                    chunk_candidate.rfind(". "),
                    chunk_candidate.rfind(".\n")
                )
                if last_sent > target_chars * 0.5:
                    end_idx = start_idx + last_sent + 2
                else:
                    # Fall back to word boundary
                    last_space = chunk_candidate.rfind(" ")
                    if last_space > 0:
                        end_idx = start_idx + last_space + 1
                    else:
                        end_idx = start_idx + target_chars

            chunk_text = full_text[start_idx:end_idx].strip()

        # Determine page_start and page_end for this chunk
        chunk_page_start = 0
        chunk_page_end = len(clean_pages) - 1

        for boundary_start, boundary_end, page_num in page_boundaries:
            if start_idx >= boundary_start:
                chunk_page_start = page_num
            if end_idx <= boundary_end:
                chunk_page_end = page_num
                break

        # Extract section hint if present (e.g., "3.2 Financial Conditions")
        section_hint = _extract_section_hint(chunk_text)

        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    page_start=chunk_page_start,
                    page_end=chunk_page_end,
                    chunk_index=chunk_index,
                    section_hint=section_hint,
                )
            )
            chunk_index += 1

        # Move start position with overlap
        overlap_tokens = int(len(chunk_text.split()) * overlap_pct)
        # Find position of overlap_tokens from end of chunk
        words = chunk_text.split()
        if len(words) > overlap_tokens:
            overlap_text = " ".join(words[-overlap_tokens:])
            next_start = full_text.find(overlap_text, start_idx)
            if next_start > start_idx:
                start_idx = next_start
            else:
                start_idx = end_idx
        else:
            start_idx = end_idx

        # Safety check to avoid infinite loop
        if start_idx >= len(full_text):
            break

    return chunks


def _extract_section_hint(text: str) -> str | None:
    """
    Extract section header hint from chunk text (e.g., '3.2 Financial Conditions').

    Common RBA patterns:
    - "3.2 Financial Conditions"
    - "Chapter 3: Inflation"
    - "Box A: Housing Market"
    """
    head = text[:800]
    lines = [line.strip(" -\t") for line in head.splitlines() if line.strip()]

    def _match_patterns(candidate: str) -> str | None:
        # Pattern 1: "3.2 Section Title" or nested numbering "2.3.1"
        enum = re.match(r"^(\d+(?:\.\d+){0,2})[\.:]?\s+(.{3,80})", candidate)
        if enum:
            return f"{enum.group(1)} {enum.group(2).strip()}"

        # Pattern 2: Chapter/Section labels
        chapter = re.match(r"^(Chapter|Section|Appendix)\s+([A-Za-z0-9]+)[\.:]?\s+(.{3,80})",
                           candidate,
                           flags=re.IGNORECASE)
        if chapter:
            label = chapter.group(1).title()
            return f"{label} {chapter.group(2)} {chapter.group(3).strip()}"

        # Pattern 3: Boxes (Box A, Box B etc)
        box = re.match(r"^(Box)\s+([A-Z0-9]+)[\.:]?\s+(.{3,80})", candidate, flags=re.IGNORECASE)
        if box:
            return f"{box.group(1).title()} {box.group(2)} {box.group(3).strip()}"

        # Pattern 4: ALL CAPS short headings (<= 10 words)
        words = candidate.split()
        if 1 <= len(words) <= 10:
            letters = [ch for ch in candidate if ch.isalpha()]
            if letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.7:
                return candidate.title()

        return None

    for line in lines[:10]:
        match = _match_patterns(line)
        if match:
            return match.strip()

    # Last resort: try first sentence if it looks like a heading ending with colon
    sentence_match = re.match(r"^([^\n]{3,80}):\s", head)
    if sentence_match:
        candidate = sentence_match.group(1).strip()
        parsed = _match_patterns(candidate)
        return parsed or candidate

    return None
