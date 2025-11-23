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


def _find_paragraph_boundary(text: str, target_pos: int, window: int = 200) -> int:
    """Find nearest paragraph break near target position.

    Args:
        text: Full text to search in
        target_pos: Desired split position
        window: Look within ±window chars of target

    Returns:
        Position of best paragraph boundary

    Why this helps:
    - Avoids splitting mid-paragraph
    - Preserves semantic context
    - Better than arbitrary char count
    """
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)

    # Look for double newline (paragraph break)
    para_break = text.find("\n\n", start, end)
    if para_break != -1:
        return para_break + 2

    # Fallback: single newline
    line_break = text.find("\n", start, end)
    if line_break != -1:
        return line_break + 1

    return target_pos


def _get_sentence_overlap(text: str, num_sentences: int = 2) -> str:
    """Extract last N complete sentences for overlap.

    Args:
        text: Text to extract from
        num_sentences: Number of sentences to include

    Returns:
        Last N sentences as string

    Why sentence-based overlap?
    - Preserves complete thoughts
    - Better than word-based (may split mid-sentence)
    - Helps LLM maintain context across chunks
    """
    # Split on sentence boundaries (. ! ?)
    sentences = re.split(r"[.!?]\s+", text)

    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= num_sentences:
        return text

    # Take last N sentences
    overlap_sentences = sentences[-num_sentences:]
    return ". ".join(overlap_sentences) + "."


def _contains_table_marker(text: str) -> bool:
    """Check if text contains table indicators.

    Args:
        text: Text to check

    Returns:
        True if likely contains table content

    Table indicators:
    - Pipe characters (|) for table borders
    - "Table" keyword
    - "Row count:" from table chunks
    - "Columns:" from table chunks
    """
    indicators = ["|", "table", "row count:", "columns:", "caption:"]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)


def _find_table_aware_boundary(text: str, target_pos: int, window: int = 200) -> int:
    """Find boundary that avoids splitting tables mid-content.

    Args:
        text: Full text to search in
        target_pos: Desired split position
        window: Look within ±window chars of target

    Returns:
        Adjusted position that doesn't split tables

    Strategy:
    1. Check if we're near a table (within window)
    2. If table detected, try to extend past it or contract before it
    3. Fallback to regular paragraph boundary

    Why this matters:
    - Prevents corrupting structured data
    - Ensures complete tables in chunks
    - Better retrieval of numerical data
    """
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)

    # Sample text around target position
    sample = text[start:end]

    # If table markers detected near boundary
    if _contains_table_marker(sample):
        # Try to find end of table (double newline after table content)
        # Tables often end with blank line before next paragraph
        table_end = text.find("\n\n", target_pos, target_pos + window * 2)
        if table_end != -1:
            # Extend past table
            return table_end + 2

        # If can't find table end, try to find table start and contract before it
        # Look backward for "Table" keyword or table marker
        search_start = max(0, target_pos - window * 2)
        table_indicators = ["table", "|", "caption:"]
        for indicator in table_indicators:
            indicator_pos = text.lower().rfind(indicator, search_start, target_pos)
            if indicator_pos != -1:
                # Found table start, look for paragraph break before it
                before_table = text.rfind("\n\n", search_start, indicator_pos)
                if before_table != -1:
                    return before_table + 2

    # Fallback to regular paragraph boundary
    return _find_paragraph_boundary(text, target_pos, window)


def _score_chunk_quality(text: str) -> float:
    """Score chunk quality based on heuristics.

    Args:
        text: Chunk text to evaluate

    Returns:
        Quality score between 0.0 and 1.0

    Quality factors:
    - Sentence boundary ratio (complete sentences)
    - Length variance (not too short/long)
    - Basic coherence signals

    Why filter low-quality chunks?
    - Reduces index noise
    - Improves retrieval precision
    - Better embedding quality
    """
    if not text or len(text) < 10:
        return 0.0

    score = 1.0

    # Factor 1: Sentence completeness
    # Count sentence-ending punctuation
    sentence_endings = text.count(".") + text.count("!") + text.count("?")
    # Rough sentence count (split on whitespace)
    words = text.split()
    word_count = len(words)

    if word_count > 0:
        # Ideal: ~20-30 words per sentence
        avg_words_per_sentence = word_count / max(1, sentence_endings)
        # Penalize very long run-ons (>100 words/sentence) or very short fragments (<5)
        if avg_words_per_sentence > 100:
            score *= 0.7
        elif avg_words_per_sentence < 5:
            score *= 0.8

    # Factor 2: Length variance
    # Penalize very short chunks (likely fragments)
    if word_count < 50:
        score *= 0.6
    # Slightly penalize very long chunks (may be corrupted or table dumps)
    elif word_count > 2000:
        score *= 0.9

    # Factor 3: Check for broken encoding or excessive special chars
    special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
    if special_char_ratio > 0.3:  # More than 30% special characters
        score *= 0.7

    return min(1.0, max(0.0, score))


def _detect_rba_section(text: str) -> str | None:
    """Detect RBA-specific section headings.

    Args:
        text: Chunk text (usually first 500 chars)

    Returns:
        Section name if detected, None otherwise

    RBA report patterns:
    - "Inflation" (standalone heading)
    - "Labour Market" (title case)
    - "Economic Outlook"
    - "GDP Growth"
    - Numbered sections: "1. Introduction", "2.3 Forecast"
    """
    # Check first few lines
    lines = text[:500].split("\n")[:5]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Pattern 1: Common RBA section keywords
        rba_keywords = [
            "inflation",
            "labour market",
            "economic outlook",
            "forecast",
            "gdp",
            "unemployment",
            "wages",
            "financial conditions",
            "housing",
            "consumption",
            "investment",
            "trade",
            "monetary policy",
        ]

        line_lower = line.lower()
        if any(keyword in line_lower for keyword in rba_keywords):
            # Check if it's a heading (short, title case)
            words = line.split()
            if 1 <= len(words) <= 6:  # Headings are usually 1-6 words
                return line

        # Pattern 2: Numbered sections
        if re.match(r"^\d+[\.\)]\s+[A-Z]", line):
            return line

    return None


def chunk_pages(
    clean_pages: List[str],
    max_tokens: int = 768,
    overlap_pct: float = 0.15,
    quality_threshold: float = 0.5,
) -> List[Chunk]:
    """
    Group cleaned page text into overlapping chunks using paragraph-aware,
    section-aware, and table-aware recursive splitting with quality filtering.

    Smart boundary detection order:
    1. Paragraph breaks (\\n\\n) - preserves semantic units
    2. Sentence breaks (. ! ?) - maintains complete thoughts
    3. Word breaks ( ) - fallback only

    Overlap strategy:
    - Sentence-based (not word-based) to preserve context
    - Default 2 sentences overlap between chunks

    Phase 6 improvements:
    - Table-aware boundaries: Extends/contracts to avoid mid-table splits
    - Chunk quality scoring: Filters low-quality fragments
    - Configurable quality threshold (default: 0.5)

    Args:
        clean_pages: List of cleaned page texts
        max_tokens: Maximum tokens per chunk (default: 768)
        overlap_pct: Percentage overlap between chunks (default: 0.15 = 15%)
        quality_threshold: Minimum quality score (0.0-1.0) to keep chunk (default: 0.5)

    Returns:
        List of high-quality chunks with metadata
    """
    chunks: List[Chunk] = []
    chunk_index = 0

    # Track page boundaries for accurate page_start/page_end
    page_boundaries = []
    current_pos = 0

    for page_num, page_text in enumerate(clean_pages):
        page_boundaries.append((current_pos, current_pos + len(page_text), page_num))
        current_pos += len(page_text) + 1  # +1 for space between pages

    # Concatenate all pages with space separator
    full_text = " ".join(clean_pages)

    # Split into chunks with smart boundaries
    start_idx = 0

    while start_idx < len(full_text):
        # Calculate target end position (rough estimate: 1 token ≈ 4.5 chars)
        target_chars = int(max_tokens * 4.5)
        rough_end = min(start_idx + target_chars, len(full_text))

        # Phase 6: Use table-aware boundary detection
        # This extends/contracts boundaries to avoid splitting tables mid-content
        end_idx = _find_table_aware_boundary(full_text, rough_end, window=200)

        # CRITICAL: Ensure we always make forward progress
        # If boundary finder returns same position (no newlines found), force advance
        if end_idx <= start_idx:
            # No valid boundary found, take at least some text to avoid infinite loop
            end_idx = min(start_idx + 100, len(full_text))
            # Extra safety: if still stuck, consume all remaining text
            if end_idx <= start_idx:
                end_idx = len(full_text)

        # Extract candidate chunk
        chunk_text = full_text[start_idx:end_idx].strip()

        # Count tokens (rough: split on whitespace)
        tokens = chunk_text.split()

        # If still over limit after boundary adjustment, force a hard split
        if len(tokens) > max_tokens * 1.2:  # 20% tolerance
            # Recalculate with stricter limit, still using table-aware detection
            strict_target = int(max_tokens * 4.0)
            end_idx = _find_table_aware_boundary(
                full_text,
                start_idx + strict_target,
                window=100,  # Smaller window for stricter control
            )
            # CRITICAL: Ensure we always make forward progress
            if end_idx <= start_idx:
                end_idx = min(start_idx + 100, len(full_text))
                # Extra safety: if still stuck, consume all remaining text
                if end_idx <= start_idx:
                    end_idx = len(full_text)
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

        # Section detection: try RBA-specific patterns first, fallback to generic
        # RBA patterns: "Inflation", "Labour Market", "3.2 GDP Growth", etc.
        section_hint = _detect_rba_section(chunk_text) or _extract_section_hint(chunk_text)

        # Phase 6: Quality filtering
        # Only keep chunks that meet the quality threshold
        if chunk_text:
            quality_score = _score_chunk_quality(chunk_text)
            if quality_score >= quality_threshold:
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
            # Note: Low-quality chunks are silently dropped (score < threshold)

        # Calculate next start position with sentence-based overlap
        # This preserves complete thoughts vs word-based splitting
        overlap_text = _get_sentence_overlap(chunk_text, num_sentences=2)

        # Calculate overlap position (overlap is at END of current chunk)
        if overlap_text and overlap_text != chunk_text:
            overlap_length = len(overlap_text)
            # Overlap starts this many chars before end_idx
            potential_start = end_idx - overlap_length
            # Only use overlap if it's within current chunk
            if potential_start > start_idx:
                start_idx = potential_start
            else:
                # Overlap longer than chunk, skip overlap and move forward
                start_idx = end_idx
        else:
            # No overlap possible (chunk too short)
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
        chapter = re.match(
            r"^(Chapter|Section|Appendix)\s+([A-Za-z0-9]+)[\.:]?\s+(.{3,80})",
            candidate,
            flags=re.IGNORECASE,
        )
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
