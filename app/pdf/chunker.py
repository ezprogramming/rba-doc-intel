"""Chunking logic for cleaned text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    chunk_index: int


def chunk_pages(clean_pages: List[str], max_tokens: int = 500) -> List[Chunk]:
    """Group cleaned page text into overlapping chunks."""
    chunks: List[Chunk] = []
    buffer: List[str] = []
    page_start = 0
    chunk_index = 0

    for page_num, page_text in enumerate(clean_pages):
        tokens = page_text.split()
        if not buffer:
            page_start = page_num
        buffer.extend(tokens)
        while len(buffer) > max_tokens:
            chunk_tokens = buffer[:max_tokens]
            chunk_text = " ".join(chunk_tokens)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    page_start=page_start,
                    page_end=page_num,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
            buffer = buffer[max_tokens // 4 :]  # 75% overlap
            page_start = max(page_start, page_num - 1)

    if buffer:
        chunks.append(
            Chunk(
                text=" ".join(buffer),
                page_start=page_start,
                page_end=len(clean_pages) - 1,
                chunk_index=chunk_index,
            )
        )

    return chunks
