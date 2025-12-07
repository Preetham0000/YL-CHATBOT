"""Simple sentence-aware text splitter."""
from __future__ import annotations

import re
from typing import Iterable, List


_SENTENCE_END = re.compile(r"(?<=[.!?]) +|\n")


def _normalize_blocks(text: str) -> Iterable[str]:
    for block in text.splitlines():
        block = block.strip()
        if block:
            yield block


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping windows respecting sentence boundaries."""

    sentences: List[str] = []
    for block in re.split(r"\n\n+", text):
        block = block.strip()
        if not block:
            continue
        sentences.extend(_SENTENCE_END.split(block))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = len(sentence)
        if current_len + sentence_len > chunk_size and current:
            chunks.append(" ".join(current).strip())
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed)
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]
