"""
src.chunking
=============
Splits raw document text into overlapping chunks for embedding + retrieval.
A simple word-count sliding window: no external tokenizer dependency, and
overlap keeps sentences that straddle a chunk boundary retrievable from
whichever chunk gets matched.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int


def chunk_text(text: str, doc_id: str, chunk_size: int = 180, overlap: int = 40) -> list[Chunk]:
    """
    chunk_size / overlap are in words, not tokens - close enough for a
    demo-scale RAG pipeline and avoids pulling in a tokenizer just for
    splitting. Blank lines are treated as normal whitespace (collapsed),
    so paragraph structure doesn't create degenerate tiny chunks.

    Raises ValueError if overlap >= chunk_size (infinite loop guard).
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    idx = 0
    while start < len(words):
        piece = words[start:start + chunk_size]
        chunks.append(Chunk(text=" ".join(piece), doc_id=doc_id, chunk_index=idx))
        idx += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_file(path: str, chunk_size: int = 180, overlap: int = 40) -> list[Chunk]:
    import os
    with open(path, encoding="utf-8") as f:
        text = f.read()
    doc_id = os.path.basename(path)
    return chunk_text(text, doc_id, chunk_size=chunk_size, overlap=overlap)
