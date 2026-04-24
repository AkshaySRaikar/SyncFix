# ingestion/chunker.py
from typing import List

def sliding_window_chunk(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50
) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks




