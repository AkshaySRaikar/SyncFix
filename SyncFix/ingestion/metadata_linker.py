# ingestion/metadata_linker.py
import json
from typing import List, Optional

def load_yt_metadata(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def build_chunk_metadata(
    chunk_text: str,
    chunk_id: str,
    page_num: int,
    pdf_name: str,
    image_paths: List[str],
    yt_map: Optional[dict] = None,
) -> dict:
    """
    Attach image paths and YouTube timestamps to a chunk.
    YT map format: {"keyword": {"url": "...", "t": 120}}
    """
    yt_link = None
    if yt_map:
        for keyword, info in yt_map.items():
            if keyword.lower() in chunk_text.lower():
                yt_link = f"{info['url']}&t={info['t']}"
                break

    return {
        "chunk_id": chunk_id,
        "page": page_num,
        "pdf": pdf_name,
        "images": json.dumps(image_paths),   # ChromaDB requires string metadata
        "yt_link": yt_link or "",
    }
