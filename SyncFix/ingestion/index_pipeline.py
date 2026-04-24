# ingestion/index_pipeline.py  (run once to build the index)
from ingestion.pdf_extractor import extract_text_and_images
from ingestion.chunker import sliding_window_chunk
from ingestion.metadata_linker import build_chunk_metadata, load_yt_metadata
from embedding.embedder import Embedder
from storage.chroma_store import ChromaStore

def index_pdf(pdf_path: str, image_dir: str, yt_map_path: str = None):
    embedder = Embedder()
    store    = ChromaStore()
    yt_map   = load_yt_metadata(yt_map_path) if yt_map_path else {}

    all_texts, all_embeddings, all_metas = [], [], []

    for page_data in extract_text_and_images(pdf_path, image_dir):
        chunks = sliding_window_chunk(page_data["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{page_data['pdf']}_p{page_data['page']}_c{i}"
            meta = build_chunk_metadata(
                chunk_text=chunk,
                chunk_id=chunk_id,
                page_num=page_data["page"],
                pdf_name=page_data["pdf"],
                image_paths=page_data["image_paths"],
                yt_map=yt_map,
            )
            all_texts.append(chunk)
            all_metas.append(meta)

    embeddings = embedder.embed_texts(all_texts).tolist()
    store.upsert_chunks(all_texts, embeddings, all_metas)
    print(f"Indexed {len(all_texts)} chunks from {pdf_path}")