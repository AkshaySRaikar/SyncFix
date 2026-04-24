# storage/chroma_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid

class ChromaStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection: str = "syncfix"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"Upserted {len(ids)} chunks.")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()