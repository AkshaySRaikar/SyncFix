# retrieval/pipeline.py
import json
import time
from typing import List, Dict, Any
from embedding.embedder import Embedder
from storage.chroma_store import ChromaStore
from embedding.qa_engine import QAEngine

class RetrievalPipeline:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embedder = Embedder()
        self.store = ChromaStore(persist_dir=persist_dir)
        self.qa = QAEngine()

    def _build_context(self, hits: List[Dict], max_hits: int = 3) -> str:
        parts = []
        for i, hit in enumerate(hits[:max_hits]):
            source = f"[Excerpt {i+1} | Page {hit['page']} | {hit['pdf']}]"
            parts.append(f"{source}\n{hit['text'].strip()}")
        return "\n\n".join(parts)



    def retrieve(
        self, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Embed query
        q_emb = self.embedder.embed_query(query).tolist()

        # 2. ANN search
        results = self.store.query(q_emb, top_k=top_k)

        # 3. Parse results
        chunks = results["documents"][0]
        metas  = results["metadatas"][0]
        scores = results["distances"][0]

        # 4. Resolve images and YT links per result
        hits = []
        for chunk, meta, score in zip(chunks, metas, scores):
            hits.append({
                "text":    chunk,
                "score":   round(1 - score, 4),
                "page":    meta.get("page"),
                "pdf":     meta.get("pdf"),
                "images":  json.loads(meta.get("images", "[]")),
                "yt_link": meta.get("yt_link", ""),
            })

        # 5. Build structured context and generate answer
        context = self._build_context(hits)
        answer  = self.qa.answer_question(query, context)

        latency = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "query":      query,
            "hits":       hits,
            "answer":     answer,
            "latency_ms": latency,
        }