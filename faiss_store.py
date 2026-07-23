"""FAISS-backed vector store using sentence-transformers for embeddings.

This module gracefully raises ImportError if required native packages are
not available so the application can fall back to TF-IDF or sklearn.
"""
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    ST_AVAILABLE = False

import numpy as np


class FaissStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is not available")
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers is required for FaissStore")
        self.embedder = SentenceTransformer(model_name)
        self.ids = []
        self.docs = []
        self.metadatas = []
        self.index = None
        self.embeddings = None

    def ingest(self, documents):
        texts = []
        for doc in documents:
            text = doc.get("text") or doc.get("content") or ""
            if not text:
                continue
            self.ids.append(str(doc.get("id") or f"doc_{len(self.docs)+1}"))
            self.docs.append(text)
            self.metadatas.append(doc.get("metadata") or {})
            texts.append(text)

        if texts:
            embs = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")
            # normalize for cosine similarity using IP on normalized vectors
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / (norms + 1e-9)
            if self.embeddings is None:
                self.embeddings = embs
            else:
                self.embeddings = np.vstack([self.embeddings, embs])

            if self.index is None:
                dim = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(self.embeddings)
            else:
                self.index.add(embs)

        return {"ingested": len(texts), "total": len(self.docs)}

    def search(self, query: str, k: int = 5):
        if self.index is None:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "id": self.ids[idx],
                    "score": float(score),
                    "text": self.docs[idx],
                    "metadata": self.metadatas[idx],
                }
            )
        return results
