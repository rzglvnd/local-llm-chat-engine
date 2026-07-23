from typing import List, Dict, Any, Optional, Tuple
import pickle
from time import perf_counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentStore:
    """In-memory document store with TF-IDF retrieval.

    This is intentionally simple: it rebuilds TF-IDF on every ingest, which
    is fine for small demo datasets. For production, replace with a proper
    vector DB (FAISS, Milvus, Weaviate, etc.).
    """

    def __init__(self):
        self.docs: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def ingest(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents. Each document must include keys: id (optional), text, metadata (optional).

        Returns summary with timing and counts.
        """
        start = perf_counter()
        for doc in documents:
            text = doc.get("text") or doc.get("content") or ""
            if not text:
                continue
            doc_id = str(doc.get("id") or f"doc_{len(self.docs)+1}")
            self.ids.append(doc_id)
            self.docs.append(text)
            self.metadatas.append(doc.get("metadata") or {})

        # rebuild vectorizer
        if self.docs:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.docs)

        return {"ingested": len(documents), "total": len(self.docs), "time_s": perf_counter() - start}

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k documents for the query with scores and metadata."""
        if not self.docs or self.vectorizer is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idx = np.argsort(sims)[::-1][:k]
        results = []
        for i in idx:
            results.append({
                "id": self.ids[i],
                "score": float(sims[i]),
                "text": self.docs[i],
                "metadata": self.metadatas[i],
            })
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"ids": self.ids, "docs": self.docs, "metadatas": self.metadatas}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.ids = data.get("ids", [])
        self.docs = data.get("docs", [])
        self.metadatas = data.get("metadatas", [])
        if self.docs:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.docs)
