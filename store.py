from typing import List, Dict, Any, Optional
import pickle
import threading
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
        self._lock = threading.RLock()
        self.docs: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def _rebuild_vectorizer(self) -> None:
        if self.docs:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.docs)
        else:
            self.vectorizer = None
            self.matrix = None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "documents": len(self.docs),
                "index_ready": self.vectorizer is not None and self.matrix is not None,
            }

    def reset(self) -> None:
        with self._lock:
            self.docs = []
            self.metadatas = []
            self.ids = []
            self._id_to_index = {}
            self.vectorizer = None
            self.matrix = None

    def ingest(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents. Each document must include keys: id (optional), text, metadata (optional).

        Returns summary with timing and counts.
        """
        if not isinstance(documents, list):
            raise ValueError("documents must be a list of objects")

        start = perf_counter()
        ingested = 0
        updated = 0

        with self._lock:
            for doc in documents:
                if not isinstance(doc, dict):
                    continue

                text = doc.get("text") or doc.get("content") or ""
                if not text:
                    continue

                incoming_id = doc.get("id")
                doc_id = str(incoming_id) if incoming_id else f"doc_{len(self.docs) + 1}"
                metadata = doc.get("metadata") or {}

                existing_index = self._id_to_index.get(doc_id)
                if existing_index is not None:
                    self.docs[existing_index] = text
                    self.metadatas[existing_index] = metadata
                    updated += 1
                else:
                    self.ids.append(doc_id)
                    self.docs.append(text)
                    self.metadatas.append(metadata)
                    self._id_to_index[doc_id] = len(self.docs) - 1
                    ingested += 1

            if ingested or updated:
                self._rebuild_vectorizer()

            total_docs = len(self.docs)

        return {
            "received": len(documents),
            "ingested": ingested,
            "updated": updated,
            "total": total_docs,
            "time_s": perf_counter() - start,
        }

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k documents for the query with scores and metadata."""
        if not query:
            return []

        with self._lock:
            if not self.docs or self.vectorizer is None or self.matrix is None:
                return []

            safe_k = max(1, int(k))
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.matrix)[0]
            idx = np.argsort(sims)[::-1][:safe_k]
            results = []
            for i in idx:
                results.append(
                    {
                        "id": self.ids[i],
                        "score": float(sims[i]),
                        "text": self.docs[i],
                        "metadata": self.metadatas[i],
                    }
                )
            return results

    def save(self, path: str):
        with self._lock:
            data = {
                "ids": self.ids,
                "docs": self.docs,
                "metadatas": self.metadatas,
            }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        with self._lock:
            self.ids = [str(x) for x in data.get("ids", [])]
            self.docs = data.get("docs", [])
            self.metadatas = data.get("metadatas", [])
            self._id_to_index = {doc_id: i for i, doc_id in enumerate(self.ids)}
            self._rebuild_vectorizer()
