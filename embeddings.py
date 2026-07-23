from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    EMBED_MODEL_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class EmbeddingStore:
    """Embeddings-backed document store.

    If `sentence-transformers` is not installed, this raises ImportError on
    initialization. Use the fallback `DocumentStore` (TF-IDF) in that case.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not EMBED_MODEL_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        self.model = SentenceTransformer(model_name)
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index: Optional[NearestNeighbors] = None

    def _build_index(self):
        if self.embeddings is None:
            return
        # Normalize to use cosine similarity via dot product
        self.embeddings = normalize(self.embeddings)
        self.index = NearestNeighbors(n_neighbors=min(10, len(self.embeddings)), metric="cosine")
        self.index.fit(self.embeddings)

    def ingest(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            emb = self.model.encode(texts, convert_to_numpy=True)
            if self.embeddings is None:
                self.embeddings = emb
            else:
                self.embeddings = np.vstack([self.embeddings, emb])
        self._build_index()
        return {"ingested": len(texts), "total": len(self.docs)}

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = normalize(q_emb)
        dists, idx = self.index.kneighbors(q_emb, n_neighbors=min(k, len(self.docs)))
        results = []
        for dist, i in zip(dists[0], idx[0]):
            score = 1 - float(dist)
            results.append({"id": self.ids[i], "score": score, "text": self.docs[i], "metadata": self.metadatas[i]})
        return results
