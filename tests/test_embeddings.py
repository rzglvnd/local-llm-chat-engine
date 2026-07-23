import pytest

try:
    from local_llm_chat_engine.embeddings import EmbeddingStore
    EMB_AVAILABLE = True
except Exception:
    EMB_AVAILABLE = False


def test_embeddings_basic():
    if not EMB_AVAILABLE:
        pytest.skip("sentence-transformers not installed; skipping embeddings test")
    try:
        store = EmbeddingStore()
    except Exception:
        pytest.skip("sentence-transformers backend not available at runtime; skipping embeddings test")
    docs = [
        {"id": "1", "text": "Machine learning and transformers."},
        {"id": "2", "text": "Databases and SQL tuning."},
    ]
    store.ingest(docs)
    res = store.search("transformers and LLMs", k=2)
    assert len(res) >= 1
