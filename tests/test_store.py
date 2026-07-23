import json
from local_llm_chat_engine.store import DocumentStore


def test_ingest_and_search():
    store = DocumentStore()
    docs = [
        {"id": "1", "text": "This document is about machine learning and transformers."},
        {"id": "2", "text": "This text discusses databases, SQL, and performance."},
        {"id": "3", "text": "Notes on LLM evaluation and retrieval-augmented generation."},
    ]
    store.ingest(docs)
    res = store.search("transformers and LLMs", k=2)
    assert len(res) == 2
    # top result should be doc 1 or 3
    ids = {r["id"] for r in res}
    assert ids <= {"1", "3"}
