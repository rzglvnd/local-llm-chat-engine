from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from fastapi.testclient import TestClient

import app as api
import store as store_module
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


def test_save_captures_documents_before_concurrent_ingest(monkeypatch, tmp_path):
    store = DocumentStore()
    store.ingest([{"id": "one", "text": "original text", "metadata": {"version": 1}}])
    snapshot_path = tmp_path / "store.pkl"
    dump = store_module.pickle.dump
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(
        api, "settings", replace(api.settings, api_key="test-key", autoload_snapshot=False)
    )

    with ThreadPoolExecutor(max_workers=1) as workers:
        def dump_after_ingest(data, file, protocol):
            ingest = workers.submit(store.ingest, [
                {"id": "one", "text": "updated text", "metadata": {"version": 2}},
                {"id": "two", "text": "new text"},
            ])
            assert ingest.result(timeout=5)["total"] == 2
            dump(data, file, protocol=protocol)

        monkeypatch.setattr(store_module.pickle, "dump", dump_after_ingest)
        with TestClient(api.app) as client:
            saved = client.post(
                "/snapshot/save",
                json={"path": str(snapshot_path)},
                headers={"X-API-Key": "test-key"},
            )
        assert saved.status_code == 200
        assert saved.json() == {"saved": str(snapshot_path), "documents": 1}

    restored = DocumentStore()
    assert restored.load(snapshot_path) == 1
    assert store.stats()["documents"] == 2
    assert restored.stats()["documents"] == 1
    hit, = restored.search("original text", k=1)
    assert hit["id"] == "one"
    assert hit["text"] == "original text"
    assert hit["metadata"] == {"version": 1}
