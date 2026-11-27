from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer

from app import app, store
from rate_limit import InMemoryRateLimiter


client = TestClient(app)


def setup_function() -> None:
    store.reset()


def test_health_and_ready() -> None:
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    ready_data = ready_resp.json()
    assert ready_data["status"] == "ok"
    assert "stores" in ready_data


def test_ingest_search_and_chat_echo() -> None:
    ingest_resp = client.post(
        "/ingest",
        json={
            "documents": [
                {"id": "d1", "text": "FastAPI service for retrieval augmented generation"}
            ]
        },
    )
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["ingested"] == 1

    search_resp = client.post("/search", json={"query": "retrieval", "k": 3})
    assert search_resp.status_code == 200
    results = search_resp.json()["results"]
    assert len(results) == 1

    chat_resp = client.post(
        "/chat",
        json={"message": "What is this document about?", "k": 3, "model": "echo"},
    )
    assert chat_resp.status_code == 200
    body = chat_resp.json()
    assert body["model"] == "echo"
    assert "response" in body


def test_ingest_returns_422_when_resulting_corpus_has_no_indexable_terms() -> None:
    http = TestClient(app, raise_server_exceptions=False)
    initial = http.post(
        "/ingest", json={"documents": [{"id": "d1", "text": "postgres recovery"}]},
    )
    assert initial.status_code == 200
    query = {"query": "postgres recovery", "k": 2}
    before = http.post("/search", json=query)
    assert before.status_code == 200

    rejected = http.post("/ingest", json={"documents": [
        {"id": "d1", "text": "the and"}, {"id": "d2", "text": "the or"},
    ]})
    assert rejected.status_code == 422
    assert rejected.json() == {
        "detail": (
            "The resulting corpus has no indexable terms. Add descriptive words to the documents."
        ),
    }
    after = http.post("/search", json=query)
    assert after.status_code == 200
    assert after.json() == before.json()

    retried = http.post("/ingest", json={"documents": [
        {"id": "d1", "text": "postgres recovery guide"}, {"id": "d2", "text": "the or"},
    ]})
    assert retried.status_code == 200
    summary = retried.json()
    assert summary.pop("time_s") >= 0
    assert summary == {"received": 2, "ingested": 1, "updated": 1, "total": 2}
    searched = http.post("/search", json=query)
    assert searched.status_code == 200
    hits = searched.json()["results"]
    assert [(hit["id"], hit["text"]) for hit in hits] == [
        ("d1", "postgres recovery guide"), ("d2", "the or"),
    ]


def test_ingest_keeps_unexpected_vectorizer_errors_as_500(monkeypatch) -> None:
    def fail_fit(*args, **kwargs):
        raise ValueError("unexpected vectorizer failure")

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", fail_fit)
    http = TestClient(app, raise_server_exceptions=False)
    response = http.post("/ingest", json={"documents": [{"text": "postgres recovery"}]})
    assert response.status_code == 500


def test_snapshot_save_and_load(tmp_path) -> None:
    snapshot_path = tmp_path / "backup" / "store.pkl"
    missing = client.post("/snapshot/load", json={"path": str(snapshot_path)})
    assert missing.status_code == 404
    assert missing.json() == {"detail": f"Snapshot not found at '{snapshot_path}'"}

    client.post(
        "/ingest",
        json={"documents": [{"id": "d1", "text": "stateful persistence check"}]},
    )

    save_resp = client.post("/snapshot/save", json={"path": str(snapshot_path)})
    assert save_resp.status_code == 200

    store.reset()
    empty_search = client.post("/search", json={"query": "stateful", "k": 3}).json()["results"]
    assert empty_search == []

    load_resp = client.post("/snapshot/load", json={"path": str(snapshot_path)})
    assert load_resp.status_code == 200

    loaded_search = client.post("/search", json={"query": "stateful", "k": 3}).json()["results"]
    assert len(loaded_search) == 1


def test_rate_limit_enforced_on_non_exempt_route() -> None:
    original_limiter = app.state.rate_limiter
    original_exempt = app.state.rate_limit_exempt_paths

    app.state.rate_limiter = InMemoryRateLimiter(limit=1, window_seconds=60, enabled=True)
    app.state.rate_limit_exempt_paths = {"/health", "/ready"}

    try:
        first = client.post("/search", json={"query": "hello", "k": 1})
        assert first.status_code == 200

        second = client.post("/search", json={"query": "hello", "k": 1})
        assert second.status_code == 429
        assert second.json()["detail"] == "Rate limit exceeded"

        health = client.get("/health")
        assert health.status_code == 200
    finally:
        app.state.rate_limiter = original_limiter
        app.state.rate_limit_exempt_paths = original_exempt
