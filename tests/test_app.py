from fastapi.testclient import TestClient

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


def test_snapshot_save_and_load(tmp_path) -> None:
    snapshot_path = tmp_path / "store.pkl"

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
