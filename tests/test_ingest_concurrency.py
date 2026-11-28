import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest

import app as api
from store import DocumentStore


def test_search_uses_live_index_during_tfidf_rebuild(monkeypatch):
    store = DocumentStore()
    store.ingest([{"id": "existing", "text": "postgres backup"}])
    rebuild = DocumentStore._rebuild_vectorizer
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "settings", replace(api.settings, api_key="test-key"))

    async def scenario():
        loop = asyncio.get_running_loop()
        rebuild_entered = asyncio.Event()
        release = threading.Event()
        watchdog_expired = threading.Event()

        def gated_rebuild(staged):
            loop.call_soon_threadsafe(rebuild_entered.set)
            # The watchdog can release the rebuild even if the event loop stalls.
            if not release.wait(timeout=5):
                watchdog_expired.set()
            rebuild(staged)

        monkeypatch.setattr(DocumentStore, "_rebuild_vectorizer", gated_rebuild)
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending_ingest = asyncio.create_task(client.post(
                "/ingest",
                headers={"X-API-Key": "test-key"},
                json={"documents": [
                    {"id": "new", "content": "postgres replication", "metadata": {"version": 1}},
                ]},
            ))
            query = {"query": "postgres replication", "k": 1}
            try:
                await asyncio.wait_for(rebuild_entered.wait(), timeout=5)
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                searched = await asyncio.wait_for(client.post("/search", json=query), timeout=5)
                assert searched.status_code == 200
                hit, = searched.json()["results"]
                assert hit["id"] == "existing"
                assert hit["text"] == "postgres backup"
                assert not watchdog_expired.is_set(), "ingestion blocked the event loop"
                assert not pending_ingest.done()
            finally:
                release.set()
                await asyncio.wait_for(
                    asyncio.gather(pending_ingest, return_exceptions=True), timeout=5,
                )

            assert not watchdog_expired.is_set()
            ingested = pending_ingest.result()
            assert ingested.status_code == 200
            summary = ingested.json()
            assert summary.pop("time_s") >= 0
            assert summary == {"received": 1, "ingested": 1, "updated": 0, "total": 2}

            searched = await asyncio.wait_for(client.post("/search", json=query), timeout=5)
            assert searched.status_code == 200
            hit, = searched.json()["results"]
            assert hit["id"] == "new"
            assert hit["text"] == "postgres replication"
            assert hit["metadata"] == {"version": 1}
            assert hit["score"] > 0

    asyncio.run(scenario())


@pytest.mark.parametrize("next_action, expected_ids", [
    ("ingest", {"existing", "first", "second"}),
    ("reset", set()),
    ("restore", {"backup"}),
])
def test_writers_wait_for_ingest_publication(monkeypatch, tmp_path, next_action, expected_ids):
    store = DocumentStore()
    store.ingest([{"id": "existing", "text": "postgres backup"}])
    snapshot_path = tmp_path / "backup.pkl"
    if next_action == "restore":
        backup = DocumentStore()
        backup.ingest([{"id": "backup", "text": "postgres recovery"}])
        backup.save(snapshot_path)

    rebuild = DocumentStore._rebuild_vectorizer
    writer_lock = store._writer_lock
    entered = threading.Event()
    waiting = threading.Event()
    release = threading.Event()

    def gated_rebuild(staged):
        if "postgres replication" in staged.docs and not release.is_set():
            entered.set()
            assert release.wait(timeout=5), "ingestion rebuild watchdog expired"
        rebuild(staged)

    class ObservedWriterLock:
        def __enter__(self):
            if not writer_lock.acquire(blocking=False):
                waiting.set()
                writer_lock.acquire()

        def __exit__(self, exc_type, exc, traceback):
            writer_lock.release()

    monkeypatch.setattr(DocumentStore, "_rebuild_vectorizer", gated_rebuild)
    monkeypatch.setattr(store, "_writer_lock", ObservedWriterLock())
    following_write = {
        "ingest": (store.ingest, [{"id": "second", "text": "postgres indexing"}]),
        "reset": (store.reset,),
        "restore": (store.load, snapshot_path),
    }[next_action]
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(store.ingest, [{"id": "first", "text": "postgres replication"}])
        try:
            assert entered.wait(timeout=5)
            second = workers.submit(*following_write)
            assert waiting.wait(timeout=5), "second writer did not wait for the first"
            assert not second.done()
        finally:
            release.set()

        assert first.result(timeout=5)["total"] == 2
        result = second.result(timeout=5)
        if next_action == "ingest":
            assert result["total"] == 3

    assert store.stats()["documents"] == len(expected_ids)
    assert {hit["id"] for hit in store.search("postgres", k=3)} == expected_ids
