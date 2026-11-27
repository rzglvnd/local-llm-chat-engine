import asyncio
import threading
from dataclasses import replace

import httpx

import app as api
from store import DocumentStore


def test_health_responds_during_tfidf_rebuild_and_search_sees_new_document(monkeypatch):
    store = DocumentStore()
    store.ingest([{"id": "existing", "text": "python interpreter"}])
    rebuild = DocumentStore._rebuild_vectorizer
    search = store.search
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "settings", replace(api.settings, api_key="test-key"))

    async def scenario():
        loop = asyncio.get_running_loop()
        rebuild_entered = asyncio.Event()
        search_entered = asyncio.Event()
        release = threading.Event()
        watchdog_expired = threading.Event()

        def gated_rebuild(staged):
            loop.call_soon_threadsafe(rebuild_entered.set)
            # The watchdog can release the rebuild even if the event loop stalls.
            if not release.wait(timeout=5):
                watchdog_expired.set()
            rebuild(staged)

        def observed_search(query, k):
            loop.call_soon_threadsafe(search_entered.set)
            return search(query, k=k)

        monkeypatch.setattr(DocumentStore, "_rebuild_vectorizer", gated_rebuild)
        monkeypatch.setattr(store, "search", observed_search)
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending_ingest = asyncio.create_task(client.post(
                "/ingest",
                headers={"X-API-Key": "test-key"},
                json={"documents": [
                    {"id": "new", "content": "postgres replication", "metadata": {"version": 1}},
                ]},
            ))
            pending_search = None
            try:
                await asyncio.wait_for(rebuild_entered.wait(), timeout=5)
                pending_search = asyncio.create_task(client.post(
                    "/search", json={"query": "postgres replication", "k": 1},
                ))
                await asyncio.wait_for(search_entered.wait(), timeout=5)
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                assert not watchdog_expired.is_set(), "ingestion blocked the event loop"
                assert not pending_ingest.done()
                assert not pending_search.done()
            finally:
                release.set()
                pending = [pending_ingest]
                if pending_search is not None:
                    pending.append(pending_search)
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=5)

            assert not watchdog_expired.is_set()
            ingested = pending_ingest.result()
            assert ingested.status_code == 200
            summary = ingested.json()
            assert summary.pop("time_s") >= 0
            assert summary == {"received": 1, "ingested": 1, "updated": 0, "total": 2}

            searched = pending_search.result()
            assert searched.status_code == 200
            hit, = searched.json()["results"]
            assert hit["id"] == "new"
            assert hit["text"] == "postgres replication"
            assert hit["metadata"] == {"version": 1}
            assert hit["score"] > 0

    asyncio.run(scenario())
