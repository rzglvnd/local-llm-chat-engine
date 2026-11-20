import asyncio
import threading
from dataclasses import replace

import httpx
import pytest

import app as api
from store import DocumentStore


@pytest.mark.parametrize("path", ["/search", "/search_auto"])
def test_health_responds_while_tfidf_search_waits_for_store_lock(monkeypatch, path):
    store = DocumentStore()
    store.ingest([
        {"id": "exact", "text": "postgres replication", "metadata": {"source": "guide"}},
        {"id": "related", "text": "postgres replication slot"},
        {"id": "other", "text": "python interpreter"},
    ])
    query = "postgres replication"
    search = store.search
    expected = {"results": search(query, k=2)}
    if path == "/search_auto":
        expected["backend"] = "tfidf"

    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "settings", replace(api.settings, max_k=2))
    monkeypatch.setattr(api, "faiss_available", False)
    monkeypatch.setattr(api, "embeddings_available", False)

    async def scenario():
        loop = asyncio.get_running_loop()
        lock_held = asyncio.Event()
        search_entered = asyncio.Event()
        release = threading.Event()
        watchdog_expired = threading.Event()

        def hold_lock():
            with store._lock:
                loop.call_soon_threadsafe(lock_held.set)
                # The watchdog can release the lock even if the event loop stalls.
                if not release.wait(timeout=5):
                    watchdog_expired.set()

        def observed_search(query, k):
            loop.call_soon_threadsafe(search_entered.set)
            return search(query, k=k)

        monkeypatch.setattr(store, "search", observed_search)
        holder = threading.Thread(target=hold_lock)
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending_search = None
            holder.start()
            try:
                await asyncio.wait_for(lock_held.wait(), timeout=5)
                pending_search = asyncio.create_task(
                    client.post(path, json={"query": query, "k": 3})
                )
                await asyncio.wait_for(search_entered.wait(), timeout=5)
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                assert not pending_search.done()
                assert not watchdog_expired.is_set(), "search blocked the event loop"
            finally:
                release.set()
                holder.join(timeout=5)
                if pending_search is not None:
                    await asyncio.wait_for(
                        asyncio.gather(pending_search, return_exceptions=True), timeout=5
                    )

            assert not holder.is_alive()
            assert not watchdog_expired.is_set()
            response = pending_search.result()
            assert response.status_code == 200
            assert response.json() == expected

    asyncio.run(scenario())
