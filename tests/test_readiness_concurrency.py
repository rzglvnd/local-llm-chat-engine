import asyncio
import threading

import httpx

import app as api
from store import DocumentStore


def test_health_responds_while_readiness_waits_for_store_lock(monkeypatch):
    async def scenario():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        store = DocumentStore()
        store.ingest([{"id": "one", "text": "retrieval readiness regression"}])
        monkeypatch.setattr(api, "store", store)
        lock_held = asyncio.Event()
        stats_entered = asyncio.Event()
        release = threading.Event()
        lock_released = threading.Event()
        watchdog_expired = threading.Event()
        stats_threads = []
        original_stats = store.stats

        def hold_lock():
            # RLock ownership must be in another thread to create contention.
            with store._lock:
                loop.call_soon_threadsafe(lock_held.set)
                # Release even if a regression blocks the event loop itself.
                if not release.wait(timeout=5):
                    watchdog_expired.set()
            lock_released.set()

        def observed_stats():
            stats_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(stats_entered.set)
            return original_stats()

        monkeypatch.setattr(store, "stats", observed_stats)
        holder = threading.Thread(target=hold_lock)
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            readiness = None
            holder.start()
            try:
                await asyncio.wait_for(lock_held.wait(), timeout=5)
                readiness = asyncio.create_task(client.get("/ready"))
                await asyncio.wait_for(stats_entered.wait(), timeout=5)
                assert not watchdog_expired.is_set(), "readiness blocked the event loop"
                assert not readiness.done()
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                assert not release.is_set()
                assert not lock_released.is_set()
                assert not readiness.done()
            finally:
                release.set()
                try:
                    if readiness is not None:
                        await asyncio.wait_for(
                            asyncio.gather(readiness, return_exceptions=True), timeout=5
                        )
                finally:
                    holder.join(timeout=5)
                    assert not holder.is_alive()

            response = readiness.result()
            assert response.status_code == 200
            body = response.json()
            assert set(body) == {"status", "stores", "adapters", "runtime"}
            assert body["status"] == "ok"
            assert body["stores"]["tfidf"] == {"documents": 1, "index_ready": True}
            assert set(body["stores"]) == {"tfidf", "embeddings", "faiss"}
            assert body["adapters"]["echo"] is True
            assert set(body["runtime"]) == {"rate_limit", "snapshot"}
        assert len(stats_threads) == 1
        assert stats_threads[0] != loop_thread
        assert not watchdog_expired.is_set()

    asyncio.run(scenario())


def test_readiness_stats_failure_remains_http_500(monkeypatch):
    stats_threads = []

    def failed_stats():
        stats_threads.append(threading.get_ident())
        raise RuntimeError("status collection failed")

    monkeypatch.setattr(api.store, "stats", failed_stats)

    async def scenario():
        transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ready")
        assert response.status_code == 500
        assert response.text == "Internal Server Error"
        assert len(stats_threads) == 1
        assert stats_threads[0] != threading.get_ident()

    asyncio.run(scenario())
