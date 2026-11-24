import asyncio
import threading
from dataclasses import replace

import httpx
import pytest

import app as api
from store import DocumentStore


@pytest.mark.parametrize("first_action", ["save", "load"])
def test_snapshot_requests_serialize_without_blocking_health(monkeypatch, tmp_path, first_action):
    snapshot = tmp_path / "backup" / "store.pkl"
    snapshot.parent.mkdir()
    backup = DocumentStore()
    backup.ingest([{"id": "backup", "text": "postgres replication"}])
    backup.save(snapshot)
    live = DocumentStore()
    live.ingest([
        {"id": "live", "text": "postgres replication slots"},
        {"id": "extra", "text": "python interpreter"},
    ])
    monkeypatch.setattr(api, "store", live)
    monkeypatch.setattr(api, "settings", replace(api.settings, api_key="test-key"))
    second_action = "load" if first_action == "save" else "save"
    first_operation = getattr(live, first_action)
    second_operation = getattr(live, second_action)

    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        queued = asyncio.Event()
        release = threading.Event()
        watchdog_expired = threading.Event()
        first_finished = threading.Event()

        def gated_operation(path):
            loop.call_soon_threadsafe(entered.set)
            # Restore progress if a regression stalls the event loop.
            if not release.wait(timeout=5):
                watchdog_expired.set()
            result = first_operation(path)
            first_finished.set()
            return result

        def check_order(path):
            assert first_finished.is_set(), "snapshot operations overlapped"
            return second_operation(path)

        monkeypatch.setattr(live, first_action, gated_operation)
        monkeypatch.setattr(live, second_action, check_order)
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async def request(action):
                return await client.post(
                    f"/snapshot/{action}", json={"path": str(snapshot)},
                    headers={"X-API-Key": "test-key"},
                )

            first = asyncio.create_task(request(first_action))
            second = None
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                assert not watchdog_expired.is_set(), "snapshot operation blocked the event loop"
                submit = api._snapshot_executor.submit

                def observe_submit(*args, **kwargs):
                    future = submit(*args, **kwargs)
                    queued.set()
                    return future

                monkeypatch.setattr(api._snapshot_executor, "submit", observe_submit)
                second = asyncio.create_task(request(second_action))
                await asyncio.wait_for(queued.wait(), timeout=5)
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                assert not first.done()
                assert not second.done()
                assert not watchdog_expired.is_set()
            finally:
                release.set()
                pending = [first] if second is None else [first, second]
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=5)

            expected_count = 2 if first_action == "save" else 1
            expected_id = "live" if first_action == "save" else "backup"
            for action, task in [(first_action, first), (second_action, second)]:
                response = task.result()
                assert response.status_code == 200
                key = "saved" if action == "save" else "loaded"
                assert response.json() == {key: str(snapshot), "documents": expected_count}

            response = await client.post("/search", json={"query": "replication", "k": 1})
            assert response.status_code == 200
            hit, = response.json()["results"]
            assert hit["id"] == expected_id

    asyncio.run(scenario())
