import asyncio
import logging
import threading
from dataclasses import replace

import httpx
import pytest

import app as api
from store import DocumentStore


@pytest.mark.parametrize("first_action", ["save", "load"])
def test_snapshot_overload_does_not_block_health(monkeypatch, tmp_path, caplog, first_action):
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
    monkeypatch.setattr(
        api, "settings", replace(api.settings, api_key="test-key", request_logging=True),
    )
    caplog.set_level(logging.INFO, logger=api.logger.name)
    # Admission and start for the first request, admission for the second,
    # first completion, then the second request's start and completion.
    ticks = iter([0.0, 0.125, 0.25, 1.0, 1.125, 1.375])
    monkeypatch.setattr(api, "_snapshot_clock", lambda: next(ticks, 2.0))
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
            async def request(action, path=snapshot, request_id=None):
                headers = {"X-API-Key": "test-key"}
                if request_id is not None:
                    headers["X-Request-ID"] = request_id
                return await client.post(
                    f"/snapshot/{action}", json={"path": str(path)},
                    headers=headers,
                )

            first = asyncio.create_task(request(first_action))
            second = None
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                assert not watchdog_expired.is_set(), "snapshot operation blocked the event loop"
                reserve = api._reserve_snapshot_slot

                def observe_reservation():
                    slot = reserve()
                    queued.set()
                    return slot

                monkeypatch.setattr(api, "_reserve_snapshot_slot", observe_reservation)
                second = asyncio.create_task(request(second_action, request_id="queued-request"))
                await asyncio.wait_for(queued.wait(), timeout=5)
                excess_path = tmp_path / "excess.pkl"
                excess = await asyncio.wait_for(request("save", excess_path), timeout=5)
                assert excess.status_code == 503
                assert excess.json() == {"detail": "Snapshot queue is full"}
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

            first_id = first.result().headers["X-Request-ID"]
            assert second.result().headers["X-Request-ID"] == "queued-request"
            completions = [msg for msg in caplog.messages if msg.startswith("snapshot_complete ")]
            assert completions == [
                f"snapshot_complete request_id={first_id!r} action={first_action} "
                "outcome=succeeded queue_ms=125.00 worker_ms=875.00",
                f"snapshot_complete request_id='queued-request' action={second_action} "
                "outcome=succeeded queue_ms=875.00 worker_ms=250.00",
            ]

            response = await client.post("/search", json={"query": "replication", "k": 1})
            assert response.status_code == 200
            hit, = response.json()["results"]
            assert hit["id"] == expected_id
            assert not excess_path.exists()
            retry = await request("save", excess_path)
            assert retry.status_code == 200
            assert retry.json() == {"saved": str(excess_path), "documents": expected_count}

    asyncio.run(scenario())


@pytest.mark.parametrize("active_fails", [False, True])
def test_snapshot_cancellation_keeps_worker_slot(monkeypatch, tmp_path, caplog, active_fails):
    live = DocumentStore()
    live.ingest([{"id": "live", "text": "postgres replication"}])
    save = live.save
    active_path = tmp_path / "active.pkl"
    cancelled_path = tmp_path / "cancelled.pkl"
    replacement_path = tmp_path / "replacement.pkl"
    reserve = api._reserve_snapshot_slot
    submit = api._snapshot_executor.submit
    monkeypatch.setattr(api, "store", live)
    monkeypatch.setattr(
        api, "settings", replace(api.settings, api_key="test-key", request_logging=True),
    )
    caplog.set_level(logging.INFO, logger=api.logger.name)

    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        reserved = asyncio.Event()
        release = threading.Event()
        submitted = []

        def gated_save(path):
            if path == str(active_path):
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(timeout=5), "snapshot worker watchdog expired"
                if active_fails:
                    raise OSError("snapshot write failed")
            return save(path)

        def observe_reservation():
            slot = reserve()
            reserved.set()
            return slot

        def observe_submission(*args, **kwargs):
            future = submit(*args, **kwargs)
            submitted.append(future)
            return future

        monkeypatch.setattr(live, "save", gated_save)
        monkeypatch.setattr(api, "_reserve_snapshot_slot", observe_reservation)
        monkeypatch.setattr(api._snapshot_executor, "submit", observe_submission)
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async def request(path):
                return await client.post(
                    "/snapshot/save", json={"path": str(path)},
                    headers={"X-API-Key": "test-key", "X-Request-ID": path.stem},
                )

            active = asyncio.create_task(request(active_path))
            tasks = [active]
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                active.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await active

                reserved.clear()
                waiting = asyncio.create_task(request(cancelled_path))
                tasks.append(waiting)
                await asyncio.wait_for(reserved.wait(), timeout=5)
                assert not submitted[0].done()
                excess = await asyncio.wait_for(request(replacement_path), timeout=5)
                assert excess.status_code == 503

                waiting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiting
                reserved.clear()
                replacement = asyncio.create_task(request(replacement_path))
                tasks.append(replacement)
                await asyncio.wait_for(reserved.wait(), timeout=5)
                assert len(submitted) == 1
                assert not replacement.done()
            finally:
                release.set()
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
                if submitted:
                    await asyncio.wait_for(
                        asyncio.gather(asyncio.wrap_future(submitted[0]), return_exceptions=True),
                        timeout=5,
                    )

            assert replacement.result().status_code == 200
            assert len(submitted) == 2
            assert active_path.is_file() is not active_fails
            assert replacement_path.is_file()
            assert not cancelled_path.exists()
            completion = "snapshot_complete request_id='active' action=save"
            outcome = "failed" if active_fails else "succeeded"
            assert f"{completion} outcome={outcome} queue_ms=" in caplog.text
            assert "snapshot_complete request_id='cancelled'" not in caplog.text
            if active_fails:
                assert "Snapshot operation failed after request cancellation" in caplog.text

    asyncio.run(scenario())
