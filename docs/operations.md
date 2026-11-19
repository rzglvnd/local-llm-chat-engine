# Operations Runbook

Service run command:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

Health checks

- `GET /health` for liveness.
- `GET /ready` for backend availability status.

`/ready` awaits status collection in the framework's shared thread pool. TF-IDF
statistics acquire the store lock, which can be held by a chat retrieval worker.
Readiness waits for that lock in a worker so the wait itself does not block the
event loop or prevent `/health` from responding. The successful HTTP 200 payload
is unchanged; an unhandled status-collection failure still returns HTTP 500.
The endpoint reports backend state and availability flags, not active provider
connectivity or a new readiness policy.

This adds no probe deadline or fail-fast response. Readiness can still wait for
worker capacity or the store lock. Other endpoints that perform synchronous
work or acquire the lock on the event loop can still delay liveness.

`tests/test_readiness_concurrency.py` holds the real store lock in another thread
and verifies that `/health` responds while `/ready` is pending. After releasing
the lock, it checks the readiness response and store statistics. A separate case
checks status-collection failure. These controlled ordering checks do not
measure production probe latency or establish behavior under worker exhaustion.

Rate limiting

- Enable with `LOCAL_LLM_RATE_LIMIT_ENABLED=true`.
- Tune throughput with `LOCAL_LLM_RATE_LIMIT_REQUESTS_PER_MINUTE`.
- Keep `/health` and `/ready` exempt (default behavior).

Recommended startup sequence

1. Start service.
2. Verify `GET /ready` returns `status=ok`.
3. Ingest a small known dataset.
4. Run `POST /search` smoke query.

Incident checklist

- `5xx` spike:
  - Check adapter availability (`/ready`).
  - Validate upstream key/env configuration.
  - Temporarily route traffic to `model=echo` if remote model is failing.
- `429` spike:
  - Review current caller traffic patterns.
  - Increase request budget if legitimate traffic is being throttled.
  - Ensure probes and internal callbacks use exempt endpoints where appropriate.
- Empty retrieval results:
  - Confirm documents were ingested.
  - Check `k` values and query quality.
  - Validate expected backend (`/search_auto`).
- Snapshot failures:
  - Verify write permissions for `LOCAL_LLM_SNAPSHOT_PATH`.

Backup and restore

- Save: `POST /snapshot/save`
- Restore: `POST /snapshot/load`
- Store snapshots in a durable volume when running in containers.
