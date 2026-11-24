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

TF-IDF search

`/search` and the TF-IDF fallback in `/search_auto` run scoring and wait for the
store lock in the shared thread pool. Search latency still depends on worker
availability and store contention. The optional FAISS and embeddings search
paths run synchronously on the event loop.

`tests/test_search_concurrency.py` verifies that `/health` responds while either
TF-IDF route waits for the real store lock. It then releases the lock and
checks the ranked results, the configured `k` limit, and the fallback label.

TF-IDF ingestion

`/ingest` runs the document update and index rebuild in the shared thread pool.
The store lock covers both steps, so a search that arrives mid-rebuild waits for
the new index. `/health` can respond while the ingest and search are pending.
`tests/test_ingest_concurrency.py` exercises that ordering and verifies that the
waiting search returns the newly indexed document. Worker capacity and the
shared lock still limit concurrent ingestion and search.

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
- Save copies the document lists under the store lock, then serializes the
  captured state. Ingestion can continue while that copy is being written.
- Save writes and closes a temporary file in the destination directory before
  atomically replacing the snapshot. A write or replacement error preserves the
  previous file. The process needs write access to the destination directory.
  Power-loss durability would require additional filesystem synchronization.
- Restore builds a replacement index before acquiring the store lock to install
  the documents and index together. A read or rebuild failure leaves the current
  index searchable. A successful restore supersedes any ingestion that finished
  before the swap. Both indexes can be in memory during the rebuild.
- Snapshot responses count the documents captured by save or installed by load.
- A dedicated worker executes one HTTP snapshot operation at a time per process.
  Directory creation, file checks, serialization, and index rebuilding run on
  that worker. Queued snapshot requests wait without occupying search or chat
  workers. An in-progress snapshot can finish after its requester stops waiting.
