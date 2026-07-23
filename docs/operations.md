# Operations Runbook

Service run command:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

Health checks

- `GET /health` for liveness.
- `GET /ready` for backend availability status.

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
