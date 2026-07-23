# Local LLM Chat Engine

Mission
> Build a production-ready local AI assistant with API, CLI, Docker, RAG, and plugin support.

## What this service provides

- FastAPI endpoints for ingestion, search, chat, and streaming chat.
- Retrieval backends:
	- TF-IDF (always available)
	- Embeddings + `sentence-transformers` (optional)
	- FAISS (optional)
- Adapter pattern for model backends (`echo`, `openai`, `hf`).
- Snapshot save/load endpoints for TF-IDF store persistence.
- Optional API key protection for write endpoints.

## Quick start

Install core dependencies:

```bash
cd local-llm-chat-engine
python -m pip install -r requirements.txt
```

Install optional model/retrieval dependencies:

```bash
python -m pip install -r requirements-optional.txt
```

Run the API:

```bash
uvicorn app:app --reload --port 8001
```

Smoke test:

```bash
curl -s -X POST "http://127.0.0.1:8001/ingest" \
	-H "Content-Type: application/json" \
	-d '{"documents":[{"id":"1","text":"RAG combines retrieval and generation"}]}'

curl -s -X POST "http://127.0.0.1:8001/chat" \
	-H "Content-Type: application/json" \
	-d '{"message":"Explain RAG briefly","model":"echo"}'
```

## Configuration

Environment variables:

You can start from `.env.example` and set required values.

- `LOCAL_LLM_HOST` (default: `0.0.0.0`)
- `LOCAL_LLM_PORT` (default: `8001`)
- `LOCAL_LLM_LOG_LEVEL` (default: `INFO`)
- `LOCAL_LLM_DEFAULT_K` (default: `3`)
- `LOCAL_LLM_MAX_K` (default: `20`)
- `LOCAL_LLM_API_KEY` (optional; if set, required via `X-API-Key` on write endpoints)
- `LOCAL_LLM_CORS_ORIGINS` (optional comma-separated list)
- `LOCAL_LLM_SNAPSHOT_PATH` (default: `./data/store.pkl`)
- `LOCAL_LLM_STREAM_CHUNK_SIZE` (default: `64`)

## API and docs

- API reference: `docs/api.md`
- Architecture: `docs/architecture.md`
- Embeddings: `docs/embeddings.md`
- FAISS runbook: `docs/faiss_runbook.md`
- Adapters + streaming: `docs/adapters.md`
- Operations: `docs/operations.md`
- Deployment: `docs/deployment.md`
- Security: `docs/security.md`

## CLI usage

The CLI supports ingest/query/stats and snapshot operations:

```bash
python cli.py ingest examples/sample_docs.jsonl --save
python cli.py query "retrieval architecture" --k 3
python cli.py stats
python cli.py save --path ./data/store.pkl
python cli.py load --path ./data/store.pkl
```

## Testing and quality

```bash
python -m pip install -r requirements.txt
pytest -q
flake8 .
```

CI runs on push/PR and executes lint + tests on Python 3.10 and 3.11.


