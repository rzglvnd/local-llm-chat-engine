# Local LLM Chat Engine

Mission
> Build a production-ready local AI assistant with API, CLI, Docker, RAG, and plugin support.

## Getting started
- See `docs/roadmap.md`
- Run `examples/quickstart.md`

## Run locally (bare minimum)

1. Install dependencies:

```bash
cd local-llm-chat-engine
python -m pip install -r requirements.txt
```

2. Start the API server:

```bash
uvicorn app:app --reload --port 8001
```

3. Example request:

```bash
curl -s -X POST "http://127.0.0.1:8001/chat" -H "Content-Type: application/json" -d '{"message":"Hello"}' | jq
```

This repository contains a lightweight stub API to iterate on integration and RAG workflows. Replace the stubbed response with a model or RAG pipeline when ready.

## Architecture

- `store.py` — in-memory TF-IDF-based retrieval for quick demos.
- `engine.py` — model adapters (EchoModel for testing, OpenAIModel optional).
- `app.py` — FastAPI endpoints: `/ingest`, `/search`, `/chat`, `/health`.
- `cli.py` — small CLI to ingest and query local files.

## API Reference

- POST `/ingest` — JSON: `{ "documents": [{"id":"...","text":"...","metadata":{}}] }`
- POST `/search` — JSON: `{ "query": "...", "k": 5 }` returns top documents
- POST `/chat` — JSON: `{ "message": "...", "k": 3 }` returns model response and context count

## Development & Tests

Run tests:

```bash
cd local-llm-chat-engine
python -m pip install -r requirements.txt
pytest -q
```

## Next steps / Improvements

- Swap TF-IDF for vector DB (FAISS, Milvus, Weaviate)
- Add persistence for document store
- Add streaming model responses and async inference
- Add authentication and rate limits for multi-tenant setups


