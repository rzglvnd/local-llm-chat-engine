# FAISS Runbook

This guide explains how to enable FAISS-backed dense retrieval for `local-llm-chat-engine`.

Prerequisites

- Python 3.10+
- `pip` and a system with a compatible wheel for `faiss-cpu` (Linux/Windows wheels are published for many platforms). If `faiss-cpu` fails to install, consider using the Docker image in this repo's `docker-compose.yml` which isolates binary dependencies.
- `sentence-transformers` for embeddings.

Install (local, minimal):

```bash
python -m pip install --upgrade pip
pip install faiss-cpu sentence-transformers
```

If installation fails on Windows, use the Docker Compose flow (recommended):

```bash
docker compose up --build
```

Endpoints

- `POST /ingest_faiss` — JSON `{ "documents": [{"id":"...","text":"...","metadata":{}}] }`
- `POST /search_faiss` — JSON `{ "query": "...", "k": 5 }` returns top-k results.

Example (curl):

```bash
curl -s -X POST "http://127.0.0.1:8001/ingest_faiss" -H "Content-Type: application/json" -d '{"documents":[{"id":"1","text":"This is a sample doc about LLMs."}]}' | jq

curl -s -X POST "http://127.0.0.1:8001/search_faiss" -H "Content-Type: application/json" -d '{"query":"LLMs"}' | jq
```

Troubleshooting

- If FAISS import fails with native errors, prefer the Docker route. The `faiss-cpu` wheel is the simplest cross-platform option.
- Check memory usage: FAISS keeps embeddings in RAM; for large corpora use an index that supports on-disk persistence.
