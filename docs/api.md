# API Reference

Base URL: `http://127.0.0.1:8001`

Authentication

- If `LOCAL_LLM_API_KEY` is set, write endpoints require `X-API-Key`.

Health

- `GET /health` → liveness check.
- `GET /ready` → backend readiness and optional dependency status.

Ingestion

- `POST /ingest`
- `POST /ingest_embeddings`
- `POST /ingest_faiss`

Request JSON:

```json
{ "documents": [{"id":"doc1","text":"...","metadata":{}}] }
```

Response:

```json
{ "received": 1, "ingested": 1, "updated": 0, "total": 1, "time_s": 0.002 }
```

Search

- `POST /search`
- `POST /search_embeddings`
- `POST /search_faiss`
- `POST /search_auto`

Request JSON:

```json
{ "query": "how to deploy", "k": 5 }
```

Response JSON:

```json
{ "results": [{"id":"doc1","score":0.8,"text":"...","metadata":{}}] }
```

For `search_auto`, response includes selected backend:

```json
{ "backend": "tfidf", "results": [{"id":"doc1","score":0.8,"text":"..."}] }
```

Chat

- `POST /chat`

Request JSON:

```json
{ "message": "How do I deploy this system?", "k": 3, "model": "echo" }
```

Response JSON:

```json
{ "model": "echo", "response": "[EchoModel]...", "context_count": 2 }
```

Streaming chat

- `POST /chat_stream`
- Content-Type: `application/json`
- Response Content-Type: `text/event-stream`

Request JSON:

```json
{ "message": "Explain RAG", "model": "openai", "k": 3 }
```

SSE events:

- `{"event":"start","model":"openai"}`
- `{"event":"chunk","text":"..."}` repeated
- `{"event":"done"}`

Snapshots

- `POST /snapshot/save`
- `POST /snapshot/load`

Request JSON:

```json
{ "path": "./data/store.pkl" }
```

Response JSON (save):

```json
{ "saved": "./data/store.pkl", "documents": 123 }
```

Response JSON (load):

```json
{ "loaded": "./data/store.pkl", "documents": 123 }
```
