# API Reference

Base URL: `http://127.0.0.1:8001`

Authentication

- If `LOCAL_LLM_API_KEY` is set, write endpoints require `X-API-Key`.
- If rate limiting is enabled, non-exempt routes may return `429`.

Health

- `GET /health` → liveness check.
- `GET /ready` → backend readiness and optional dependency status.

Rate limit headers (when enabled)

- `x-ratelimit-limit`
- `x-ratelimit-remaining`
- `x-ratelimit-reset`
- `retry-after` (on `429`)

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

`POST /ingest` returns HTTP 422 if applying the batch would leave the TF-IDF
corpus with no indexable terms. The previous documents and index stay
searchable. Add descriptive words to the documents and retry. A document
containing only stop words can be accepted if the resulting corpus still
contains indexable terms.

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

Save and load share one active operation and one waiting slot per server
process. When both slots are occupied, either route returns HTTP 503 with
`{"detail":"Snapshot queue is full"}` before starting the requested operation.

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
