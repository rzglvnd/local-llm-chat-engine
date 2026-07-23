# API Reference

POST /ingest

Request JSON:

```json
{ "documents": [{"id":"doc1","text":"...","metadata":{}}] }
```

Response:

```json
{ "ingested": 1, "total": 1, "time_s": 0.002 }
```

POST /search

Request JSON:

```json
{ "query": "how to deploy", "k": 5 }
```

Response JSON:

```json
{ "results": [{"id":"doc1","score":0.8,"text":"...","metadata":{}}] }
```

POST /chat

Request JSON:

```json
{ "message": "How do I deploy this system?", "k": 3 }
```

Response JSON:

```json
{ "response": "[EchoModel]...", "context_count": 2 }
```
