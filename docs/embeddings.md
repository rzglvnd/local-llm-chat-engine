# Embeddings & Dense Retrieval

This repository supports an optional embeddings-backed retrieval flow using `sentence-transformers`.

Install the optional dependency:

```bash
python -m pip install -r requirements.txt
```

Endpoints:

- POST `/ingest_embeddings` — ingest documents into the embeddings store (requires `sentence-transformers`).
- POST `/search_embeddings` — run a dense nearest-neighbors search using embeddings.

If `sentence-transformers` is not available, the endpoints will return 400 and you can use the TF-IDF-based `/ingest` and `/search` endpoints instead.
