# Embeddings & Dense Retrieval

This repository supports an optional embeddings-backed retrieval flow using `sentence-transformers`.

Install optional dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

Endpoints:

- POST `/ingest_embeddings` — ingest documents into the embeddings store (requires `sentence-transformers`).
- POST `/search_embeddings` — run a dense nearest-neighbors search using embeddings.

If `sentence-transformers` is not available, the endpoints will return 400 and you can use the TF-IDF-based `/ingest` and `/search` endpoints instead.

Operational notes

- Embedding model default: `all-MiniLM-L6-v2`.
- Dense retrieval quality is sensitive to chunk size and document normalization.
- For production workloads, track retrieval quality with offline relevance tests.
- Use `POST /search_auto` to automatically select the best available backend.
