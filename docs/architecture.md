# Architecture — Local LLM Chat Engine

This service is intentionally small but production-oriented. It provides clear
boundaries between retrieval, model adapters, API orchestration, and runtime
configuration.

Core components

- `app.py`
	- FastAPI entrypoint.
	- Request validation and endpoint orchestration.
	- Readiness and liveness endpoints.
	- Optional API key guard for write operations.
- `store.py`
	- TF-IDF retrieval backend with update semantics.
	- Snapshot save/load support.
- `embeddings.py`
	- Optional `sentence-transformers` + sklearn nearest-neighbor retrieval.
- `faiss_store.py`
	- Optional FAISS dense vector backend.
- `adapters/`
	- `openai_adapter.py` supports modern and legacy OpenAI SDKs.
	- `huggingface_adapter.py` wraps local text-generation pipeline.
- `settings.py`
	- Centralized environment-driven runtime configuration.
- `cli.py`
	- Ingestion/query utility and snapshot management.

Data flow

1. Client sends ingestion requests with normalized documents.
2. Documents are indexed in one or more retrieval backends.
3. Search retrieves context snippets.
4. Chat endpoints route to selected model adapters with context attached.
5. Streaming chat emits incremental SSE chunks.

Design choices

- Retrieval and generation are decoupled to allow backend replacement.
- Optional dependencies degrade gracefully: TF-IDF remains available even when
	dense retrieval/model packages are absent.
- Configuration is environment-based for containerized deployments.
- Snapshot endpoints make local and CI smoke testing reproducible.
