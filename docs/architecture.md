# Architecture — Local LLM Chat Engine

The project is intentionally small and modular to serve as a reference implementation for LLM integration patterns.

Core components

- `store.py` — in-memory TF-IDF document store. Rebuilds the TF-IDF matrix on each ingest for simplicity.
- `engine.py` — model adapters. Provide a minimal `BaseModel` interface so you can plug in OpenAI, HuggingFace, or local inference engines.
- `app.py` — FastAPI service exposing ingest, search, and chat endpoints.
- `cli.py` — convenience commands for local ingestion and queries.

Design notes

- The store separates retrieval from generation so you can swap in a vector database later.
- The model interface is synchronous and minimal; in production, use async adapters and streaming.
