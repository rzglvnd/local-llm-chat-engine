# Model Adapters & Streaming

This repository provides adapter interfaces for different model backends:

- `adapters/openai_adapter.py` — uses the OpenAI Python SDK and supports streaming via server-sent events (SSE).
- `adapters/huggingface_adapter.py` — wraps `transformers` pipelines; streaming is a best-effort chunked fallback.

Usage examples

1. Streaming via SSE (curl):

```bash
curl -N -X POST http://127.0.0.1:8001/chat_stream -H "Content-Type: application/json" -d '{"message":"Explain RAG in one paragraph","model":"openai"}'
```

2. Non-streaming usage — POST `/chat` with `{ "message": "..." }`.

Notes

- Streaming behavior depends on the backend: OpenAI SDK supports true streaming; local HuggingFace pipelines do not, so we stream generated text in chunks.
- Never expose production API keys in browser clients; use a backend to proxy and apply rate limits and authentication.
