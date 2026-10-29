# Model Adapters & Streaming

This repository provides adapter interfaces for different model backends:

- `adapters/openai_adapter.py`
	- Supports both modern (`OpenAI`) and legacy (`ChatCompletion`) SDK shapes.
	- Native token streaming support.
- `adapters/huggingface_adapter.py`
	- Wraps `transformers` text-generation pipeline.
	- Streaming is best-effort chunking after generation.

Usage examples

1. Streaming via SSE (curl):

```bash
curl -N -X POST http://127.0.0.1:8001/chat_stream -H "Content-Type: application/json" -d '{"message":"Explain RAG in one paragraph","model":"openai"}'
```

2. Non-streaming usage — POST `/chat` with `{ "message": "..." }`.

## Non-streaming execution

`/chat` retrieves contexts in the request's existing event-loop context, then
awaits one framework thread-pool call that initializes the adapter and generates
the response. Both adapter steps run in the same worker thread, so blocking
provider I/O or model initialization does not execute directly on the event loop.
The response still contains `model`, `response`, and `context_count`. Unavailable
adapters and initialization failures still return HTTP 400; unhandled generation
failures return HTTP 500.

`tests/test_chat_concurrency.py` gates fake initialization and generation with
events. In each case, `/health` must respond before the fake work is released.
It also checks execution threads, context forwarding, and error responses.
Run it with `python -m pytest -q tests/test_chat_concurrency.py`; no provider
credentials, network calls, or model downloads are needed.

The framework's shared worker capacity is finite, so calls can queue under load.
Offloading does not impose a provider timeout or forcibly stop a running thread
on cancellation. Retrieval remains synchronous on the event loop; this test does
not establish responsiveness during slow retrieval or CPU-heavy inference.
Provider timeouts, SDK client cleanup, and `/chat_stream` lifecycle behavior need
separate work.

## Streaming response ownership

`OpenAIAdapter.stream()` owns the provider response it acquires for each call.
For both modern and legacy SDK shapes, it calls the response's `close()` when
available after exhaustion, iteration failure, or explicit closure of the adapter
generator. The legacy finish marker also runs cleanup. Iterators without a
callable `close` remain supported. Cleanup failures are logged without replacing
the original iteration error or changing successfully generated output.

A caller that consumes only part of the output must close the adapter generator:

```python
from contextlib import closing

with closing(adapter.stream("Explain retrieval")) as chunks:
    for chunk in chunks:
        print(chunk)
        break
```

A `for` loop's `break` alone does not promise iterator cleanup. Closing a
partially consumed adapter generator runs its `finally` block; closing an
unstarted generator does not acquire a provider response. Response cleanup does
not close the reusable SDK client. Client ownership and `/chat_stream` disconnect
handling remain separate work: this change does not guarantee that an HTTP
disconnect closes the adapter generator or interrupts a blocked provider read.

`tests/test_openai_stream_cleanup.py` uses fake SDKs and response iterators to
check cleanup, unchanged chunks and request arguments, error propagation, and
client ownership. It needs no credentials, SDK installation, or network access.

Configuration

- `OPENAI_API_KEY` for OpenAI backend.
- `LOCAL_LLM_STREAM_CHUNK_SIZE` controls fallback chunking size.

Notes

- Streaming behavior depends on the backend: OpenAI SDK supports true streaming; local HuggingFace pipelines do not, so we stream generated text in chunks.
- Never expose production API keys in browser clients; use a backend to proxy and apply rate limits and authentication.
