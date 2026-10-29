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
Provider timeouts and SDK client cleanup remain separate work.

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
not close the reusable SDK client; client ownership remains separate work.

`tests/test_openai_stream_cleanup.py` uses fake SDKs and response iterators to
check cleanup, unchanged chunks and request arguments, error propagation, and
client ownership. It needs no credentials, SDK installation, or network access.

## HTTP streaming teardown

`/chat_stream` now owns both layers of iteration: the response closes its SSE
generator, whose `finally` closes the adapter iterator when it supports `close()`.
This runs after completion, iteration failure, send failure, or framework
disconnect/cancellation teardown. Successful responses retain the SSE `start`,
`chunk`, and `done` events. Failed or interrupted streams do not manufacture a
`done` event. Cleanup failures are logged without replacing the original result.

The response serializes synchronous reads and closure with a lock. Both run in
framework worker threads, so cleanup can wait for an active read without blocking
the event loop or closing a generator while it is executing. An AnyIO cancellation
shield lets finalization run even inside a cancelled scope. AnyIO is an explicit
core dependency because this finalization uses its public cancellation API.

Cancellation does not kill a worker thread. A blocked provider read must return
before closure can proceed, and a slow `close()` can also delay teardown. There
is no hard shutdown deadline, thread affinity, or guarantee after process
termination or repeated direct task cancellation. Worker capacity remains shared
and finite. Streaming adapter initialization and retrieval still execute
synchronously in the endpoint; this change only addresses response ownership.

`tests/test_chat_stream_cleanup.py` drives ASGI receive/send directly for both
receive-disconnect and send-error paths, and also checks the full application
with its request middleware. Retained iterators expose missing cleanup without
relying on garbage collection. Event-gated reads check AnyIO scope cancellation,
asyncio task cancellation, and a concurrent `/health` request before releasing the
worker. Echo fallback and iterators without `close` remain covered. These tests
use fake adapters and require no provider credentials or network access.

Configuration

- `OPENAI_API_KEY` for OpenAI backend.
- `LOCAL_LLM_STREAM_CHUNK_SIZE` controls fallback chunking size.

Notes

- Streaming behavior depends on the backend: OpenAI SDK supports true streaming; local HuggingFace pipelines do not, so we stream generated text in chunks.
- Never expose production API keys in browser clients; use a backend to proxy and apply rate limits and authentication.
