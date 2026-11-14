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
Provider I/O timeout configuration is described below.

## Provider I/O timeout

`LOCAL_LLM_OPENAI_TIMEOUT_SECONDS` sets the OpenAI transport timeout default for
both `/chat` and `/chat_stream`. Its default is **60 seconds**; positive finite
fractional values are accepted. Empty, malformed, zero, negative, NaN, and
infinite values raise a configuration error when settings load, before the API
starts. Set the environment variable before starting the process; `.env.example`
is a configuration template, not an automatically loaded file.

The API passes this value to `OpenAIAdapter(timeout_seconds=...)`. Direct adapter
construction defaults to the same 60 seconds and validates the constructor value
before creating a client. The SDK branches use different transport arguments:

| SDK shape | Default forwarding |
| --- | --- |
| Modern `OpenAI` client | Constructor `timeout=<seconds>` |
| Legacy `ChatCompletion` module | Per-call `request_timeout=<seconds>` for generation and streaming |

The legacy mapping was checked against the published `openai==0.27.0` source,
the minimum supported optional dependency. It does not mutate a shared module
timeout. Existing SDK-specific request kwargs remain forwarded: direct callers
can override these defaults with modern `timeout` or legacy `request_timeout`.
Those overrides retain SDK semantics and are outside this setting's validation.
The HTTP chat request schema does not expose these overrides.

This is an SDK I/O timeout default, not a deadline for the entire request. For
the synchronous transports, read timeouts concern waiting for data; continuing
stream activity can extend total duration. The modern adapter defaults to two
SDK retries, configurable as described below; legacy retry details depend on the
installed version. See the
[official SDK timeout and retry guidance](https://developers.openai.com/api/reference/python#timeouts).
Retrieval, worker queueing, local Hugging Face inference, and cleanup are not
bounded by this setting. A timeout does not forcibly terminate a worker thread.

Timeout errors retain the existing API behavior: unhandled generation errors
produce HTTP 500. Once SSE headers have been sent, a stream error cannot replace
the status; the stream can end with partial output and no `done` event. The
existing ownership cleanup still attempts stream and client closure.

`tests/test_openai_timeouts.py` checks configuration and constructor validation.
`tests/test_openai_client_cleanup.py` checks both SDK mappings through the API,
native request overrides, and cleanup after simulated request/read timeouts.
These are fake SDK tests, not measurements of real network timeout enforcement;
they require no provider credentials or live requests.

## Provider retry policy

`LOCAL_LLM_OPENAI_MAX_RETRIES` controls retries in the modern SDK client for both
chat endpoints. Set it before process startup to a non-negative integer. `0`
disables SDK retries; `N` permits up to `N` additional attempts for SDK-eligible
failures. Two retries therefore permit up to three attempts. Empty, negative,
fractional, and malformed settings fail configuration loading. Direct constructor
validation also rejects booleans and floats, including integral floats.

| Configuration | Modern `OpenAI` client | Legacy `ChatCompletion` module |
| --- | --- | --- |
| Unset / constructor `max_retries=None` | Explicitly passes `max_retries=2`, preserving the current SDK default | Keeps native SDK retry behavior |
| Explicit `0` | Disables retries | Rejects initialization |
| Explicit positive integer | Passes that retry limit | Rejects initialization |

The unset value is retained as `None` in settings so it remains distinct from an
explicit zero. Legacy SDKs do not expose the same per-client control. Explicit
retry configuration on a legacy adapter raises a clear error before requests or
shared-module mutation; the API returns HTTP 400 JSON before starting SSE.
Leave the variable unset when using a legacy SDK. The example environment file
keeps it commented out so copying that template preserves legacy compatibility.

Direct callers can construct `OpenAIAdapter(max_retries=0)` or another supported
integer. The setting is passed to the modern client constructor, not to model
request bodies or shared legacy retry globals. The HTTP request schema does not
expose a per-request retry override.

The SDK retains responsibility for eligible errors, backoff, and server retry
hints. The application adds no retry loop. According to the
[official SDK guidance](https://developers.openai.com/api/reference/python#retries),
stream consumption is not automatically retried. An error after partial output
therefore follows the existing cleanup path without application-level replay or
a fabricated `done` event. This avoids duplicating already delivered output.

A retry count does not establish a total time budget: I/O phases, retry delays,
stream activity, and worker queueing can extend elapsed time. In particular,
`timeout=60` and `max_retries=2` are not a 180-second request deadline.

`tests/test_openai_retries.py` checks settings and constructor validation plus the
explicit modern default. API tests in `tests/test_openai_client_cleanup.py` check
zero/positive forwarding, legacy rejection, cleanup, and a single application SDK
invocation after simulated failure. These fake SDK tests do not measure or
simulate internal HTTP attempt counts or backoff timing and need no credentials.

## Offline SDK retry contracts

`tests/sdk_contract/test_retry_transport.py` complements those fake SDK tests by
running the real **OpenAI SDK 3.51.0** with **HTTPX2 2.16.0**. The pair is pinned
in `requirements-sdk-contract.txt`; this environment requires Python 3.11+.
The dedicated CI job runs on Python 3.11. Local validation also ran on 3.12.

Only the client factory is wrapped to inject an HTTPX2 mock transport and an
`.invalid` base URL. SDK requests, retry decisions, response parsing, error
translation, and streaming code execute normally. Requests use a dummy key;
socket connection and DNS entry points are blocked during each test. Observed
transport requests establish the following contracts for the pinned pair:

| Scenario | Expected transport attempts / result |
| --- | --- |
| Persistent 503, `max_retries=0` | One attempt, then SDK server error |
| Persistent 503, `max_retries=2` | Three attempts, then SDK server error |
| 503 followed by 200 | Two attempts, then generated text |
| Ordinary 400, `max_retries=2` | One attempt, then SDK bad-request error |
| Read failure after one streamed token | One attempt, no replay; response, client, and transport close |

Synthetic error responses include `retry-after-ms: 1`, a public server retry
hint, to keep the real SDK waits short. Tests count requests and assert cleanup;
they neither replace SDK sleep/retry internals nor assert elapsed time. Version
assertions make a changed SDK/transport pair require deliberate revalidation.

In a dedicated Python 3.11+ environment, from the repository root:

```bash
python -m pip install -r requirements-sdk-contract.txt
python -m pytest -q tests/sdk_contract
```

The ordinary suite skips this module when its optional SDK/transport dependencies
are absent. The dedicated CI job installs the pinned pair, reports their versions
with real imports, and runs the contract module explicitly. No model downloads,
provider credentials, or live API requests are needed. Supporting dependencies
are resolved by pip; the SDK and HTTP transport are the explicit version pins.

These checks validate SDK behavior with synthetic HTTP responses, not service
latency, real network timeout enforcement, legacy SDK retries, or every supported
SDK release. They supplement the API wiring and lifecycle tests above.

## SDK client ownership

The modern `OpenAIAdapter` constructs and owns its SDK client. Its `close()`
attempts client closure once; repeated calls do nothing. Closure is terminal even
if the SDK raises: new `generate()` calls and starting `stream()` iteration raise
`RuntimeError`. Direct callers receive SDK close errors and must finish active
generation and close any active stream before closing the adapter. This is a
sequential ownership contract, not a concurrent close/cancel API.

`/chat` runs adapter construction, generation, and a `finally` cleanup in one
worker operation. Cleanup follows success or generation failure, so it also runs
if the awaiting request has stopped waiting but the worker later finishes.
Cleanup exceptions are logged without changing successful output or replacing
the original generation error. A failed close is not retried and does not prove
the underlying resources were released. Blocking generation or closure can delay
worker completion; no hard cleanup deadline is introduced.

The legacy adapter borrows the SDK module; `close()` marks the adapter closed
without closing that shared module. Both endpoints leave the shared echo model
and the Hugging Face adapter outside this SDK cleanup path. `/chat_stream`
transfers its client ownership to the response as described below.

`tests/test_openai_client_cleanup.py` uses fake SDKs to check worker ordering,
success/failure cleanup, idempotence, closed-state behavior, cleanup errors, and
borrowed resources. It requires no credentials, SDK installation, or network.

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
unstarted generator does not acquire a provider response. Closing this adapter
generator alone does not close the SDK client. `/chat_stream` handles that client
separately during response teardown; direct adapter callers must close it after
its active streams have finished or been closed.

`tests/test_openai_stream_cleanup.py` uses fake SDKs and response iterators to
check cleanup, unchanged chunks and request arguments, error propagation, and
client ownership. It needs no credentials, SDK installation, or network access.

## Streaming initialization

`/chat_stream` awaits adapter construction in the framework thread pool before
returning its SSE response. A request owner retains the constructor's result
even if the request stops awaiting its worker. Construction and owner cleanup
share a lock: cleanup waits for construction to finish, or prevents it from
starting if the owner was already closed. Blocking model or client initialization
does not run directly on the event loop. Unavailable adapters and constructor
failures still return HTTP 400 JSON before any streaming headers are sent.

`tests/test_chat_concurrency.py` checks that `/health` responds while fake
initialization is gated by an event, before the fake constructor is released.
It also checks SSE output and context forwarding; initialization error cases
cover both adapter backends. The tests require no provider credentials or model
downloads. Offloading uses shared worker capacity and adds no hard deadline.

## HTTP streaming teardown

`/chat_stream` owns its response iterator and request adapter. Setup retains
ownership until the response is constructed successfully; retrieval or response
construction failures release the client in a worker before propagating the
original failure. Response teardown closes its SSE generator, whose `finally`
closes the adapter iterator when it supports `close()`, then attempts client
closure. Client cleanup is attached to the response itself, so it also runs if
the SSE generator never starts, such as a send failure on the response headers.

This ordering applies after completion, iteration failure, send failure, or
framework disconnect/cancellation teardown. Each owner attempts cleanup once;
client cleanup still runs when stream closure fails. Successful responses retain
the SSE `start`, `chunk`, and `done` events. Failed or interrupted streams do not manufacture a
`done` event. Cleanup failures are logged without replacing the original result.

The response serializes synchronous reads and closure with a lock. Both run in
framework worker threads, so cleanup can wait for an active read without blocking
the event loop or closing a generator while it is executing. An AnyIO cancellation
shield lets finalization run even inside a cancelled scope. AnyIO is an explicit
core dependency because this finalization uses its public cancellation API.

Cancellation does not kill a worker thread. A blocked constructor or provider
read must return before closure can proceed, and a slow `close()` can also delay teardown. There
is no hard shutdown deadline, thread affinity, or guarantee after process
termination or repeated direct task cancellation. Worker capacity remains shared
and finite. Retrieval still executes synchronously in the endpoint; the response
ownership mechanism does not offload retrieval or bound cleanup duration.

`tests/test_chat_stream_cleanup.py` drives ASGI receive/send directly for both
receive-disconnect and send-error paths, and also checks the full application
with its request middleware. Retained iterators expose missing cleanup without
relying on garbage collection. Event-gated reads check AnyIO scope cancellation,
asyncio task cancellation, and a concurrent `/health` request before releasing the
worker. Echo fallback and iterators without `close` remain covered. These tests
also check cancellation during construction, setup failures, unstarted streams,
and ordered client cleanup. `tests/test_openai_client_cleanup.py` checks both SDK
shapes through the API, including legacy module and shared echo ownership.
These tests use fake SDKs/adapters and require no credentials or network access.

Configuration

- `OPENAI_API_KEY` for OpenAI backend.
- `LOCAL_LLM_STREAM_CHUNK_SIZE` controls fallback chunking size.

Notes

- Streaming behavior depends on the backend: OpenAI SDK supports true streaming; local HuggingFace pipelines do not, so we stream generated text in chunks.
- Never expose production API keys in browser clients; use a backend to proxy and apply rate limits and authentication.
