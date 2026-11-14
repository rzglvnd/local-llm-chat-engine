"""Exercise real SDK retry/stream code at an offline HTTP transport boundary."""
import json
import socket
from importlib.metadata import version
from types import SimpleNamespace

import pytest

from adapters import openai_adapter

openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")

BASE_URL = "https://sdk-contract.invalid/v1"
API_KEY = "sk-contract-test-only"
MODEL = "contract-model"


@pytest.fixture(autouse=True)
def require_pinned_versions_and_block_network(monkeypatch):
    assert version("openai") == "3.51.0", "Install requirements-sdk-contract.txt"
    assert version("httpx2") == "2.16.0", "Install requirements-sdk-contract.txt"

    def deny_network(*args, **kwargs):
        pytest.fail("SDK contract tests must not resolve hosts or open network sockets")

    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    for name in ("OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)


class RecordingTransport(httpx2.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.requests = []
        self.responses = []
        self.close_calls = 0

    def handle_request(self, request):
        assert request.method == "POST"
        assert request.url.host == "sdk-contract.invalid"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        assert json.loads(request.content)["model"] == MODEL
        self.requests.append(request)
        response = super().handle_request(request)
        self.responses.append(response)
        return response

    def close(self):
        self.close_calls += 1
        super().close()


@pytest.fixture
def build_adapter(monkeypatch):
    real_sdk_client = openai.OpenAI
    resources = []

    def build(handler, max_retries):
        transport = RecordingTransport(handler)
        client = httpx2.Client(transport=transport, trust_env=False)

        def sdk_factory(**kwargs):
            # Inject only the HTTP boundary; SDK requests, parsing, errors,
            # retry decisions, and backoff still execute in the real SDK.
            return real_sdk_client(base_url=BASE_URL, http_client=client, **kwargs)

        monkeypatch.setattr(openai_adapter.openai, "OpenAI", sdk_factory)
        try:
            adapter = openai_adapter.OpenAIAdapter(
                api_key=API_KEY, model=MODEL, max_retries=max_retries
            )
        except BaseException:
            client.close()
            raise
        resource = SimpleNamespace(adapter=adapter, client=client, transport=transport)
        resources.append(resource)
        return resource

    yield build
    for resource in resources:
        resource.adapter.close()
        assert resource.client.is_closed
        assert resource.transport.close_calls == 1
        assert all(response.is_closed for response in resource.transport.responses)


def error_response(status):
    return httpx2.Response(
        status,
        json={"error": {"message": "synthetic error", "type": "test_error", "code": None}},
        # Public server retry hint keeps real SDK backoff short. No clock or
        # private retry implementation is replaced, and no duration is asserted.
        headers={"retry-after-ms": "1"},
    )


def success_response():
    return httpx2.Response(200, json={
        "id": "chatcmpl-contract", "object": "chat.completion", "created": 0, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
    })


@pytest.mark.parametrize("max_retries, expected_attempts", [(0, 1), (2, 3)])
def test_persistent_503_obeys_real_sdk_retry_limit(build_adapter, max_retries, expected_attempts):
    resource = build_adapter(lambda request: error_response(503), max_retries)
    with pytest.raises(openai.InternalServerError):
        resource.adapter.generate("question")
    assert len(resource.transport.requests) == expected_attempts


def test_real_sdk_recovers_from_503_then_200(build_adapter):
    responses = iter([error_response(503), success_response()])
    resource = build_adapter(lambda request: next(responses), max_retries=2)
    assert resource.adapter.generate("question") == "ok"
    assert len(resource.transport.requests) == 2


def test_real_sdk_does_not_retry_ordinary_400(build_adapter):
    resource = build_adapter(lambda request: error_response(400), max_retries=2)
    with pytest.raises(openai.BadRequestError):
        resource.adapter.generate("question")
    assert len(resource.transport.requests) == 1


def test_partial_stream_read_failure_is_not_replayed_and_releases_resources(build_adapter):
    failure = httpx2.ReadError("injected failure after one token")

    class BrokenBody(httpx2.SyncByteStream):
        close_calls = 0

        def __iter__(self):
            payload = {
                "id": "chatcmpl-contract", "object": "chat.completion.chunk",
                "created": 0, "model": MODEL,
                "choices": [{"index": 0, "delta": {"content": "first"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
            raise failure

        def close(self):
            self.close_calls += 1

    body = BrokenBody()
    resource = build_adapter(lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body
    ), max_retries=2)
    chunks = resource.adapter.stream("question")
    assert next(chunks) == "first"
    with pytest.raises(openai.APIConnectionError) as raised:
        next(chunks)
    assert raised.value.__cause__ is failure
    assert len(resource.transport.requests) == 1
    assert body.close_calls == 1
    assert resource.transport.responses[0].is_closed
    resource.adapter.close()
    assert resource.client.is_closed
    assert resource.transport.close_calls == 1
