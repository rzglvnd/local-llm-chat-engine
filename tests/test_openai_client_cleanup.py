import asyncio
import json
import threading
from types import SimpleNamespace
from dataclasses import replace

import httpx
import pytest

import app as api
from adapters import openai_adapter


@pytest.fixture(params=["modern", "legacy"])
def provider(request, monkeypatch):
    state = SimpleNamespace(
        kind=request.param, calls=[], generation_error=None, close_error=None,
        requests=[], client_timeout=None, client_retries=None, stream_error=None,
    )

    def record(event):
        state.calls.append((event, threading.get_ident()))

    def create(**kwargs):
        state.requests.append(kwargs)
        record("stream" if kwargs.get("stream") else "generate")
        if state.generation_error:
            raise state.generation_error
        if kwargs.get("stream"):
            def chunks():
                try:
                    if state.kind == "modern":
                        yield SimpleNamespace(choices=[SimpleNamespace(
                            delta=SimpleNamespace(content="fake answer")
                        )])
                    else:
                        yield {"choices": [{"delta": {"content": "fake answer"}}]}
                    if state.stream_error:
                        raise state.stream_error
                finally:
                    record("close_stream")

            state.response = chunks()
            return state.response
        if state.kind == "modern":
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="fake answer")
            )])
        return {"choices": [{"message": {"content": "fake answer"}}]}

    def close():
        record("close" if state.kind == "modern" else "close_borrowed")
        if state.close_error:
            raise state.close_error

    def new_client(api_key=None, timeout=None, max_retries=None):
        state.client_timeout = timeout
        state.client_retries = max_retries
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close
        )

    sdk = (SimpleNamespace(OpenAI=new_client) if state.kind == "modern" else
           SimpleNamespace(ChatCompletion=SimpleNamespace(create=create), close=close))
    sdk.api_key = "borrowed-key-sentinel"
    state.sdk = sdk
    monkeypatch.setattr(openai_adapter, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(openai_adapter, "openai", sdk)

    def make_adapter(**kwargs):
        record("initialize")
        kwargs["api_key"] = "fake-key"
        return openai_adapter.OpenAIAdapter(**kwargs)

    state.make_adapter = make_adapter
    monkeypatch.setattr(api, "openai_adapter_available", True)
    monkeypatch.setattr(api, "OpenAIAdapter", make_adapter)
    monkeypatch.setattr(api.store, "search", lambda query, k: [])
    return state


@pytest.mark.parametrize("generation_fails", [False, True])
def test_chat_releases_owned_client_in_its_generation_worker(provider, generation_fails):
    if generation_fails:
        provider.generation_error = RuntimeError("generation failed")

    async def scenario():
        loop_thread = threading.get_ident()
        transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/chat", json={"message": "question", "model": "openai"})
        if generation_fails:
            assert response.status_code == 500
            assert response.text == "Internal Server Error"
        else:
            assert response.status_code == 200
            assert response.json() == {
                "model": "openai", "response": "fake answer", "context_count": 0
            }
        expected = ["initialize", "generate"] + (["close"] if provider.kind == "modern" else [])
        assert [event for event, _ in provider.calls] == expected
        worker_threads = {thread for _, thread in provider.calls}
        assert len(worker_threads) == 1
        assert loop_thread not in worker_threads

    asyncio.run(scenario())


def test_adapter_close_is_idempotent_and_preserves_borrowed_module(provider):
    adapter = provider.make_adapter()
    adapter.close()
    adapter.close()
    expected = ["initialize", "close"] if provider.kind == "modern" else ["initialize"]
    assert [event for event, _ in provider.calls] == expected


@pytest.mark.parametrize("operation", ["generate", "stream"])
def test_closed_adapter_rejects_new_work(provider, operation):
    adapter = provider.make_adapter()
    adapter.close()
    calls_after_close = list(provider.calls)
    with pytest.raises(RuntimeError, match="OpenAI adapter is closed"):
        if operation == "generate":
            adapter.generate("question")
        else:
            list(adapter.stream("question"))
    assert provider.calls == calls_after_close


@pytest.mark.parametrize("provider", ["modern"], indirect=True)
@pytest.mark.parametrize("generation_fails", [False, True])
def test_chat_cleanup_failure_is_logged_without_masking_generation(
    provider, caplog, generation_fails
):
    provider.close_error = OSError("client close failed")
    if generation_fails:
        provider.generation_error = RuntimeError("original generation failed")
    if generation_fails:
        with pytest.raises(RuntimeError) as raised:
            api._generate_chat_response("openai", "question", [])
        assert raised.value is provider.generation_error
    else:
        assert api._generate_chat_response("openai", "question", []) == "fake answer"
    assert [event for event, _ in provider.calls] == ["initialize", "generate", "close"]
    assert "Failed to close chat adapter client" in caplog.text


@pytest.mark.parametrize("provider", ["modern"], indirect=True)
def test_failed_close_is_attempted_once_and_leaves_adapter_closed(provider):
    adapter = provider.make_adapter()
    provider.close_error = OSError("client close failed")
    with pytest.raises(OSError) as raised:
        adapter.close()
    assert raised.value is provider.close_error
    adapter.close()
    with pytest.raises(RuntimeError, match="OpenAI adapter is closed"):
        adapter.generate("question")
    assert [event for event, _ in provider.calls] == ["initialize", "close"]


@pytest.mark.parametrize("path", ["/chat", "/chat_stream"])
def test_chat_never_closes_shared_echo_model(monkeypatch, path):
    calls = []
    monkeypatch.setattr(api.model, "close", lambda: calls.append("close"), raising=False)

    async def scenario():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, json={"message": "question", "model": "echo"})
        assert response.status_code == 200
        assert "question" in response.text
        assert calls == []

    asyncio.run(scenario())


def test_streaming_request_closes_response_before_owned_sdk_client(provider):
    async def scenario():
        loop_thread = threading.get_ident()
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat_stream", json={"message": "question", "model": "openai"}
            )
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ")]
        assert events == [
            {"event": "start", "model": "openai"},
            {"event": "chunk", "text": "fake answer"},
            {"event": "done"},
        ]
        expected = ["initialize", "stream", "close_stream"]
        if provider.kind == "modern":
            expected.append("close")
        assert [event for event, _ in provider.calls] == expected
        assert all(thread != loop_thread for _, thread in provider.calls)
        # Retain the provider iterator through the assertions.
        assert provider.response is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["/chat", "/chat_stream"])
def test_configured_timeout_reaches_provider_from_api(provider, monkeypatch, path):
    monkeypatch.setattr(api, "settings", replace(api.settings, openai_timeout_seconds=12.5))

    async def scenario():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, json={"message": "question", "model": "openai"})
        assert response.status_code == 200
        assert len(provider.requests) == 1
        if provider.kind == "modern":
            assert provider.client_timeout == 12.5
            assert "request_timeout" not in provider.requests[0]
        else:
            assert provider.requests[0]["request_timeout"] == 12.5
            assert "timeout" not in provider.requests[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["/chat", "/chat_stream"])
def test_simulated_provider_timeout_releases_resources(provider, monkeypatch, path):
    retry_limit = 3 if provider.kind == "modern" else None
    monkeypatch.setattr(api, "settings", replace(api.settings, openai_max_retries=retry_limit))
    failure = TimeoutError("simulated provider I/O timeout")
    if path == "/chat":
        provider.generation_error = failure
    else:
        provider.stream_error = failure

    async def scenario():
        transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, json={"message": "question", "model": "openai"})
        expected = ["initialize"]
        if path == "/chat":
            assert response.status_code == 500
            expected.append("generate")
        else:
            # Headers and a token were sent before the simulated read timeout.
            assert response.status_code == 200
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ")]
            assert events == [
                {"event": "start", "model": "openai"},
                {"event": "chunk", "text": "fake answer"},
            ]
            expected += ["stream", "close_stream"]
        if provider.kind == "modern":
            expected.append("close")
        assert [event for event, _ in provider.calls] == expected
        # The application invokes the SDK once even after partial stream output.
        # Internal HTTP retry attempts belong to the SDK and are not simulated.
        assert len(provider.requests) == 1
        assert "max_retries" not in provider.requests[0]
        assert provider.client_retries == retry_limit

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["generate", "stream"])
def test_native_request_timeout_override_is_preserved(provider, operation):
    adapter = provider.make_adapter(timeout_seconds=12.5)
    key = "timeout" if provider.kind == "modern" else "request_timeout"
    try:
        if operation == "generate":
            assert adapter.generate("question", **{key: 0.75}) == "fake answer"
        else:
            assert list(adapter.stream("question", **{key: 0.75})) == ["fake answer"]
        assert provider.requests[0][key] == 0.75
        if provider.kind == "modern":
            assert provider.client_timeout == 12.5
    finally:
        adapter.close()


@pytest.mark.parametrize("provider", ["modern"], indirect=True)
@pytest.mark.parametrize("path", ["/chat", "/chat_stream"])
@pytest.mark.parametrize("limit", [0, 3])
def test_configured_retry_limit_reaches_modern_sdk(provider, monkeypatch, path, limit):
    monkeypatch.setattr(api, "settings", replace(api.settings, openai_max_retries=limit))

    async def scenario():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, json={"message": "question", "model": "openai"})
        assert response.status_code == 200
        assert provider.client_retries == limit
        assert len(provider.requests) == 1
        assert "max_retries" not in provider.requests[0]
        assert provider.calls[-1][0] == "close"

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["legacy"], indirect=True)
@pytest.mark.parametrize("path", ["/chat", "/chat_stream"])
@pytest.mark.parametrize("limit", [0, 2])
def test_legacy_rejects_explicit_retry_setting_before_requests(provider, monkeypatch, path, limit):
    monkeypatch.setattr(api, "settings", replace(api.settings, openai_max_retries=limit))

    async def scenario():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, json={"message": "question", "model": "openai"})
        assert response.status_code == 400
        assert response.headers["content-type"] == "application/json"
        assert "max_retries requires the modern OpenAI SDK" in response.json()["detail"]
        assert [event for event, _ in provider.calls] == ["initialize"]
        assert provider.requests == []
        assert provider.sdk.api_key == "borrowed-key-sentinel"

    asyncio.run(scenario())
