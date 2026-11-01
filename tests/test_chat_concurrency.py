import asyncio
import json
import threading

import httpx
import pytest

import app as api


@pytest.mark.parametrize("blocked_phase", ["initialization", "generation"])
@pytest.mark.parametrize("path", ["/chat", "/chat_stream"])
def test_health_responds_while_chat_adapter_work_is_blocked(monkeypatch, blocked_phase, path):
    async def scenario():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        entered = asyncio.Event()
        release = threading.Event()
        returned = threading.Event()
        watchdog_expired = threading.Event()
        execution_threads = {}
        searches = []
        generations = []
        response_starts = []

        def block_adapter():
            loop.call_soon_threadsafe(entered.set)
            try:
                # This watchdog also frees a broken implementation that blocks
                # the event loop itself, where an async timeout cannot run.
                if not release.wait(timeout=5):
                    watchdog_expired.set()
            finally:
                returned.set()

        def search(query, k):
            execution_threads["retrieval"] = threading.get_ident()
            searches.append((query, k))
            return [{"text": "first context"}, {"text": "second context"}]

        class FakeAdapter:
            def generate(self, prompt, context=None):
                execution_threads["generation"] = threading.get_ident()
                generations.append((prompt, context))
                if blocked_phase == "generation":
                    block_adapter()
                return "fake response"

            def stream(self, prompt, context=None):
                yield self.generate(prompt, context=context)

        def create_adapter(api_key=None):
            execution_threads["initialization"] = threading.get_ident()
            if blocked_phase == "initialization":
                block_adapter()
            return FakeAdapter()

        monkeypatch.setattr(api.store, "search", search)
        monkeypatch.setattr(api, "openai_adapter_available", True)
        monkeypatch.setattr(api, "OpenAIAdapter", create_adapter)

        async def observed_app(scope, receive, send):
            async def observed_send(message):
                if scope["path"] == path and message["type"] == "http.response.start":
                    response_starts.append(message)
                await send(message)

            await api.app(scope, receive, observed_send)

        transport = httpx.ASGITransport(app=observed_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat = asyncio.create_task(client.post(
                path, json={"message": "explain", "k": 2, "model": "openai"}
            ))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                assert not watchdog_expired.is_set(), "chat blocked the event loop"
                assert not chat.done()
                if blocked_phase == "initialization":
                    assert response_starts == []
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                # The response must arrive before allowing the adapter to return.
                assert not release.is_set()
                assert not returned.is_set()
                assert not chat.done()
            finally:
                release.set()
                await asyncio.wait_for(
                    asyncio.gather(chat, return_exceptions=True), timeout=5
                )
            response = chat.result()
            assert response.status_code == 200
            if path == "/chat":
                assert response.json() == {
                    "model": "openai", "response": "fake response", "context_count": 2
                }
            else:
                events = [json.loads(line[6:]) for line in response.text.splitlines()
                          if line.startswith("data: ")]
                assert events == [
                    {"event": "start", "model": "openai"},
                    {"event": "chunk", "text": "fake response"},
                    {"event": "done"},
                ]
        assert searches == [("explain", 2)]
        assert generations == [("explain", ["first context", "second context"])]
        assert execution_threads["retrieval"] == loop_thread
        assert execution_threads["initialization"] != loop_thread
        assert execution_threads["generation"] != loop_thread
        if path == "/chat":
            assert execution_threads["generation"] == execution_threads["initialization"]

    asyncio.run(scenario())


@pytest.mark.parametrize("model, prefix, available_flag, adapter_name", [
    ("openai", "OpenAI", "openai_adapter_available", "OpenAIAdapter"),
    ("hf", "HuggingFace", "hf_adapter_available", "HuggingFaceAdapter"),
])
@pytest.mark.parametrize("failure", ["unavailable", "initialization"])
def test_stream_initialization_errors_precede_sse_headers(
    monkeypatch, model, prefix, available_flag, adapter_name, failure
):
    initializations = []

    def create_adapter(**kwargs):
        initializations.append(threading.get_ident())
        raise ValueError("initialization failed")

    monkeypatch.setattr(api, available_flag, failure != "unavailable")
    monkeypatch.setattr(api, adapter_name, create_adapter)

    async def scenario():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat_stream", json={"message": "explain", "model": model}
            )
        assert response.status_code == 400
        assert response.headers["content-type"] == "application/json"
        if failure == "unavailable":
            assert response.json() == {"detail": f"{prefix} adapter not available"}
            assert initializations == []
        else:
            assert response.json() == {
                "detail": f"{prefix} adapter initialization failed: initialization failed"
            }
            assert len(initializations) == 1
            assert initializations[0] != threading.get_ident()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["unavailable", "initialization", "generation"])
def test_chat_preserves_adapter_error_responses(monkeypatch, failure):
    class FailingAdapter:
        def generate(self, prompt, context=None):
            raise RuntimeError("generation failed")

    def create_adapter(api_key=None):
        if failure == "initialization":
            raise ValueError("initialization failed")
        return FailingAdapter()

    monkeypatch.setattr(api.store, "search", lambda query, k: [])
    monkeypatch.setattr(api, "openai_adapter_available", failure != "unavailable")
    monkeypatch.setattr(api, "OpenAIAdapter", create_adapter)

    async def scenario():
        transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat", json={"message": "explain", "model": "openai"}
            )
        if failure == "generation":
            assert response.status_code == 500
            assert response.text == "Internal Server Error"
        else:
            assert response.status_code == 400
            detail = (
                "OpenAI adapter not available" if failure == "unavailable"
                else "OpenAI adapter initialization failed: initialization failed"
            )
            assert response.json() == {"detail": detail}

    asyncio.run(scenario())
