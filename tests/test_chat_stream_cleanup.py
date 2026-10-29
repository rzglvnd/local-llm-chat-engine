import asyncio
import json
import threading
from types import SimpleNamespace

import anyio
import httpx
import pytest

import app as api


class TrackedStream:
    def __init__(self, items):
        self.items = iter(items)
        self.close_calls = 0
        self.close_thread = None
        self.close_error = None
        self.reading = False

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.items)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        assert not self.reading, "close raced with an active provider read"
        self.close_calls += 1
        self.close_thread = threading.get_ident()
        if self.close_error:
            raise self.close_error


def configure_stream(monkeypatch, source):
    calls = []

    def stream(message, context):
        calls.append((message, context))
        return source

    monkeypatch.setattr(api, "_resolve_adapter", lambda name: SimpleNamespace(stream=stream))
    monkeypatch.setattr(api.store, "search", lambda query, k: [{"text": "context"}])
    return calls


def exception_contains(error, target):
    if error is target:
        return True
    children = list(getattr(error, "exceptions", ()))
    if error.__cause__ is not None:
        children.append(error.__cause__)
    elif error.__context__ is not None:
        children.append(error.__context__)
    return any(exception_contains(child, target) for child in children)


@pytest.mark.parametrize("ending, spec_version", [
    ("complete", "2.0"), ("complete", "2.4"),
    ("iteration_error", "2.0"), ("iteration_error", "2.4"),
    ("send_failure", "2.0"), ("send_failure", "2.4"),
    ("disconnect", "2.0"),
])
def test_stream_response_teardown_closes_adapter(monkeypatch, spec_version, ending):
    failure = RuntimeError("provider failed")
    send_failure = OSError("client connection closed")
    source = TrackedStream(["one", failure if ending == "iteration_error" else "two"])
    calls = configure_stream(monkeypatch, source)

    async def scenario():
        loop_thread = threading.get_ident()
        response = await api.chat_stream(api.ChatRequest(message="question", model="openai", k=2))
        disconnect = anyio.Event()
        events = []

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                assert message["status"] == 200
                assert (b"content-type", b"text/event-stream; charset=utf-8") in message["headers"]
            body = message.get("body", b"")
            if not body:
                return
            event = json.loads(body.decode().removeprefix("data: ").strip())
            events.append(event)
            if event["event"] == "chunk":
                if ending == "send_failure":
                    raise send_failure
                if ending == "disconnect":
                    disconnect.set()
                    await anyio.sleep_forever()

        scope = {"type": "http", "asgi": {"spec_version": spec_version}}
        with anyio.fail_after(5):
            if ending in {"iteration_error", "send_failure"}:
                with pytest.raises(Exception) as raised:
                    await response(scope, receive, send)
                original = failure if ending == "iteration_error" else send_failure
                assert exception_contains(raised.value, original)
            else:
                await response(scope, receive, send)

        assert source.close_calls == 1
        assert source.close_thread != loop_thread
        assert calls == [("question", ["context"])]
        assert events[:2] == [
            {"event": "start", "model": "openai"}, {"event": "chunk", "text": "one"}
        ]
        if ending == "complete":
            assert events[2:] == [{"event": "chunk", "text": "two"}, {"event": "done"}]
        else:
            assert len(events) == 2
        # Keep both response and source alive through the cleanup assertions.
        assert response is not None

    anyio.run(scenario)


@pytest.mark.parametrize("cancellation", ["scope", "task"])
def test_cancellation_waits_for_active_read_without_blocking_event_loop(monkeypatch, cancellation):
    async def scenario():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        entered = asyncio.Event()
        release = threading.Event()
        watchdog_expired = threading.Event()

        class BlockingStream(TrackedStream):
            def __next__(self):
                self.reading = True
                loop.call_soon_threadsafe(entered.set)
                try:
                    if not release.wait(timeout=5):
                        watchdog_expired.set()
                    return super().__next__()
                finally:
                    self.reading = False

        source = BlockingStream(["one"])
        configure_stream(monkeypatch, source)
        response = await api.chat_stream(api.ChatRequest(message="question", model="openai"))
        cancel_scope = anyio.CancelScope()

        async def receive():
            await anyio.sleep_forever()

        async def send(message):
            pass

        async def invoke():
            with cancel_scope:
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

        task = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            if cancellation == "scope":
                cancel_scope.cancel()
            else:
                task.cancel()
            # A real request must still complete while read/cleanup is waiting.
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
            assert health.status_code == 200
            assert not watchdog_expired.is_set()
            assert source.reading
            assert source.close_calls == 0
            assert not task.done()
        finally:
            release.set()
            result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
        if cancellation == "task":
            assert isinstance(result[0], asyncio.CancelledError)
        else:
            assert result == [None]
        assert source.close_calls == 1
        assert source.close_thread != loop_thread
        assert not source.reading

    anyio.run(scenario)


@pytest.mark.parametrize("iteration_fails", [False, True])
def test_adapter_cleanup_failure_preserves_result(monkeypatch, caplog, iteration_fails):
    failure = RuntimeError("provider failed")
    source = TrackedStream([failure] if iteration_fails else ["one"])
    source.close_error = OSError("close failed")
    configure_stream(monkeypatch, source)

    async def scenario():
        response = await api.chat_stream(api.ChatRequest(message="question", model="openai"))

        async def receive():
            await anyio.sleep_forever()

        async def send(message):
            pass

        if iteration_fails:
            with pytest.raises(RuntimeError) as raised:
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            assert raised.value is failure
        else:
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert source.close_calls == 1
        assert "Failed to close chat adapter stream" in caplog.text

    anyio.run(scenario)


@pytest.mark.parametrize("ending", ["complete", "send_failure", "disconnect"])
def test_stream_cleanup_through_application_middleware(monkeypatch, ending):
    source = TrackedStream(["one"] * (2 if ending == "complete" else 1000))
    configure_stream(monkeypatch, source)

    async def scenario():
        request_sent = False
        disconnect = anyio.Event()
        disconnect_received = anyio.Event()
        events = []
        failure = OSError("send failed")

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": json.dumps({
                    "message": "question", "model": "openai"
                }).encode(), "more_body": False}
            await disconnect.wait()
            disconnect_received.set()
            return {"type": "http.disconnect"}

        async def send(message):
            body = message.get("body", b"")
            if not body:
                return
            event = json.loads(body.decode().removeprefix("data: ").strip())
            events.append(event)
            if event["event"] == "chunk":
                if ending == "send_failure":
                    raise failure
                if ending == "disconnect":
                    disconnect.set()
                    await disconnect_received.wait()

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/chat_stream", "raw_path": b"/chat_stream", "query_string": b"",
            "root_path": "", "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345), "server": ("test", 80),
        }
        with anyio.fail_after(5):
            if ending == "send_failure":
                with pytest.raises(Exception) as raised:
                    await api.app(scope, receive, send)
                assert exception_contains(raised.value, failure)
            else:
                await api.app(scope, receive, send)
        assert source.close_calls == 1
        if ending == "complete":
            assert [event["event"] for event in events] == ["start", "chunk", "chunk", "done"]
        else:
            assert "done" not in [event["event"] for event in events]

    anyio.run(scenario)


@pytest.mark.parametrize("backend", ["echo", "iterator_without_close"])
def test_streaming_fallbacks_remain_compatible(monkeypatch, backend):
    if backend == "iterator_without_close":
        configure_stream(monkeypatch, iter(["", "one"]))

    async def scenario():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/chat_stream", json={"message": "question"})
        assert response.status_code == 200
        events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
                  if line.startswith("data: ")]
        assert events[0] == {"event": "start", "model": "echo"}
        assert events[-1] == {"event": "done"}
        text = "".join(event["text"] for event in events if event["event"] == "chunk")
        if backend == "iterator_without_close":
            assert text == "one"
        else:
            assert "question" in text

    anyio.run(scenario)
