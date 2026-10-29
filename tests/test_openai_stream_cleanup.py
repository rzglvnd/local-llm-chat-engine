from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adapters import openai_adapter


class FakeStream:
    def __init__(self):
        self.items = iter(())
        self.close_calls = 0
        self.close_error = None

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.items)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture(params=["modern", "legacy"])
def provider(request, monkeypatch):
    stream = FakeStream()
    create = Mock(return_value=stream)
    client_close = Mock()
    if request.param == "modern":
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=client_close,
        )
        sdk = SimpleNamespace(OpenAI=Mock(return_value=client))

        def chunk(text):
            return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
    else:
        sdk = SimpleNamespace(ChatCompletion=SimpleNamespace(create=create), close=client_close)

        def chunk(text):
            return {"choices": [{"delta": {"content": text}}]}

    monkeypatch.setattr(openai_adapter, "openai", sdk)
    monkeypatch.setattr(openai_adapter, "OPENAI_AVAILABLE", True)
    adapter = openai_adapter.OpenAIAdapter(api_key="fake-key", model="fake-model")
    return SimpleNamespace(
        adapter=adapter, stream=stream, create=create, client_close=client_close, chunk=chunk
    )


@pytest.mark.parametrize("ending", ["exhaustion", "iteration_error", "explicit_close"])
def test_stream_closes_owned_response(provider, ending):
    failure = RuntimeError("provider iteration failed")
    last_item = failure if ending == "iteration_error" else provider.chunk("second")
    provider.stream.items = iter([provider.chunk(""), provider.chunk("first"), last_item])
    chunks = provider.adapter.stream("question", context=["context"], temperature=0)
    assert next(chunks) == "first"
    assert provider.stream.close_calls == 0
    if ending == "explicit_close":
        chunks.close()
    elif ending == "iteration_error":
        with pytest.raises(RuntimeError) as raised:
            next(chunks)
        assert raised.value is failure
    else:
        assert list(chunks) == ["second"]

    assert provider.stream.close_calls == 1
    chunks.close()
    assert provider.stream.close_calls == 1
    provider.client_close.assert_not_called()
    provider.create.assert_called_once_with(
        model="fake-model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "Context:\ncontext"},
            {"role": "user", "content": "question"},
        ],
        stream=True,
        temperature=0,
    )


def test_stream_supports_provider_iterators_without_close(provider):
    provider.stream.close = None
    provider.stream.items = iter([provider.chunk("text")])
    assert list(provider.adapter.stream("question")) == ["text"]
    provider.client_close.assert_not_called()


@pytest.mark.parametrize("iteration_fails", [False, True])
def test_cleanup_failure_is_logged_without_changing_stream_result(
    provider, caplog, iteration_fails
):
    failure = RuntimeError("provider iteration failed")
    provider.stream.close_error = OSError("response close failed")
    provider.stream.items = iter([failure] if iteration_fails else [provider.chunk("text")])
    if iteration_fails:
        with pytest.raises(RuntimeError) as raised:
            list(provider.adapter.stream("question"))
        assert raised.value is failure
    else:
        assert list(provider.adapter.stream("question")) == ["text"]
    assert provider.stream.close_calls == 1
    assert "Failed to close provider stream" in caplog.text
    provider.client_close.assert_not_called()


@pytest.mark.parametrize("provider", ["legacy"], indirect=True)
def test_legacy_finish_reason_closes_response_before_exhaustion(provider):
    provider.stream.items = iter([
        {"choices": [{"delta": {"content": "delta"}, "text": "text", "finish_reason": "stop"}]},
        RuntimeError("must not read beyond finish_reason"),
    ])
    assert list(provider.adapter.stream("question")) == ["delta", "text"]
    assert provider.stream.close_calls == 1
    provider.client_close.assert_not_called()
