from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adapters import openai_adapter
from settings import load_settings


def test_openai_timeout_has_finite_default(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_OPENAI_TIMEOUT_SECONDS", raising=False)
    assert load_settings().openai_timeout_seconds == 60.0


@pytest.mark.parametrize("value, expected", [("0.25", 0.25), (" 12.5 ", 12.5)])
def test_openai_timeout_setting_accepts_positive_seconds(monkeypatch, value, expected):
    monkeypatch.setenv("LOCAL_LLM_OPENAI_TIMEOUT_SECONDS", value)
    assert load_settings().openai_timeout_seconds == expected


@pytest.mark.parametrize("value", ["", "invalid", "0", "-1", "nan", "inf", "-inf", "1e309"])
def test_openai_timeout_setting_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("LOCAL_LLM_OPENAI_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="LOCAL_LLM_OPENAI_TIMEOUT_SECONDS.*positive finite"):
        load_settings()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), None, True])
def test_adapter_rejects_invalid_timeout_before_creating_client(monkeypatch, value):
    constructor = Mock()
    monkeypatch.setattr(openai_adapter, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(openai_adapter, "openai", SimpleNamespace(OpenAI=constructor))
    with pytest.raises(ValueError, match="timeout_seconds.*positive finite"):
        openai_adapter.OpenAIAdapter(api_key="fake-key", timeout_seconds=value)
    constructor.assert_not_called()
