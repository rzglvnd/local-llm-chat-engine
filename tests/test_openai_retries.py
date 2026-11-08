from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adapters import openai_adapter
from settings import load_settings


def test_retry_setting_distinguishes_unset_from_zero(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_OPENAI_MAX_RETRIES", raising=False)
    assert load_settings().openai_max_retries is None
    monkeypatch.setenv("LOCAL_LLM_OPENAI_MAX_RETRIES", "0")
    assert load_settings().openai_max_retries == 0


@pytest.mark.parametrize("value, expected", [("0", 0), ("2", 2), (" 5 ", 5)])
def test_retry_setting_accepts_non_negative_integers(monkeypatch, value, expected):
    monkeypatch.setenv("LOCAL_LLM_OPENAI_MAX_RETRIES", value)
    assert load_settings().openai_max_retries == expected


@pytest.mark.parametrize("value", ["", "-1", "2.5", "nan", "inf", "invalid"])
def test_retry_setting_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("LOCAL_LLM_OPENAI_MAX_RETRIES", value)
    with pytest.raises(ValueError, match="LOCAL_LLM_OPENAI_MAX_RETRIES.*non-negative integer"):
        load_settings()


@pytest.mark.parametrize("value", [True, -1, 2.0, "1.5", float("inf")])
def test_adapter_validates_retry_limit_before_client_construction(monkeypatch, value):
    constructor = Mock()
    monkeypatch.setattr(openai_adapter, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(openai_adapter, "openai", SimpleNamespace(OpenAI=constructor))
    with pytest.raises(ValueError, match="max_retries.*non-negative integer"):
        openai_adapter.OpenAIAdapter(api_key="fake-key", max_retries=value)
    constructor.assert_not_called()


def test_modern_client_has_explicit_default_retry_limit(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(openai_adapter, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(openai_adapter, "openai", SimpleNamespace(OpenAI=constructor))
    adapter = openai_adapter.OpenAIAdapter(api_key="fake-key")
    try:
        constructor.assert_called_once_with(api_key="fake-key", timeout=60.0, max_retries=2)
    finally:
        adapter.close()
