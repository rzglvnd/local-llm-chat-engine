import importlib
import pytest


def test_openai_adapter_import_and_api_presence():
    """Import-check for OpenAI adapter; skip if dependency not present."""
    try:
        mod = importlib.import_module("local_llm_chat_engine.adapters.openai_adapter")
    except Exception:
        pytest.skip("OpenAI adapter not available (openai package missing)")
    OpenAIAdapter = getattr(mod, "OpenAIAdapter", None)
    assert OpenAIAdapter is not None
    assert callable(getattr(OpenAIAdapter, "generate", None))
    assert callable(getattr(OpenAIAdapter, "stream", None))


def test_huggingface_adapter_import_and_api_presence():
    """Import-check for HuggingFace adapter; skip if dependency not present."""
    try:
        mod = importlib.import_module("local_llm_chat_engine.adapters.huggingface_adapter")
    except Exception:
        pytest.skip("HuggingFace adapter not available (transformers package missing)")
    HuggingFaceAdapter = getattr(mod, "HuggingFaceAdapter", None)
    assert HuggingFaceAdapter is not None
    assert callable(getattr(HuggingFaceAdapter, "generate", None))
    assert callable(getattr(HuggingFaceAdapter, "stream", None))
