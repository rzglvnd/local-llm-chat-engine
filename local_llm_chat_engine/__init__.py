"""Compatibility package for local_llm_chat_engine imports."""

from .store import DocumentStore
from .engine import BaseModel, EchoModel, OpenAIModel

__all__ = ["DocumentStore", "BaseModel", "EchoModel", "OpenAIModel"]
