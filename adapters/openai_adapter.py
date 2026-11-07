"""OpenAI adapter with optional streaming support."""
import logging
import os

from settings import DEFAULT_OPENAI_TIMEOUT_SECONDS, validate_openai_timeout_seconds

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False


logger = logging.getLogger(__name__)


def _close_stream(stream) -> None:
    try:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    except Exception:
        # Cleanup must not replace a provider error or a completed response.
        logger.exception("Failed to close provider stream")


class OpenAIAdapter:
    def __init__(
        self, api_key: str = None, model: str = "gpt-4o-mini",
        timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
    ):
        self._timeout_seconds = validate_openai_timeout_seconds(timeout_seconds)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed")

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._legacy_client = None
        self._v1_client = None
        self._closed = False

        if hasattr(openai, "OpenAI"):
            self._v1_client = openai.OpenAI(
                api_key=self._api_key, timeout=self._timeout_seconds
            )
        else:
            if self._api_key:
                openai.api_key = self._api_key
            self._legacy_client = openai

        self.model = model

    def close(self) -> None:
        """Release this adapter's owned client after active work has finished."""
        if self._closed:
            return
        # A failed close is still terminal; never reuse a partly closed client.
        self._closed = True
        if self._v1_client is not None:
            self._v1_client.close()

    @staticmethod
    def _messages(prompt: str, context=None):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context:
            messages.append({"role": "assistant", "content": "Context:\n" + "\n".join(context)})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, context=None, max_tokens: int = 256, **kwargs) -> str:
        if self._closed:
            raise RuntimeError("OpenAI adapter is closed")
        messages = self._messages(prompt, context=context)

        if self._v1_client is not None:
            resp = self._v1_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
            text = resp.choices[0].message.content
            return text or ""

        kwargs.setdefault("request_timeout", self._timeout_seconds)
        resp = self._legacy_client.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp["choices"][0]["message"]["content"]

    def stream(self, prompt: str, context=None, **kwargs):
        """Yields text chunks from the OpenAI streaming API.

        Each yielded value is a str containing a fragment of generated text.
        Close this generator explicitly if consuming only part of its output.
        """
        if self._closed:
            raise RuntimeError("OpenAI adapter is closed")
        messages = self._messages(prompt, context=context)

        if self._v1_client is not None:
            stream = self._v1_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            try:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    text = getattr(delta, "content", None)
                    if text:
                        yield text
            finally:
                _close_stream(stream)
            return

        kwargs.setdefault("request_timeout", self._timeout_seconds)
        stream = self._legacy_client.ChatCompletion.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        try:
            for chunk in stream:
                choices = chunk.get("choices", [])
                for c in choices:
                    delta = c.get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield text
                    text2 = c.get("text")
                    if text2:
                        yield text2
                    if c.get("finish_reason"):
                        return
        finally:
            _close_stream(stream)
