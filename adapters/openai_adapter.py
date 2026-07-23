"""OpenAI adapter with optional streaming support."""
import os

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False


class OpenAIAdapter:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed")

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._legacy_client = None
        self._v1_client = None

        if hasattr(openai, "OpenAI"):
            self._v1_client = openai.OpenAI(api_key=self._api_key)
        else:
            if self._api_key:
                openai.api_key = self._api_key
            self._legacy_client = openai

        self.model = model

    @staticmethod
    def _messages(prompt: str, context=None):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context:
            messages.append({"role": "assistant", "content": "Context:\n" + "\n".join(context)})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, context=None, max_tokens: int = 256, **kwargs) -> str:
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
        """
        messages = self._messages(prompt, context=context)

        if self._v1_client is not None:
            stream = self._v1_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
            return

        for chunk in self._legacy_client.ChatCompletion.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs,
        ):
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
