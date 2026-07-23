"""OpenAI adapter with optional streaming support."""
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
        if api_key:
            openai.api_key = api_key
        # rely on environment variable OPENAI_API_KEY if not provided
        self.model = model

    def generate(self, prompt: str, context=None, max_tokens: int = 256, **kwargs) -> str:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context:
            messages.append({"role": "assistant", "content": "Context:\n" + "\n".join(context)})
        messages.append({"role": "user", "content": prompt})
        resp = openai.ChatCompletion.create(model=self.model, messages=messages, max_tokens=max_tokens, **kwargs)
        # best-effort parsing across SDK versions
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            try:
                return resp.choices[0].message.content
            except Exception:
                return str(resp)

    def stream(self, prompt: str, context=None, **kwargs):
        """Yields text chunks from the OpenAI streaming API.

        Each yielded value is a str containing a fragment of generated text.
        """
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context:
            messages.append({"role": "assistant", "content": "Context:\n" + "\n".join(context)})
        messages.append({"role": "user", "content": prompt})

        for chunk in openai.ChatCompletion.create(model=self.model, messages=messages, stream=True, **kwargs):
            # chunk is typically a dict with choices[].delta.content
            try:
                choices = chunk.get("choices", [])
            except Exception:
                choices = []
            for c in choices:
                delta = c.get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
                # handle older SDK shapes
                text2 = c.get("text")
                if text2:
                    yield text2
                if c.get("finish_reason"):
                    return
