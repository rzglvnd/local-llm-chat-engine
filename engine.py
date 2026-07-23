from typing import Optional, List


class BaseModel:
    def generate(self, prompt: str, context: Optional[List[str]] = None) -> str:
        raise NotImplementedError()


class EchoModel(BaseModel):
    def generate(self, prompt: str, context: Optional[List[str]] = None) -> str:
        ctx = "\n\n--context--\n" + "\n---\n".join(context) if context else ""
        return f"[EchoModel]\nPrompt: {prompt}\n{ctx}\nResponse: (this is a stub)"


class OpenAIModel(BaseModel):
    """Optional adapter for OpenAI — uses the `openai` package when available.

    This adapter is intentionally optional. If `openai` is not installed or
    `OPENAI_API_KEY` is not set, you can fall back to `EchoModel`.
    """

    def __init__(self, client=None):
        try:
            import openai
            self.openai = client or openai
        except Exception:
            self.openai = None

    def generate(self, prompt: str, context: Optional[List[str]] = None) -> str:
        if not self.openai:
            return "OpenAI client not available — install `openai` and set OPENAI_API_KEY"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if context:
            messages.insert(1, {"role": "assistant", "content": "Context:\n" + "\n".join(context)})
        resp = self.openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
        return resp.choices[0].message.content
