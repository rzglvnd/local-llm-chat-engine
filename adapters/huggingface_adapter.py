"""Simple HuggingFace adapter with a best-effort streaming wrapper."""
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    pipeline = None
    HF_AVAILABLE = False


class HuggingFaceAdapter:
    def __init__(self, model_name: str = "gpt2", device: int = -1):
        if not HF_AVAILABLE:
            raise ImportError("transformers not installed")
        self.model_name = model_name
        # pipeline will download model on first use; device=-1 selects CPU
        self.generator = pipeline("text-generation", model=model_name, device=device)

    def generate(self, prompt: str, context=None, max_new_tokens: int = 128, **kwargs) -> str:
        input_text = prompt if not context else "\n".join(context) + "\n\n" + prompt
        outputs = self.generator(
            input_text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            **kwargs,
        )
        if outputs and isinstance(outputs, list):
            return outputs[0].get("generated_text", "")
        return str(outputs)

    def stream(self, prompt: str, context=None, chunk_size: int = 64, **kwargs):
        # Best-effort streaming: generate full text then yield chunks
        text = self.generate(prompt, context=context, **kwargs)
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
