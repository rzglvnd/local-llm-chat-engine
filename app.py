from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from store import DocumentStore
from engine import EchoModel
import os
import json
from fastapi.responses import StreamingResponse

# embedding/FAISS optional backends
try:
    from faiss_store import FaissStore
    faiss_available = True
    try:
        faiss_store = FaissStore()
    except Exception:
        faiss_store = None
        faiss_available = False
except Exception:
    faiss_store = None
    faiss_available = False

try:
    from embeddings import EmbeddingStore
    embeddings_available = True
    embedding_store = None
    try:
        embedding_store = EmbeddingStore()
    except Exception:
        embedding_store = None
        embeddings_available = False
except Exception:
    embedding_store = None
    embeddings_available = False

# model adapters
try:
    from adapters.openai_adapter import OpenAIAdapter
    openai_adapter_available = True
except Exception:
    OpenAIAdapter = None
    openai_adapter_available = False

try:
    from adapters.huggingface_adapter import HuggingFaceAdapter
    hf_adapter_available = True
except Exception:
    HuggingFaceAdapter = None
    hf_adapter_available = False

app = FastAPI(title="Local LLM Chat Engine")

store = DocumentStore()
model = EchoModel()


class ChatRequest(BaseModel):
    message: str
    k: Optional[int] = 3


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(payload: dict):
    documents = payload.get("documents")
    if not documents:
        raise HTTPException(status_code=400, detail="Missing 'documents' list")
    res = store.ingest(documents)
    return res


@app.post("/ingest_embeddings")
async def ingest_embeddings(payload: dict):
    if not embeddings_available or embedding_store is None:
        raise HTTPException(status_code=400, detail="Embeddings not available. Install sentence-transformers.")
    documents = payload.get("documents")
    if not documents:
        raise HTTPException(status_code=400, detail="Missing 'documents' list")
    res = embedding_store.ingest(documents)
    return res


@app.post("/search")
async def search(payload: dict):
    query = payload.get("query")
    k = int(payload.get("k", 5))
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    return {"results": store.search(query, k=k)}


@app.post("/search_embeddings")
async def search_embeddings(payload: dict):
    if not embeddings_available or embedding_store is None:
        raise HTTPException(status_code=400, detail="Embeddings not available. Install sentence-transformers.")
    query = payload.get("query")
    k = int(payload.get("k", 5))
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    return {"results": embedding_store.search(query, k=k)}


@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.message
    k = req.k or 3
    contexts = [r["text"] for r in store.search(query, k=k)]
    resp = model.generate(query, context=contexts)
    return {"response": resp, "context_count": len(contexts)}


def _sse_encode(obj: dict) -> str:
    # Minimal SSE event encoding with JSON payload under `data:`
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/chat_stream")
async def chat_stream(payload: dict):
    message = payload.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")
    model_name = payload.get("model", "echo")
    k = int(payload.get("k", 3))

    # choose adapter
    adapter = None
    if model_name == "openai":
        if not openai_adapter_available:
            raise HTTPException(status_code=400, detail="OpenAI adapter not available")
        adapter = OpenAIAdapter(api_key=os.environ.get("OPENAI_API_KEY"))
    elif model_name == "hf":
        if not hf_adapter_available:
            raise HTTPException(status_code=400, detail="HuggingFace adapter not available")
        adapter = HuggingFaceAdapter()
    else:
        adapter = model

    # get context from store
    contexts = [r["text"] for r in store.search(message, k=k)]

    def event_stream():
        # adapter may be an EchoModel with no stream() method
        if hasattr(adapter, "stream"):
            for chunk in adapter.stream(message, context=contexts):
                yield _sse_encode({"text": chunk})
        else:
            # fallback to generate then stream in chunks
            text = adapter.generate(message, context=contexts)
            for i in range(0, len(text), 64):
                yield _sse_encode({"text": text[i : i + 64]})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest_faiss")
async def ingest_faiss(payload: dict):
    if not faiss_available or faiss_store is None:
        raise HTTPException(status_code=400, detail="FAISS store not available. Install faiss and sentence-transformers.")
    documents = payload.get("documents")
    if not documents:
        raise HTTPException(status_code=400, detail="Missing 'documents' list")
    return faiss_store.ingest(documents)


@app.post("/search_faiss")
async def search_faiss(payload: dict):
    if not faiss_available or faiss_store is None:
        raise HTTPException(status_code=400, detail="FAISS store not available. Install faiss and sentence-transformers.")
    query = payload.get("query")
    k = int(payload.get("k", 5))
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    return {"results": faiss_store.search(query, k=k)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

