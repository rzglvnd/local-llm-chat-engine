import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine import EchoModel
from rate_limit import InMemoryRateLimiter
from settings import load_settings
from store import DocumentStore

settings = load_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("local_llm_chat_engine")

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

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

store = DocumentStore()
model = EchoModel()

app.state.rate_limiter = InMemoryRateLimiter(
    limit=settings.rate_limit_requests_per_minute,
    window_seconds=settings.rate_limit_window_seconds,
    enabled=settings.rate_limit_enabled,
)
rate_limit_exempt_paths = set(settings.rate_limit_exempt_paths or [])
rate_limit_exempt_paths.update({"/health", "/ready"})
app.state.rate_limit_exempt_paths = rate_limit_exempt_paths


@app.on_event("startup")
async def startup_load_snapshot() -> None:
    if not settings.autoload_snapshot:
        return

    snapshot_path = settings.snapshot_path
    if not os.path.exists(snapshot_path):
        logger.info("snapshot autoload skipped: file does not exist (%s)", snapshot_path)
        return

    try:
        store.load(snapshot_path)
        logger.info(
            "snapshot autoloaded from %s with %s docs",
            snapshot_path,
            store.stats()["documents"],
        )
    except Exception:
        logger.exception("snapshot autoload failed for %s", snapshot_path)
        if settings.fail_on_snapshot_error:
            raise


def _model_dump(model_obj: BaseModel) -> Dict[str, Any]:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump(exclude_none=True)
    return model_obj.dict(exclude_none=True)


def _sanitize_k(k: int) -> int:
    return max(1, min(k, settings.max_k))


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    k: int = Field(default=settings.default_k, ge=1)
    model: Literal["echo", "openai", "hf"] = "echo"


class DocumentInput(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: List[DocumentInput] = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=settings.default_k, ge=1)


class SnapshotRequest(BaseModel):
    path: Optional[str] = None


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _normalize_documents(items: List[DocumentInput]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in items:
        raw = _model_dump(item)
        text = raw.get("text") or raw.get("content")
        if not text:
            continue
        docs.append(
            {
                "id": raw.get("id"),
                "text": text,
                "metadata": raw.get("metadata") or {},
            }
        )
    if not docs:
        raise HTTPException(status_code=400, detail="No valid documents were provided")
    return docs


def _resolve_adapter(model_name: str):
    if model_name == "echo":
        return model
    if model_name == "openai":
        if not openai_adapter_available:
            raise HTTPException(status_code=400, detail="OpenAI adapter not available")
        try:
            return OpenAIAdapter(api_key=os.environ.get("OPENAI_API_KEY"))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"OpenAI adapter initialization failed: {exc}",
            )
    if model_name == "hf":
        if not hf_adapter_available:
            raise HTTPException(status_code=400, detail="HuggingFace adapter not available")
        try:
            return HuggingFaceAdapter()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"HuggingFace adapter initialization failed: {exc}",
            )
    raise HTTPException(status_code=400, detail=f"Unsupported model '{model_name}'")


def _backend_status() -> Dict[str, Any]:
    return {
        "stores": {
            "tfidf": store.stats(),
            "embeddings": {
                "available": embeddings_available,
                "initialized": embedding_store is not None,
            },
            "faiss": {
                "available": faiss_available,
                "initialized": faiss_store is not None,
            },
        },
        "adapters": {
            "openai": openai_adapter_available,
            "huggingface": hf_adapter_available,
            "echo": True,
        },
        "runtime": {
            "rate_limit": {
                "enabled": app.state.rate_limiter.enabled,
                "requests_per_minute": app.state.rate_limiter.limit,
                "window_seconds": app.state.rate_limiter.window_seconds,
                "exempt_paths": sorted(app.state.rate_limit_exempt_paths),
            },
            "snapshot": {
                "path": settings.snapshot_path,
                "autoload": settings.autoload_snapshot,
            },
        },
    }


def _safe_stream_text(text: str):
    if not text:
        return
    chunk_size = max(1, settings.stream_chunk_size)
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


@app.middleware("http")
async def attach_request_id_and_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid4().hex)
    start = time.perf_counter()

    rate_decision = None
    response = None

    limiter = app.state.rate_limiter
    exempt_paths = app.state.rate_limit_exempt_paths
    should_rate_limit = (
        limiter.enabled
        and request.method != "OPTIONS"
        and request.url.path not in exempt_paths
    )

    if should_rate_limit:
        client_host = request.client.host if request.client else "unknown"
        rate_decision = limiter.check(client_host)
        if not rate_decision.allowed:
            response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
            response.headers["retry-after"] = str(rate_decision.reset_after_seconds)

    try:
        if response is None:
            response = await call_next(request)
    except Exception:
        logger.exception(
            "request_id=%s path=%s method=%s unhandled_error",
            request_id,
            request.url.path,
            request.method,
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000

    if should_rate_limit and rate_decision is not None:
        response.headers["x-ratelimit-limit"] = str(limiter.limit)
        response.headers["x-ratelimit-remaining"] = str(rate_decision.remaining)
        response.headers["x-ratelimit-reset"] = str(rate_decision.reset_after_seconds)

    response.headers["x-request-id"] = request_id
    if settings.request_logging:
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return {"status": "ok", **_backend_status()}


@app.post("/ingest")
async def ingest(payload: IngestRequest, _: None = Depends(require_api_key)):
    documents = _normalize_documents(payload.documents)
    res = store.ingest(documents)
    return res


@app.post("/ingest_embeddings")
async def ingest_embeddings(payload: IngestRequest, _: None = Depends(require_api_key)):
    if not embeddings_available or embedding_store is None:
        raise HTTPException(
            status_code=400,
            detail="Embeddings not available. Install sentence-transformers.",
        )
    documents = _normalize_documents(payload.documents)
    res = embedding_store.ingest(documents)
    return res


@app.post("/search")
async def search(payload: SearchRequest):
    k = _sanitize_k(payload.k)
    query = payload.query
    return {"results": store.search(query, k=k)}


@app.post("/search_embeddings")
async def search_embeddings(payload: SearchRequest):
    if not embeddings_available or embedding_store is None:
        raise HTTPException(
            status_code=400,
            detail="Embeddings not available. Install sentence-transformers.",
        )
    query = payload.query
    k = _sanitize_k(payload.k)
    return {"results": embedding_store.search(query, k=k)}


@app.post("/search_auto")
async def search_auto(payload: SearchRequest):
    query = payload.query
    k = _sanitize_k(payload.k)

    if faiss_available and faiss_store is not None:
        return {"backend": "faiss", "results": faiss_store.search(query, k=k)}
    if embeddings_available and embedding_store is not None:
        return {"backend": "embeddings", "results": embedding_store.search(query, k=k)}
    return {"backend": "tfidf", "results": store.search(query, k=k)}


@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.message
    k = _sanitize_k(req.k)
    contexts = [r["text"] for r in store.search(query, k=k)]
    adapter = _resolve_adapter(req.model)
    resp = adapter.generate(query, context=contexts)
    return {"model": req.model, "response": resp, "context_count": len(contexts)}


def _sse_encode(obj: dict) -> str:
    # Minimal SSE event encoding with JSON payload under `data:`
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/chat_stream")
async def chat_stream(payload: ChatRequest):
    message = payload.message
    model_name = payload.model
    k = _sanitize_k(payload.k)

    adapter = _resolve_adapter(model_name)

    # get context from store
    contexts = [r["text"] for r in store.search(message, k=k)]

    def event_stream():
        yield _sse_encode({"event": "start", "model": model_name})
        # adapter may be an EchoModel with no stream() method
        if hasattr(adapter, "stream"):
            for chunk in adapter.stream(message, context=contexts):
                if chunk:
                    yield _sse_encode({"event": "chunk", "text": chunk})
        else:
            # fallback to generate then stream in chunks
            text = adapter.generate(message, context=contexts)
            for chunk in _safe_stream_text(text):
                yield _sse_encode({"event": "chunk", "text": chunk})
        yield _sse_encode({"event": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/snapshot/save")
async def save_snapshot(payload: SnapshotRequest, _: None = Depends(require_api_key)):
    path = payload.path or settings.snapshot_path
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    store.save(path)
    return {"saved": path, "documents": store.stats()["documents"]}


@app.post("/snapshot/load")
async def load_snapshot(payload: SnapshotRequest, _: None = Depends(require_api_key)):
    path = payload.path or settings.snapshot_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Snapshot not found at '{path}'")
    store.load(path)
    return {"loaded": path, "documents": store.stats()["documents"]}


@app.post("/ingest_faiss")
async def ingest_faiss(payload: IngestRequest, _: None = Depends(require_api_key)):
    if not faiss_available or faiss_store is None:
        raise HTTPException(
            status_code=400,
            detail="FAISS store not available. Install faiss and sentence-transformers.",
        )
    documents = _normalize_documents(payload.documents)
    return faiss_store.ingest(documents)


@app.post("/search_faiss")
async def search_faiss(payload: SearchRequest):
    if not faiss_available or faiss_store is None:
        raise HTTPException(
            status_code=400,
            detail="FAISS store not available. Install faiss and sentence-transformers.",
        )
    query = payload.query
    k = _sanitize_k(payload.k)
    return {"results": faiss_store.search(query, k=k)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)

