from dataclasses import dataclass
import math
import os
from typing import List, Optional, Union


DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENAI_MAX_RETRIES = 2


def validate_openai_max_retries(
    value: Optional[Union[str, int]], *, name: str = "max_retries"
) -> Optional[int]:
    if value is None:
        return None
    message = f"{name} must be a non-negative integer"
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(message)
    try:
        retries = int(value)
    except ValueError:
        raise ValueError(message) from None
    if retries < 0:
        raise ValueError(message)
    return retries


def validate_openai_timeout_seconds(
    value: Union[str, float], *, name: str = "timeout_seconds"
) -> float:
    message = f"{name} must be a positive finite number of seconds"
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if isinstance(value, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(message)
    return seconds


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    log_level: str
    default_k: int
    max_k: int
    api_key: str
    cors_origins: List[str]
    snapshot_path: str
    autoload_snapshot: bool
    fail_on_snapshot_error: bool
    request_logging: bool
    stream_chunk_size: int
    rate_limit_enabled: bool
    rate_limit_requests_per_minute: int
    rate_limit_window_seconds: int
    rate_limit_exempt_paths: List[str]
    openai_timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS
    openai_max_retries: Optional[int] = None


def load_settings() -> Settings:
    return Settings(
        host=os.environ.get("LOCAL_LLM_HOST", "0.0.0.0"),
        port=_as_int(os.environ.get("LOCAL_LLM_PORT"), 8001),
        log_level=os.environ.get("LOCAL_LLM_LOG_LEVEL", "INFO"),
        default_k=_as_int(os.environ.get("LOCAL_LLM_DEFAULT_K"), 3),
        max_k=max(1, _as_int(os.environ.get("LOCAL_LLM_MAX_K"), 20)),
        api_key=os.environ.get("LOCAL_LLM_API_KEY", "").strip(),
        cors_origins=_as_list(os.environ.get("LOCAL_LLM_CORS_ORIGINS", "")),
        snapshot_path=os.environ.get("LOCAL_LLM_SNAPSHOT_PATH", "./data/store.pkl"),
        autoload_snapshot=_as_bool(os.environ.get("LOCAL_LLM_AUTOLOAD_SNAPSHOT"), True),
        fail_on_snapshot_error=_as_bool(
            os.environ.get("LOCAL_LLM_FAIL_ON_SNAPSHOT_ERROR"),
            False,
        ),
        request_logging=_as_bool(os.environ.get("LOCAL_LLM_REQUEST_LOGGING"), True),
        stream_chunk_size=max(1, _as_int(os.environ.get("LOCAL_LLM_STREAM_CHUNK_SIZE"), 64)),
        rate_limit_enabled=_as_bool(os.environ.get("LOCAL_LLM_RATE_LIMIT_ENABLED"), False),
        rate_limit_requests_per_minute=max(
            1,
            _as_int(os.environ.get("LOCAL_LLM_RATE_LIMIT_REQUESTS_PER_MINUTE"), 120),
        ),
        rate_limit_window_seconds=max(
            1,
            _as_int(os.environ.get("LOCAL_LLM_RATE_LIMIT_WINDOW_SECONDS"), 60),
        ),
        rate_limit_exempt_paths=_as_list(
            os.environ.get("LOCAL_LLM_RATE_LIMIT_EXEMPT_PATHS", "/health,/ready"),
        ),
        openai_timeout_seconds=validate_openai_timeout_seconds(
            os.environ.get("LOCAL_LLM_OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS),
            name="LOCAL_LLM_OPENAI_TIMEOUT_SECONDS",
        ),
        openai_max_retries=validate_openai_max_retries(
            os.environ.get("LOCAL_LLM_OPENAI_MAX_RETRIES"),
            name="LOCAL_LLM_OPENAI_MAX_RETRIES",
        ),
    )
