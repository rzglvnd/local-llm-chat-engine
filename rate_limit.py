import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class LimitDecision:
    allowed: bool
    remaining: int
    reset_after_seconds: int


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60, enabled: bool = False):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self.enabled = enabled
        self._lock = threading.Lock()
        self._events: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, key: str, now: Optional[float] = None) -> LimitDecision:
        if not self.enabled:
            return LimitDecision(allowed=True, remaining=self.limit, reset_after_seconds=0)

        current = now if now is not None else time.time()
        cutoff = current - self.window_seconds

        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= self.limit:
                reset_after = int(max(1, self.window_seconds - (current - events[0])))
                return LimitDecision(allowed=False, remaining=0, reset_after_seconds=reset_after)

            events.append(current)
            remaining = max(0, self.limit - len(events))
            reset_after = int(max(1, self.window_seconds - (current - events[0])))
            return LimitDecision(
                allowed=True,
                remaining=remaining,
                reset_after_seconds=reset_after,
            )
