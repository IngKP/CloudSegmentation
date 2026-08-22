from __future__ import annotations

from collections import deque
from threading import Lock
from time import monotonic


class PredictionRateLimiter:
    """A fixed-window request limiter shared by all clients of one process."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        if limit < 1:
            raise ValueError("prediction rate limit must be at least 1")
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> int | None:
        """Record a request, or return whole seconds until the next slot."""
        now = monotonic()
        with self._lock:
            cutoff = now - self.window_seconds
            while self._requests and self._requests[0] <= cutoff:
                self._requests.popleft()
            if len(self._requests) >= self.limit:
                retry_after = self._requests[0] + self.window_seconds - now
                return max(1, int(retry_after + 0.999999))
            self._requests.append(now)
        return None


_process_limiter: PredictionRateLimiter | None = None
_process_limit: int | None = None
_process_limiter_lock = Lock()


def process_prediction_rate_limiter(limit: int) -> PredictionRateLimiter:
    """Return the process-wide limiter for the configured limit."""
    global _process_limiter, _process_limit
    with _process_limiter_lock:
        if _process_limiter is None or _process_limit != limit:
            _process_limiter = PredictionRateLimiter(limit)
            _process_limit = limit
        return _process_limiter
