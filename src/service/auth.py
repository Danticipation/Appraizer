from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth:
    """Validate requests against a set of allowed API keys.

    Keys are stored as SHA-256 hashes to avoid plaintext in memory.
    Disabled when no keys are configured (development mode).
    """

    def __init__(self, allowed_keys: list[str] | None = None) -> None:
        self._enabled = bool(allowed_keys)
        self._hashes: set[str] = set()
        for key in (allowed_keys or []):
            if key.strip():
                self._hashes.add(self._hash(key.strip()))

    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def validate(self, api_key: str | None) -> bool:
        if not self._enabled:
            return True
        if api_key is None:
            return False
        return hmac.compare_digest(self._hash(api_key), self._hash(api_key)) and self._hash(api_key) in self._hashes

    async def __call__(self, api_key: str | None = Security(_api_key_header)) -> str | None:
        if not self._enabled:
            return None
        if not self.validate(api_key):
            logger.warning("Rejected request with invalid API key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        return api_key


class RateLimiter:
    """Simple in-memory sliding-window rate limiter per client IP.

    For production, replace with Redis-backed implementation (e.g., slowapi).
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rpm = requests_per_minute
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._enabled = requests_per_minute > 0

    def _cleanup(self, key: str, now: float) -> None:
        cutoff = now - 60.0
        self._windows[key] = [t for t in self._windows[key] if t > cutoff]

    def check(self, client_ip: str) -> bool:
        if not self._enabled:
            return True
        now = time.monotonic()
        self._cleanup(client_ip, now)
        if len(self._windows[client_ip]) >= self.rpm:
            return False
        self._windows[client_ip].append(now)
        return True

    async def middleware(self, request: Request, call_next: Any) -> Any:
        if not self._enabled:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        if not self.check(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        return await call_next(request)
