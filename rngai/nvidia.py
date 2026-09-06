"""Thin HTTP client for the NVIDIA (OpenAI-compatible) API.

Replaces the previous mix of the ``openai`` SDK for embeddings and hand-rolled
``requests`` calls for chat. One client means one place for connection pooling,
timeouts, retries and API-key rotation - the original retried on a *substring
match against the exception text*, which silently never fired for most 429s
because ``raise_for_status`` messages do not always contain "429".
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

import requests
from requests.adapters import HTTPAdapter

from .logging_utils import get_logger

log = get_logger("rngai.nvidia")

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class NvidiaError(RuntimeError):
    """Raised when every key/attempt has failed."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status

    @property
    def is_rate_limit(self) -> bool:
        return self.status == 429

    @property
    def is_auth(self) -> bool:
        return self.status in (401, 403)


class NvidiaClient:
    """Pooled, retrying client with round-robin failover across API keys."""

    def __init__(
        self,
        api_keys: Sequence[str],
        base_url: str,
        connect_timeout: float = 8.0,
        read_timeout: float = 45.0,
        max_attempts: int = 3,
    ):
        self._keys: List[str] = [k for k in api_keys if k]
        self._lock = threading.Lock()
        self._cursor = itertools.cycle(range(len(self._keys))) if self._keys else None
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_attempts = max_attempts

        self.session = requests.Session()
        # Keep-alive across requests removes a TLS handshake (~100-200ms) from
        # every single chat and embedding call.
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=32, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # -- key management ----------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self._keys)

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def set_keys(self, api_keys: Sequence[str]) -> None:
        with self._lock:
            self._keys = [k for k in api_keys if k]
            self._cursor = itertools.cycle(range(len(self._keys))) if self._keys else None

    def _keys_in_order(self) -> List[str]:
        """Start from the next key in the rotation, then fall through the rest."""
        with self._lock:
            if not self._keys:
                return []
            start = next(self._cursor)
            return self._keys[start:] + self._keys[:start]

    # -- requests ----------------------------------------------------
    def _headers(self, key: str, stream: bool) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

    def _request(
        self,
        path: str,
        payload: Dict[str, Any],
        stream: bool,
        read_timeout: Optional[float] = None,
        optional_fields: Sequence[str] = (),
    ) -> requests.Response:
        """POST with retries and key rotation.

        ``optional_fields`` names payload keys that are nice-to-have but not
        universally supported (model-specific switches). If the API rejects the
        request with a 400, they are stripped and the call is retried once, so
        swapping to a model that does not understand them still works.
        """
        if not self._keys:
            raise NvidiaError("NVIDIA API key is not configured", status=None)

        url = f"{self.base_url}{path}"
        timeout = (self.connect_timeout, read_timeout or self.read_timeout)
        last_error: Optional[NvidiaError] = None

        for attempt in range(self.max_attempts):
            for key in self._keys_in_order():
                try:
                    response = self.session.post(
                        url, headers=self._headers(key, stream), json=payload,
                        stream=stream, timeout=timeout,
                    )
                except requests.Timeout as exc:
                    last_error = NvidiaError(f"Request timed out: {exc}", status=408)
                    continue
                except requests.RequestException as exc:
                    last_error = NvidiaError(f"Network error: {exc}", status=None)
                    continue

                if response.status_code < 400:
                    return response

                body = ""
                try:
                    body = response.text[:400]
                except Exception:  # pragma: no cover - defensive
                    pass
                response.close()
                last_error = NvidiaError(
                    f"NVIDIA API returned {response.status_code}: {body}",
                    status=response.status_code,
                )
                log.warning(
                    "NVIDIA %s -> %s (attempt %d/%d)",
                    path, response.status_code, attempt + 1, self.max_attempts,
                )
                if response.status_code == 400 and optional_fields:
                    present = [f for f in optional_fields if f in payload]
                    if present:
                        log.info(
                            "Model rejected optional field(s) %s - retrying without them",
                            ", ".join(present),
                        )
                        payload = {k: v for k, v in payload.items() if k not in present}
                        optional_fields = ()
                        continue

                if response.status_code not in RETRYABLE_STATUS:
                    # A 400 is our bug, not a transient one - another key will
                    # not fix it, so fail fast instead of burning the quota.
                    raise last_error

            if attempt + 1 < self.max_attempts:
                # Exponential backoff with jitter, capped so a user never waits
                # more than a couple of seconds for retries.
                delay = min(1.5, (2 ** attempt) * 0.25) + random.uniform(0, 0.2)
                time.sleep(delay)

        raise last_error or NvidiaError("NVIDIA API request failed")

    def post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        read_timeout: Optional[float] = None,
        optional_fields: Sequence[str] = (),
    ) -> Dict[str, Any]:
        response = self._request(
            path, payload, stream=False, read_timeout=read_timeout,
            optional_fields=optional_fields,
        )
        try:
            return response.json()
        finally:
            response.close()

    def post_sse(
        self,
        path: str,
        payload: Dict[str, Any],
        read_timeout: Optional[float] = None,
        optional_fields: Sequence[str] = (),
    ) -> Iterator[Dict[str, Any]]:
        """Yield decoded ``data:`` frames from a server-sent-event stream."""
        import json

        response = self._request(
            path, payload, stream=True, read_timeout=read_timeout,
            optional_fields=optional_fields,
        )
        try:
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    log.debug("Skipping malformed SSE frame: %s", data[:120])
        finally:
            response.close()
