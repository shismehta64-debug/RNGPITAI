"""Authentication, rate limiting and request hardening.

Issues in the original that this module addresses:

* **Plaintext passwords.** ``if user['password_hash'] == password`` compared the
  submitted password to the stored column directly - the column was named
  ``password_hash`` but held the password. Also a non-constant-time compare.
* **No rate limiting anywhere.** ``/admin/login`` could be brute-forced, and
  ``/chat`` / ``/api/tts`` could be used to burn the API quota for free.
* **Unauthenticated admin actions.** ``/api/embeddings/regenerate`` (a very
  expensive rebuild) and ``/api/debug/toggle`` were open to the internet.
* **Wide-open CORS** combined with cookie sessions, so any site could drive the
  admin API in a logged-in user's browser.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Callable, Deque, Dict, Optional, Tuple

from flask import jsonify, redirect, request, session, url_for

from .logging_utils import get_logger

log = get_logger("rngai.security")

PBKDF2_ROUNDS = 240_000
_SCHEME = "pbkdf2_sha256"


# ---------------------------------------------------------------- passwords
def hash_password(password: str, *, rounds: int = PBKDF2_ROUNDS) -> str:
    """Return a ``pbkdf2_sha256$rounds$salt$hash`` string."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), rounds)
    return f"{_SCHEME}${rounds}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> Tuple[bool, bool]:
    """Check ``password`` against ``stored``.

    Returns ``(is_valid, needs_rehash)``. Legacy plaintext rows still
    authenticate - with a constant-time compare and a loud warning - so existing
    deployments keep working while ``needs_rehash`` drives a transparent upgrade
    on the next successful login.
    """
    if not stored:
        return False, False

    if stored.startswith(_SCHEME + "$"):
        try:
            _scheme, rounds_raw, salt, expected = stored.split("$", 3)
            rounds = int(rounds_raw)
        except (ValueError, TypeError):
            return False, False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), rounds
        )
        valid = hmac.compare_digest(digest.hex(), expected)
        return valid, valid and rounds < PBKDF2_ROUNDS

    # Legacy: a bare sha256 hex digest, or the password itself in plaintext.
    if len(stored) == 64 and all(c in "0123456789abcdefABCDEF" for c in stored):
        candidate = hashlib.sha256(password.encode("utf-8")).hexdigest()
        valid = hmac.compare_digest(candidate, stored.lower())
    else:
        valid = hmac.compare_digest(password, stored)
        if valid:
            log.warning(
                "Admin password is stored in PLAINTEXT. It will be re-hashed now; "
                "rotate the password as well."
            )
    return valid, valid


# ------------------------------------------------------------- rate limiting
class RateLimiter:
    """In-process sliding-window limiter.

    Good enough for a single-instance deployment, which is what this app is.
    Behind multiple workers, put a shared limiter (Redis / the reverse proxy) in
    front - noted in the README rather than pretended away here.
    """

    def __init__(self):
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = time.time()

    def check(self, key: str, limit: int, window: float) -> Tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        now = time.time()
        with self._lock:
            if now - self._last_sweep > 300:
                self._sweep(now)
            bucket = self._buckets[key]
            cutoff = now - window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return False, max(0.0, bucket[0] + window - now)
            bucket.append(now)
            return True, 0.0

    def reset(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)

    def _sweep(self, now: float) -> None:
        stale = [k for k, bucket in self._buckets.items() if not bucket or now - bucket[-1] > 3600]
        for key in stale:
            self._buckets.pop(key, None)
        self._last_sweep = now


rate_limiter = RateLimiter()


def parse_rate(spec: str, default: Tuple[int, float] = (20, 60.0)) -> Tuple[int, float]:
    """Parse ``"20/60"`` into ``(20, 60.0)``."""
    try:
        count, window = spec.split("/")
        return max(1, int(count)), max(1.0, float(window))
    except (ValueError, AttributeError):
        return default


def client_ip() -> str:
    """Best-effort client address.

    ``X-Forwarded-For`` is only trusted when ``BEHIND_PROXY`` is set, because an
    attacker can otherwise spoof the header to get a fresh rate-limit bucket per
    request.
    """
    from .config import config

    if config.behind_proxy:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
    return (request.remote_addr or "unknown")[:64]


def rate_limit(spec: str, scope: str = ""):
    """Decorator applying a per-IP sliding-window limit."""

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            limit, window = parse_rate(spec)
            key = f"{scope or fn.__name__}:{client_ip()}"
            allowed, retry_after = rate_limiter.check(key, limit, window)
            if not allowed:
                response = jsonify(
                    {
                        "error": "Too many requests. Please slow down.",
                        "retry_after": int(retry_after) + 1,
                    }
                )
                response.status_code = 429
                response.headers["Retry-After"] = str(int(retry_after) + 1)
                return response
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------- auth
SESSION_MAX_AGE = 12 * 3600


def _session_is_valid() -> bool:
    if not session.get("admin_logged_in"):
        return False
    issued = session.get("admin_login_at", 0)
    if not isinstance(issued, (int, float)) or time.time() - issued > SESSION_MAX_AGE:
        session.clear()
        return False
    return True


def login_required(fn: Callable):
    """HTML routes: redirect to the login page when unauthenticated."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _session_is_valid():
            return redirect(url_for("admin_login"))
        return fn(*args, **kwargs)

    return wrapper


def api_login_required(fn: Callable):
    """JSON routes: 401 rather than an HTML redirect a fetch() cannot use."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _session_is_valid():
            return jsonify({"error": "Authentication required"}), 401
        return fn(*args, **kwargs)

    return wrapper


def start_admin_session(username: str, admin_id: Optional[str]) -> None:
    session.clear()
    session["admin_logged_in"] = True
    session["admin_username"] = username
    session["admin_id"] = admin_id
    session["admin_login_at"] = time.time()
    session["csrf_token"] = secrets.token_urlsafe(32)
    session.permanent = True


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def require_csrf(fn: Callable):
    """Double-submit CSRF check for authenticated state-changing endpoints."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        submitted = request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token", "")
        if not expected or not submitted or not hmac.compare_digest(submitted, expected):
            return jsonify({"error": "Invalid or missing CSRF token"}), 403
        return fn(*args, **kwargs)

    return wrapper


# ----------------------------------------------------------------- responses
CSP = (
    "default-src 'self'; "
    # blob: is required by es-module-shims, which rewrites modules into blob
    # URLs on browsers without native import-map support.
    "script-src 'self' 'unsafe-inline' blob: https://cdn.jsdelivr.net "
    "https://cdnjs.cloudflare.com https://unpkg.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data: blob:; "
    "media-src 'self' blob: data:; "
    # blob:/data: are required here, not just in img-src: GLTFLoader reads the
    # VRM's embedded textures with ImageBitmapLoader, which fetch()es the blob
    # URLs it creates. Without them the avatar loads with no textures and
    # three-vrm then throws.
    "connect-src 'self' blob: data: https://cdn.jsdelivr.net "
    "https://cdnjs.cloudflare.com https://unpkg.com; "
    "worker-src 'self' blob:; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


def apply_security_headers(response):
    """Baseline hardening headers on every response."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy", "geolocation=(), camera=(), microphone=(self), payment=()"
    )
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    if "Content-Security-Policy" not in response.headers:
        response.headers["Content-Security-Policy"] = CSP
    return response


def safe_error(exc: Exception, fallback: str = "Something went wrong. Please try again.") -> str:
    """Never leak internal exception text to a client unless explicitly enabled."""
    from .config import config

    if config.expose_errors:
        return f"{type(exc).__name__}: {exc}"
    return fallback
