"""Centralised, environment-driven configuration.

Nothing secret is hard-coded here. Every credential comes from the environment
(or a local ``.env``); missing credentials degrade the app gracefully instead of
shipping a working key inside the source tree.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_list(name: str, default: Optional[List[str]] = None) -> List[str]:
    raw = _env(name)
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Config:
    """Runtime configuration resolved once at import time."""

    # --- paths -------------------------------------------------------
    base_dir: Path = SCRIPT_DIR
    data_dir: Path = SCRIPT_DIR / "data"
    cache_dir: Path = SCRIPT_DIR / ".rngai_cache"
    vrm_path: Path = SCRIPT_DIR / "RNGPIT_SINA.vrm"

    # --- server ------------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 7860))
    debug: bool = field(default_factory=lambda: _env_bool("FLASK_DEBUG", False))
    behind_proxy: bool = field(default_factory=lambda: _env_bool("BEHIND_PROXY", False))
    force_https_cookies: bool = field(default_factory=lambda: _env_bool("FORCE_HTTPS", False))
    cors_origins: List[str] = field(default_factory=lambda: _env_list("CORS_ORIGINS"))

    # --- secrets -----------------------------------------------------
    secret_key: str = field(default_factory=lambda: _env("SECRET_KEY"))
    nvidia_api_keys: List[str] = field(default_factory=list)
    nvidia_base_url: str = field(
        default_factory=lambda: _env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    supabase_url: str = field(default_factory=lambda: _env("SUPABASE_URL"))
    supabase_key: str = field(default_factory=lambda: _env("SUPABASE_KEY"))

    # --- models ------------------------------------------------------
    chat_model: str = field(
        default_factory=lambda: _env("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
    )
    transcribe_model: str = field(
        default_factory=lambda: _env("GROQ_TRANSCRIBE_MODEL", "whisper-large-v3-turbo")
    )
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "en-US-AvaNeural"))

    # --- generation --------------------------------------------------
    max_output_tokens: int = field(default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS", 1400))
    sina_max_output_tokens: int = field(
        default_factory=lambda: _env_int("SINA_MAX_OUTPUT_TOKENS", 220)
    )
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.35))
    # Nemotron-family models "think" before answering. For a student FAQ bot
    # that is pure latency, and the reasoning trace leaks into the voice replies
    # when the answer is short. Sent as chat_template_kwargs; automatically
    # dropped if the configured model rejects it.
    disable_thinking: bool = field(default_factory=lambda: _env_bool("DISABLE_THINKING", True))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 45.0))
    llm_connect_timeout_s: float = field(
        default_factory=lambda: _env_float("LLM_CONNECT_TIMEOUT", 8.0)
    )

    # --- retrieval ---------------------------------------------------
    chunk_target_tokens: int = field(default_factory=lambda: _env_int("CHUNK_TARGET_TOKENS", 320))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 60))
    retrieval_candidates: int = field(default_factory=lambda: _env_int("RETRIEVAL_CANDIDATES", 30))
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 8))
    sina_top_k: int = field(default_factory=lambda: _env_int("SINA_TOP_K", 4))
    context_char_budget: int = field(default_factory=lambda: _env_int("CONTEXT_CHAR_BUDGET", 9000))
    sina_context_char_budget: int = field(
        default_factory=lambda: _env_int("SINA_CONTEXT_CHAR_BUDGET", 4000)
    )
    mmr_lambda: float = field(default_factory=lambda: _env_float("MMR_LAMBDA", 0.72))
    # Below this best-chunk cosine we skip the LLM entirely and say we don't
    # know. Measured against nemotron-3-embed-1b on this corpus: on-topic
    # questions score 0.29-0.63, clearly off-topic ones 0.04-0.24. The gate sits
    # low on purpose - wrongly refusing a real question is far worse than
    # spending one LLM call to refuse an off-topic one politely.
    min_relevance: float = field(default_factory=lambda: _env_float("MIN_RELEVANCE", 0.18))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 32))
    embed_workers: int = field(default_factory=lambda: _env_int("EMBED_WORKERS", 4))

    # --- caching -----------------------------------------------------
    response_cache_size: int = field(default_factory=lambda: _env_int("RESPONSE_CACHE_SIZE", 512))
    response_cache_ttl_s: int = field(
        default_factory=lambda: _env_int("RESPONSE_CACHE_TTL", 6 * 3600)
    )
    # Cosine above which two questions are treated as the same question.
    # Measured on this corpus: a genuine paraphrase ("what's the placement
    # percentage for CS" vs "What is the placement percentage for Computer
    # Science?") scores 0.965, while the confusion that must never happen -
    # CSE placements vs Civil placements - scores 0.71. 0.94 sits in that gap
    # with room on both sides.
    semantic_cache_threshold: float = field(
        default_factory=lambda: _env_float("SEMANTIC_CACHE_THRESHOLD", 0.94)
    )
    analytics_cache_ttl_s: int = field(default_factory=lambda: _env_int("ANALYTICS_CACHE_TTL", 30))

    # --- conversation ------------------------------------------------
    history_turns: int = field(default_factory=lambda: _env_int("HISTORY_TURNS", 4))
    max_sessions_in_memory: int = field(default_factory=lambda: _env_int("MAX_SESSIONS", 2000))

    # --- limits / abuse ----------------------------------------------
    max_message_chars: int = field(default_factory=lambda: _env_int("MAX_MESSAGE_CHARS", 800))
    max_tts_chars: int = field(default_factory=lambda: _env_int("MAX_TTS_CHARS", 1200))
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024)
    )
    rate_limit_chat: str = field(default_factory=lambda: _env("RATE_LIMIT_CHAT", "20/60"))
    rate_limit_tts: str = field(default_factory=lambda: _env("RATE_LIMIT_TTS", "30/60"))
    rate_limit_transcribe: str = field(
        default_factory=lambda: _env("RATE_LIMIT_TRANSCRIBE", "20/60")
    )
    rate_limit_login: str = field(default_factory=lambda: _env("RATE_LIMIT_LOGIN", "8/300"))
    rate_limit_report: str = field(default_factory=lambda: _env("RATE_LIMIT_REPORT", "10/300"))

    # --- flags -------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())
    expose_errors: bool = field(default_factory=lambda: _env_bool("EXPOSE_ERRORS", False))

    # --- derived -----------------------------------------------------
    warnings: List[str] = field(default_factory=list)
    ephemeral_secret: bool = False

    def __post_init__(self) -> None:
        keys = [k for k in (_env("NVIDIA_API_KEY"), _env("NVIDIA_API_KEY_2")) if k]
        keys.extend(_env_list("NVIDIA_API_KEYS"))
        # Preserve order, drop duplicates.
        self.nvidia_api_keys = list(dict.fromkeys(keys))

        if not self.secret_key:
            self.secret_key = secrets.token_urlsafe(48)
            self.ephemeral_secret = True
            self.warnings.append(
                "SECRET_KEY is not set - generated an ephemeral one. Sessions are "
                "invalidated on every restart and will not work across workers. "
                "Set SECRET_KEY in production."
            )
        if not self.nvidia_api_keys:
            self.warnings.append("NVIDIA_API_KEY is not set - chat and retrieval are disabled.")
        if not (self.supabase_url and self.supabase_key):
            self.warnings.append("SUPABASE_URL / SUPABASE_KEY not set - analytics disabled.")
        if not self.groq_api_key:
            self.warnings.append("GROQ_API_KEY not set - voice transcription disabled.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------
    @property
    def has_llm(self) -> bool:
        return bool(self.nvidia_api_keys)

    @property
    def index_path(self) -> Path:
        return self.cache_dir / "knowledge_index.json.gz"

    @property
    def embedding_cache_path(self) -> Path:
        return self.cache_dir / "embedding_cache.db"


config = Config()
