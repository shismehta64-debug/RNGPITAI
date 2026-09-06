"""Text-to-speech via Edge TTS.

The original spawned ``python -m edge_tts`` as a *subprocess per request*, wrote
an MP3 to a temp file, read it back and deleted it. Interpreter startup alone
cost roughly a second before a single byte of audio was synthesised, and a burst
of requests forked a burst of Python processes.

This runs edge-tts in-process on a dedicated event loop thread and keeps a small
LRU of recent clips, so repeated phrases (greetings, fallbacks) are instant.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
from collections import OrderedDict
from typing import Optional

from .logging_utils import get_logger

log = get_logger("rngai.tts")

# Strip Markdown so the voice does not read asterisks and pipes aloud.
_MD_PATTERNS = [
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"^\s*#{1,6}\s*", re.M), ""),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"\*([^*]+)\*"), r"\1"),
    (re.compile(r"^\s*[-*•]\s+", re.M), ""),
    (re.compile(r"^\s*\|.*\|\s*$", re.M), " "),
    (re.compile(r"^\s*-{3,}\s*$", re.M), " "),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{2,}"), "\n"),
]


def strip_markdown(text: str) -> str:
    for pattern, replacement in _MD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


class TTSService:
    def __init__(self, voice: str, max_chars: int = 1200, cache_size: int = 64):
        self.voice = voice
        self.max_chars = max_chars
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import edge_tts  # noqa: F401

                self._available = True
            except ImportError:
                log.warning("edge-tts is not installed - voice output disabled")
                self._available = False
        return self._available

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop and self._loop.is_running():
                return self._loop
            loop = asyncio.new_event_loop()

            def run() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(target=run, name="tts-loop", daemon=True)
            thread.start()
            self._loop = loop
            self._thread = thread
            return loop

    async def _synthesize(self, text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                chunks.extend(chunk["data"])
        return bytes(chunks)

    def synthesize(self, text: str, voice: Optional[str] = None, timeout: float = 25.0) -> bytes:
        """Return MP3 bytes for ``text``. Raises ``RuntimeError`` on failure."""
        if not self.available:
            raise RuntimeError("Text-to-speech is not available on this server")

        clean = strip_markdown(text or "")[: self.max_chars].strip()
        if not clean:
            raise ValueError("No speakable text provided")

        voice = voice or self.voice
        key = hashlib.sha256(f"{voice}\x00{clean}".encode("utf-8")).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._synthesize(clean, voice), loop)
        try:
            audio = future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise RuntimeError("Speech synthesis timed out")
        except Exception as exc:
            raise RuntimeError(f"Speech synthesis failed: {exc}") from exc

        if not audio:
            raise RuntimeError("Speech synthesis produced no audio")

        with self._lock:
            self._cache[key] = audio
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return audio
