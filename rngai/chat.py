"""The chat pipeline: retrieve, assemble, stream, cache, record.

Latency budget, worst to best case:

* cold question  -> 1 embedding call (cached forever after) + 1 streaming LLM call
* repeat question -> 0 API calls (exact response-cache hit)
* similar question -> 1 embedding call, 0 LLM calls (semantic cache hit)
* greeting / "who made you" -> 0 API calls, answered locally

The original always paid for an embedding *and* a full LLM call with a
5,000-token system prompt, even for "hi".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

from .cache import ResponseCache
from .conversations import ConversationStore
from .knowledge import KnowledgeBase, RetrievedChunk
from .logging_utils import get_logger
from .nvidia import NvidiaClient, NvidiaError
from .prompts import (
    NO_CONTEXT_REPLY,
    NO_CONTEXT_REPLY_VOICE,
    build_messages,
    build_search_query,
    instant_reply,
)
from .text import estimate_tokens, expand_query

log = get_logger("rngai.chat")


class ThinkFilter:
    """Suppresses ``<think>...</think>`` spans from a token stream.

    Reasoning models normally return their scratchpad in ``reasoning_content``,
    but some emit it inline as content instead. Reading a model's internal
    monologue aloud through TTS is the worst version of that bug, so the tokens
    are filtered as they arrive rather than patched up afterwards.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.inside = False
        self.buffer = ""

    def feed(self, token: str) -> str:
        self.buffer += token
        out: List[str] = []
        while self.buffer:
            if self.inside:
                end = self.buffer.find(self.CLOSE)
                if end == -1:
                    # Keep only enough to recognise a tag split across tokens.
                    self.buffer = self.buffer[-len(self.CLOSE):]
                    break
                self.buffer = self.buffer[end + len(self.CLOSE):]
                self.inside = False
                continue
            start = self.buffer.find(self.OPEN)
            if start == -1:
                keep = len(self.OPEN) - 1
                if len(self.buffer) > keep:
                    out.append(self.buffer[:-keep] if keep else self.buffer)
                    self.buffer = self.buffer[-keep:] if keep else ""
                break
            out.append(self.buffer[:start])
            self.buffer = self.buffer[start + len(self.OPEN):]
            self.inside = True
        return "".join(out)

    def flush(self) -> str:
        if self.inside:
            self.buffer = ""
            return ""
        tail, self.buffer = self.buffer, ""
        return tail


@dataclass
class ChatResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    sources: List[Dict] = field(default_factory=list)
    cached: bool = False
    cache_kind: str = ""
    retrieval_ms: int = 0
    total_ms: int = 0
    error: Optional[str] = None


class ChatService:
    def __init__(
        self,
        config,
        client: NvidiaClient,
        knowledge: KnowledgeBase,
        cache: ResponseCache,
        conversations: ConversationStore,
    ):
        self.config = config
        self.client = client
        self.knowledge = knowledge
        self.cache = cache
        self.conversations = conversations

    # ------------------------------------------------------------ helpers
    def _budget(self, voice: bool) -> int:
        return (
            self.config.sina_context_char_budget if voice else self.config.context_char_budget
        )

    def _top_k(self, voice: bool) -> int:
        return self.config.sina_top_k if voice else self.config.retrieval_top_k

    def _max_tokens(self, voice: bool) -> int:
        return (
            self.config.sina_max_output_tokens if voice else self.config.max_output_tokens
        )

    def retrieve(
        self, query: str, voice: bool, search_query: str
    ) -> tuple:
        """Return ``(context, used_chunks, query_vector, elapsed_ms)``."""
        started = time.time()
        query_vector = None
        if self.knowledge.matrix is not None and self.knowledge.embedder.available:
            query_vector = self.knowledge.embedder.embed_query(expand_query(search_query))

        results: List[RetrievedChunk] = self.knowledge.search(
            search_query, top_k=self._top_k(voice), query_vector=query_vector
        )
        # Drop chunks that only scraped in via fusion noise.
        if results:
            best = results[0].dense_score
            if best and best < self.config.min_relevance:
                log.info("Weak retrieval (best dense score %.3f) for %r", best, query[:60])
                results = []

        context, used = self.knowledge.pack_context(results, self._budget(voice))
        return context, used, query_vector, int((time.time() - started) * 1000)

    # -------------------------------------------------------------- stream
    def stream(
        self, query: str, session_id: str, voice: bool = False, debug: bool = False
    ) -> Iterator[Dict]:
        """Yield NDJSON-ready event dicts for one user turn."""
        started = time.time()
        query = (query or "").strip()[: self.config.max_message_chars]
        if not query:
            yield {"error": "Empty message"}
            return

        history = self.conversations.history(session_id, self.config.history_turns)

        # 1. Local fast path - greetings and identity questions.
        canned = instant_reply(query, voice=voice)
        if canned:
            for event in self._emit_text(canned):
                yield event
            self._finish(
                session_id, query, canned, started, voice, cached=True,
                cache_kind="instant", sources=[], debug=debug, input_tokens=0,
            )
            yield self._done(canned, started, cached=True, cache_kind="instant",
                             sources=[], debug=debug, retrieval_ms=0, input_tokens=0)
            return

        if not self.client.configured:
            message = (
                "The assistant is not connected to its language model right now. "
                "Please try again shortly."
            )
            for event in self._emit_text(message):
                yield event
            yield self._done(message, started, cached=False, cache_kind="", sources=[],
                             debug=debug, retrieval_ms=0, input_tokens=0, error="llm_unconfigured")
            return

        # 2. Retrieval (also gives us the query vector for the semantic cache).
        search_query = build_search_query(query, history)
        context, used, query_vector, retrieval_ms = self.retrieve(query, voice, search_query)
        sources = [chunk.to_debug() for chunk in used]

        # 3. Response cache. Safe whenever the question stands on its own. A
        #    rewritten follow-up ("what about civil?") is context-dependent, so
        #    it is neither read from nor written to the cache. Gating on "has
        #    this session any history at all" would disable the cache for every
        #    user after their first message.
        self_contained = search_query == query
        if self_contained:
            hit = self.cache.get(query, query_vector)
            if hit:
                entry, kind = hit
                log.info("Cache %s hit for %r", kind, query[:60])
                for event in self._emit_text(entry.answer):
                    yield event
                self._finish(
                    session_id, query, entry.answer, started, voice, cached=True,
                    cache_kind=kind, sources=entry.sources, debug=debug, input_tokens=0,
                )
                yield self._done(entry.answer, started, cached=True, cache_kind=kind,
                                 sources=entry.sources, debug=debug,
                                 retrieval_ms=retrieval_ms, input_tokens=0)
                return

        # 4. No usable context - answer honestly instead of hallucinating.
        if not context:
            message = NO_CONTEXT_REPLY_VOICE if voice else NO_CONTEXT_REPLY
            for event in self._emit_text(message):
                yield event
            self._finish(
                session_id, query, message, started, voice, cached=False,
                cache_kind="", sources=[], debug=debug, input_tokens=0,
            )
            yield self._done(message, started, cached=False, cache_kind="", sources=[],
                             debug=debug, retrieval_ms=retrieval_ms, input_tokens=0)
            return

        # 5. Stream the model.
        messages = build_messages(query, context, history=history, voice=voice)
        payload = {
            "model": self.config.chat_model,
            "messages": messages,
            "max_tokens": self._max_tokens(voice),
            "temperature": self.config.temperature,
            "top_p": 0.95,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.config.disable_thinking:
            payload["chat_template_kwargs"] = {"thinking": False}

        answer_parts: List[str] = []
        usage: Dict = {}
        error: Optional[str] = None
        think_filter = ThinkFilter()
        try:
            for frame in self.client.post_sse(
                "/chat/completions",
                payload,
                read_timeout=self.config.llm_timeout_s,
                optional_fields=("chat_template_kwargs",),
            ):
                if frame.get("usage"):
                    usage = frame["usage"]
                choices = frame.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content")
                if token:
                    visible = think_filter.feed(token)
                    if visible:
                        answer_parts.append(visible)
                        yield {"token": visible}
        except NvidiaError as exc:
            error = "rate_limited" if exc.is_rate_limit else "upstream_error"
            log.error("LLM stream failed: %s", exc)
            if not answer_parts:
                message = (
                    "I'm getting a lot of requests right now - please try again in a moment."
                    if exc.is_rate_limit
                    else "I hit a problem reaching my language model. Please try again."
                )
                for event in self._emit_text(message):
                    yield event
                answer_parts = [message]
        except Exception as exc:  # pragma: no cover - defensive
            error = "internal_error"
            log.exception("Unexpected error while streaming: %s", exc)

        tail = think_filter.flush()
        if tail:
            answer_parts.append(tail)
            yield {"token": tail}

        answer = "".join(answer_parts).strip()
        input_tokens = int(usage.get("prompt_tokens") or estimate_tokens(
            "".join(m["content"] for m in messages)
        ))
        output_tokens = int(usage.get("completion_tokens") or estimate_tokens(answer))

        if answer and not error and self_contained:
            self.cache.put(query, answer, query_vector, sources)

        self._finish(
            session_id, query, answer, started, voice, cached=False, cache_kind="",
            sources=sources, debug=debug, input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        yield self._done(
            answer, started, cached=False, cache_kind="", sources=sources, debug=debug,
            retrieval_ms=retrieval_ms, input_tokens=input_tokens,
            output_tokens=output_tokens, error=error,
        )

    # ------------------------------------------------------------ plumbing
    @staticmethod
    def _emit_text(text: str) -> Iterator[Dict]:
        """Emit a locally-produced answer in chunks so the UI still animates."""
        step = 24
        for start in range(0, len(text), step):
            yield {"token": text[start : start + step]}

    def _finish(
        self,
        session_id: str,
        query: str,
        answer: str,
        started: float,
        voice: bool,
        cached: bool,
        cache_kind: str,
        sources: List[Dict],
        debug: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        if answer:
            self.conversations.append(session_id, "user", query)
            self.conversations.append(session_id, "assistant", answer)
        self.on_complete(
            session_id=session_id,
            question=query,
            answer=answer,
            response_time_ms=int((time.time() - started) * 1000),
            input_tokens=input_tokens,
            output_tokens=output_tokens or estimate_tokens(answer),
            source="sina" if voice else "chat",
            cached=cached,
        )

    def on_complete(self, **kwargs) -> None:
        """Overridden by the app to persist analytics. No-op by default."""

    def _done(
        self,
        answer: str,
        started: float,
        cached: bool,
        cache_kind: str,
        sources: List[Dict],
        debug: bool,
        retrieval_ms: int,
        input_tokens: int,
        output_tokens: int = 0,
        error: Optional[str] = None,
    ) -> Dict:
        event: Dict = {
            "done": True,
            "full_text": answer,
            "cached": cached,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens or estimate_tokens(answer),
            "elapsed_ms": int((time.time() - started) * 1000),
        }
        if error:
            event["error_kind"] = error
        if debug:
            event["debug"] = {
                "cache": cache_kind or "miss",
                "retrieval_ms": retrieval_ms,
                "chunks": len(sources),
                "model": self.config.chat_model,
                "sources": sources,
            }
        return event
