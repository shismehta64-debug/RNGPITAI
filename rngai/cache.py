"""Response caching - the "CAG" layer the original advertised but never used.

The old code defined ``RESPONSE_CACHE`` and ``get_cached_response()`` and then
never called them, while ``/health`` proudly reported
``"architecture": "CAG (Cache-Augmented Generation)"``. This one is actually
wired into the chat path, and does two things the old one could not:

* entries expire (a stale answer about fees is worse than a slow one),
* lookups are *semantic* - "what are the fees" hits the entry stored for
  "how much are the fees" via cosine similarity on the query embedding.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .text import normalize_query
from .vecmath import Matrix


@dataclass
class CacheEntry:
    answer: str
    created_at: float
    query: str
    sources: List[dict] = field(default_factory=list)
    hits: int = 0


class ResponseCache:
    """LRU + TTL cache with an optional semantic index over cached queries."""

    def __init__(self, max_size: int = 512, ttl_seconds: int = 6 * 3600, threshold: float = 0.965):
        self.max_size = max(1, max_size)
        self.ttl = ttl_seconds
        self.threshold = threshold
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._vectors: Dict[str, List[float]] = {}
        self._matrix: Optional[Matrix] = None
        self._matrix_keys: List[str] = []
        self._dirty = True
        self._lock = threading.Lock()
        self.hits = 0
        self.semantic_hits = 0
        self.misses = 0

    # -- internals ----------------------------------------------------
    def _expired(self, entry: CacheEntry) -> bool:
        # ttl == 0 disables caching outright; a negative ttl means "never expire".
        if self.ttl < 0:
            return False
        return (time.time() - entry.created_at) >= self.ttl

    def _evict(self, key: str) -> None:
        self._entries.pop(key, None)
        if self._vectors.pop(key, None) is not None:
            self._dirty = True

    def _rebuild_matrix(self) -> None:
        keys = [k for k in self._entries if k in self._vectors]
        self._matrix_keys = keys
        self._matrix = Matrix([self._vectors[k] for k in keys]) if keys else None
        self._dirty = False

    # -- api ----------------------------------------------------------
    def get(
        self, query: str, query_vector: Optional[Sequence[float]] = None
    ) -> Optional[Tuple[CacheEntry, str]]:
        """Return ``(entry, "exact"|"semantic")`` on a hit, else ``None``."""
        key = normalize_query(query)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                if self._expired(entry):
                    self._evict(key)
                else:
                    self._entries.move_to_end(key)
                    entry.hits += 1
                    self.hits += 1
                    return entry, "exact"

            if query_vector is not None:
                if self._dirty:
                    self._rebuild_matrix()
                if self._matrix is not None and len(self._matrix):
                    ranked = self._matrix.top_k(query_vector, 1)
                    if ranked:
                        index, score = ranked[0]
                        if score >= self.threshold and index < len(self._matrix_keys):
                            near_key = self._matrix_keys[index]
                            near = self._entries.get(near_key)
                            if near is not None and not self._expired(near):
                                self._entries.move_to_end(near_key)
                                near.hits += 1
                                self.hits += 1
                                self.semantic_hits += 1
                                return near, "semantic"
                            if near is not None:
                                self._evict(near_key)

            self.misses += 1
            return None

    def put(
        self,
        query: str,
        answer: str,
        query_vector: Optional[Sequence[float]] = None,
        sources: Optional[List[dict]] = None,
    ) -> None:
        if not answer or not answer.strip():
            return
        key = normalize_query(query)
        if not key:
            return
        with self._lock:
            self._entries[key] = CacheEntry(
                answer=answer,
                created_at=time.time(),
                query=query,
                sources=list(sources or []),
            )
            self._entries.move_to_end(key)
            if query_vector is not None:
                self._vectors[key] = list(query_vector)
                self._dirty = True
            while len(self._entries) > self.max_size:
                oldest, _ = self._entries.popitem(last=False)
                if self._vectors.pop(oldest, None) is not None:
                    self._dirty = True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._vectors.clear()
            self._matrix = None
            self._matrix_keys = []
            self._dirty = True

    def stats(self) -> Dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "hits": self.hits,
                "semantic_hits": self.semantic_hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "ttl_seconds": self.ttl,
            }


class TTLCache:
    """A tiny TTL cache for expensive read-only work (analytics rollups)."""

    def __init__(self, ttl_seconds: int = 30):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            hit = self._store.get(key)
            if not hit:
                return None
            created, value = hit
            if time.time() - created > self.ttl:
                self._store.pop(key, None)
                return None
            return value

    def put(self, key: str, value) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
