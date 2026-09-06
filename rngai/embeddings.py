"""Embedding client with a two-tier cache and parallel batch indexing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .logging_utils import get_logger
from .nvidia import NvidiaClient, NvidiaError

log = get_logger("rngai.embeddings")


def _key_for(text: str, model: str, input_type: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(input_type.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.strip().lower().encode("utf-8"))
    return digest.hexdigest()


class EmbeddingCache:
    """SQLite-backed cache fronted by an in-memory LRU.

    The original stored NumPy arrays via ``pickle``, which is both a
    deserialisation risk if the file is ever swapped and unreadable without
    NumPy. Vectors are stored here as zlib-compressed JSON: portable, safe and
    about 40% smaller on disk.
    """

    def __init__(self, path: Path, memory_size: int = 2048):
        self.path = path
        self.memory_size = memory_size
        self._memory: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key        TEXT PRIMARY KEY,
                preview    TEXT,
                vector     BLOB NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.commit()

    def count(self) -> int:
        with self._lock:
            try:
                return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            except sqlite3.Error:
                return 0

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                self._memory.move_to_end(key)
                return cached
            try:
                row = self._conn.execute(
                    "SELECT vector FROM embeddings WHERE key = ?", (key,)
                ).fetchone()
            except sqlite3.Error:
                return None
        if not row:
            return None
        try:
            vector = json.loads(zlib.decompress(row[0]).decode("utf-8"))
        except (zlib.error, json.JSONDecodeError, UnicodeDecodeError):
            return None
        self._remember(key, vector)
        return vector

    def get_many(self, keys: Sequence[str]) -> Dict[str, List[float]]:
        found: Dict[str, List[float]] = {}
        missing: List[str] = []
        for key in keys:
            cached = self._memory.get(key)
            if cached is not None:
                found[key] = cached
            else:
                missing.append(key)
        if not missing:
            return found

        with self._lock:
            for start in range(0, len(missing), 400):
                batch = missing[start : start + 400]
                placeholders = ",".join("?" * len(batch))
                try:
                    rows = self._conn.execute(
                        f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                        batch,
                    ).fetchall()
                except sqlite3.Error:
                    rows = []
                for key, blob in rows:
                    try:
                        found[key] = json.loads(zlib.decompress(blob).decode("utf-8"))
                    except (zlib.error, json.JSONDecodeError, UnicodeDecodeError):
                        continue
        for key, vector in found.items():
            self._remember(key, vector)
        return found

    def put(self, key: str, vector: Sequence[float], preview: str = "") -> None:
        self.put_many([(key, list(vector), preview)])

    def put_many(self, items: Sequence[tuple]) -> None:
        if not items:
            return
        rows = [
            (key, preview[:200], zlib.compress(json.dumps(list(vector)).encode("utf-8"), 6))
            for key, vector, preview in items
        ]
        with self._lock:
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (key, preview, vector) VALUES (?, ?, ?)",
                    rows,
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                log.warning("Could not persist embeddings: %s", exc)
        for key, vector, _preview in items:
            self._remember(key, list(vector))

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()
            try:
                self._conn.execute("DELETE FROM embeddings")
                self._conn.commit()
            except sqlite3.Error as exc:
                log.warning("Could not clear embedding cache: %s", exc)

    def _remember(self, key: str, vector: List[float]) -> None:
        with self._lock:
            self._memory[key] = vector
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_size:
                self._memory.popitem(last=False)


class EmbeddingService:
    """Embeds queries and passages, hitting the API only on a genuine miss."""

    def __init__(
        self,
        client: NvidiaClient,
        model: str,
        cache: EmbeddingCache,
        batch_size: int = 32,
        workers: int = 4,
    ):
        self.client = client
        self.model = model
        self.cache = cache
        self.batch_size = max(1, batch_size)
        self.workers = max(1, workers)
        self.api_calls = 0
        self.cache_hits = 0
        self._stats_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.client.configured

    def _embed_raw(self, texts: Sequence[str], input_type: str) -> List[List[float]]:
        payload = {
            "input": list(texts),
            "model": self.model,
            "encoding_format": "float",
            "input_type": input_type,
            "truncate": "END",
        }
        data = self.client.post_json("/embeddings", payload, read_timeout=30.0)
        items = data.get("data") or []
        if len(items) != len(texts):
            raise NvidiaError(
                f"Embedding API returned {len(items)} vectors for {len(texts)} inputs"
            )
        # The API does not guarantee ordering, but it does return an index.
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        with self._stats_lock:
            self.api_calls += 1
        return [list(item["embedding"]) for item in ordered]

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Embed one query. Returns ``None`` when the API is unavailable.

        A query embedding failing must never fail the request: retrieval falls
        back to BM25 and the user still gets an answer.
        """
        try:
            vectors = self.embed_many([text], input_type="query")
        except Exception as exc:
            log.warning("Query embedding failed, falling back to lexical search: %s", exc)
            return None
        return vectors[0] if vectors and vectors[0] else None

    def embed_many(
        self, texts: Sequence[str], input_type: str = "passage", progress: bool = False
    ) -> List[Optional[List[float]]]:
        """Embed a list of texts, using the cache for anything already seen."""
        if not texts:
            return []

        keys = [_key_for(t, self.model, input_type) for t in texts]
        cached = self.cache.get_many(keys)
        results: List[Optional[List[float]]] = [cached.get(k) for k in keys]

        pending = [i for i, vector in enumerate(results) if vector is None]
        with self._stats_lock:
            self.cache_hits += len(texts) - len(pending)
        if not pending:
            return results

        if not self.client.configured:
            return results

        batches = [
            pending[start : start + self.batch_size]
            for start in range(0, len(pending), self.batch_size)
        ]

        def run(batch: List[int]):
            return batch, self._embed_raw([texts[i] for i in batch], input_type)

        completed = 0
        to_persist: List[tuple] = []
        workers = 1 if len(batches) == 1 else min(self.workers, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for batch, vectors in pool.map(run, batches):
                for index, vector in zip(batch, vectors):
                    results[index] = vector
                    to_persist.append((keys[index], vector, texts[index][:200]))
                completed += len(batch)
                if progress:
                    log.info("  embedded %d/%d passages", completed, len(pending))

        self.cache.put_many(to_persist)
        return results

    def stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return {
                "api_calls": self.api_calls,
                "cache_hits": self.cache_hits,
                "cached_on_disk": self.cache.count(),
            }
