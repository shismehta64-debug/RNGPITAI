"""The retrieval half of the RAG pipeline.

What changed versus the original, and why:

* **The index is persisted.** Before, every process start re-embedded the whole
  corpus through the network API - minutes of boot time and a full quota burn on
  each restart or autoscale event. The index now lives in
  ``.rngai_cache/knowledge_index.json.gz`` and is reloaded in milliseconds; it is
  rebuilt only when the corpus fingerprint changes.
* **Retrieval is hybrid.** Dense cosine similarity is fused with BM25 via
  reciprocal-rank fusion, so both "tell me about campus life" and "vcjoshi@" work.
* **Results are diversified with MMR**, so three near-identical chunks no longer
  consume the whole context window.
* **Context is packed to a budget** by whole chunks instead of being truncated
  mid-table with a blind ``[:10000]`` slice.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .bm25 import BM25Index, reciprocal_rank_fusion
from .chunking import Chunk, corpus_fingerprint, load_corpus
from .curated import CURATED_DOCUMENTS
from .embeddings import EmbeddingService
from .logging_utils import get_logger
from .text import expand_query, normalize_query
from .vecmath import HAS_NUMPY, Matrix, dot

log = get_logger("rngai.knowledge")

INDEX_VERSION = 2


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_rank: Optional[int] = None

    def to_debug(self) -> Dict:
        return {
            "id": self.chunk.id,
            "source": self.chunk.source,
            "heading": self.chunk.breadcrumb,
            "score": round(self.score, 4),
            "dense": round(self.dense_score, 4),
            "preview": self.chunk.text[:160].replace("\n", " "),
        }


class KnowledgeBase:
    """Loads, indexes and searches the RNGPIT corpus."""

    def __init__(self, config, embedder: EmbeddingService):
        self.config = config
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self.matrix: Optional[Matrix] = None
        self.bm25: Optional[BM25Index] = None
        self.fingerprint: str = ""
        self.ready = False
        self.build_seconds = 0.0
        self.last_error: Optional[str] = None
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------
    def build(self, force: bool = False) -> bool:
        """Load the corpus and make it searchable. Returns True on success."""
        with self._lock:
            started = time.time()
            self.last_error = None

            chunks = load_corpus(
                self.config.data_dir,
                target_tokens=self.config.chunk_target_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
                extra_documents=CURATED_DOCUMENTS,
            )
            if not chunks:
                self.last_error = f"No knowledge files found in {self.config.data_dir}"
                log.error(self.last_error)
                return False

            fingerprint = corpus_fingerprint(chunks)
            log.info(
                "Corpus: %d chunks from %d sources (fingerprint %s)",
                len(chunks),
                len({c.source for c in chunks}),
                fingerprint[:12],
            )

            embeddings: Optional[List[List[float]]] = None
            if not force:
                embeddings = self._load_index(fingerprint, len(chunks))
                if embeddings is not None:
                    log.info("Loaded persisted vector index (no embedding calls needed)")

            if embeddings is None:
                if not self.embedder.available:
                    # Lexical-only mode still answers a surprising amount, and
                    # it beats a hard failure at boot.
                    log.warning("No embedding API available - falling back to lexical-only search")
                    self._install(chunks, None, fingerprint)
                    self.build_seconds = time.time() - started
                    self.last_error = "Embeddings unavailable; lexical search only"
                    return True

                log.info("Embedding %d chunks via %s...", len(chunks), self.embedder.model)
                try:
                    raw = self.embedder.embed_many(
                        [c.embedding_text for c in chunks], input_type="passage", progress=True
                    )
                except Exception as exc:
                    # A dead or renamed model must not take the whole app down at
                    # boot - degrade to lexical search and say so loudly. The
                    # previous version crashed the process here.
                    self.last_error = f"Embedding failed: {exc}"
                    log.error(
                        "%s\n  -> serving lexical-only results. Check NVIDIA_EMBEDDING_MODEL.",
                        self.last_error,
                    )
                    self._install(chunks, None, fingerprint)
                    self.build_seconds = time.time() - started
                    return True

                if any(vector is None for vector in raw):
                    missing = sum(1 for v in raw if v is None)
                    self.last_error = f"Embedding failed for {missing}/{len(raw)} chunks"
                    log.error(self.last_error)
                    self._install(chunks, None, fingerprint)
                    self.build_seconds = time.time() - started
                    return True
                embeddings = [list(v) for v in raw]  # type: ignore[arg-type]
                self._save_index(chunks, embeddings, fingerprint)

            self._install(chunks, embeddings, fingerprint)
            self.build_seconds = time.time() - started
            log.info(
                "Knowledge base ready: %d chunks, dense=%s, numpy=%s, %.2fs",
                len(self.chunks),
                self.matrix is not None,
                HAS_NUMPY,
                self.build_seconds,
            )
            return True

    def _install(
        self, chunks: List[Chunk], embeddings: Optional[List[List[float]]], fingerprint: str
    ) -> None:
        self.chunks = chunks
        self.matrix = Matrix(embeddings) if embeddings else None
        self.bm25 = BM25Index([c.embedding_text for c in chunks])
        self.fingerprint = fingerprint
        self.ready = True

    # -- persistence --------------------------------------------------
    def _load_index(self, fingerprint: str, expected: int) -> Optional[List[List[float]]]:
        path: Path = self.config.index_path
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, EOFError) as exc:
            log.warning("Ignoring unreadable index at %s: %s", path, exc)
            return None

        if payload.get("version") != INDEX_VERSION:
            log.info("Index format changed - rebuilding")
            return None
        if payload.get("fingerprint") != fingerprint:
            log.info("Corpus changed since the index was written - rebuilding")
            return None
        if payload.get("model") != self.embedder.model:
            log.info("Embedding model changed - rebuilding")
            return None
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            log.warning("Index size mismatch - rebuilding")
            return None
        return embeddings

    def _save_index(
        self, chunks: List[Chunk], embeddings: List[List[float]], fingerprint: str
    ) -> None:
        path: Path = self.config.index_path
        tmp = path.with_suffix(".tmp")
        payload = {
            "version": INDEX_VERSION,
            "fingerprint": fingerprint,
            "model": self.embedder.model,
            "created_at": time.time(),
            "chunks": [c.to_dict() for c in chunks],
            "embeddings": [[round(float(v), 6) for v in vector] for vector in embeddings],
        }
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            tmp.replace(path)
            log.info("Persisted vector index to %s (%.1f MB)", path, path.stat().st_size / 1e6)
        except OSError as exc:
            log.warning("Could not persist index: %s", exc)
            tmp.unlink(missing_ok=True)

    # -- search -------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidates: Optional[int] = None,
        query_vector: Optional[Sequence[float]] = None,
    ) -> List[RetrievedChunk]:
        """Hybrid retrieval: BM25 + dense, fused by RRF, diversified by MMR."""
        if not self.ready or not self.chunks:
            return []

        query = (query or "").strip()
        if not query:
            return []

        top_k = top_k or self.config.retrieval_top_k
        candidates = candidates or max(self.config.retrieval_candidates, top_k * 3)

        expanded = expand_query(query)

        lexical = self.bm25.search(expanded, top_k=candidates) if self.bm25 else []

        dense: List[Tuple[int, float]] = []
        dense_lookup: Dict[int, float] = {}
        if self.matrix is not None:
            if query_vector is None:
                query_vector = self.embedder.embed_query(expanded)
            if query_vector is not None:
                dense = self.matrix.top_k(query_vector, candidates)
                dense_lookup = dict(dense)

        if not dense and not lexical:
            return []

        if dense and lexical:
            # Dense is weighted a little higher: it is the better generalist,
            # while BM25 is there to rescue rare literal tokens.
            fused = reciprocal_rank_fusion([dense, lexical], weights=[1.0, 0.75])
        else:
            fused = [(doc_id, 1.0 / (rank + 1)) for rank, (doc_id, _) in enumerate(dense or lexical)]

        lexical_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(lexical)}
        pool = fused[: max(candidates, top_k * 3)]

        selected = self._mmr(pool, top_k, query_vector)

        results: List[RetrievedChunk] = []
        for doc_id, score in selected:
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[doc_id],
                    score=float(score),
                    dense_score=float(dense_lookup.get(doc_id, 0.0)),
                    lexical_rank=lexical_ranks.get(doc_id),
                )
            )
        return results

    def _mmr(
        self,
        pool: Sequence[Tuple[int, float]],
        top_k: int,
        query_vector: Optional[Sequence[float]],
    ) -> List[Tuple[int, float]]:
        """Maximal Marginal Relevance: keep relevance, drop near-duplicates."""
        if not pool:
            return []
        if self.matrix is None or query_vector is None or len(pool) <= top_k:
            return list(pool[:top_k])

        lam = self.config.mmr_lambda
        remaining = list(pool)
        chosen: List[Tuple[int, float]] = []
        chosen_ids: List[int] = []

        while remaining and len(chosen) < top_k:
            best_index = 0
            best_value = float("-inf")
            for position, (doc_id, relevance) in enumerate(remaining):
                if chosen_ids:
                    vector = self.matrix.row(doc_id)
                    redundancy = max(dot(vector, self.matrix.row(other)) for other in chosen_ids)
                else:
                    redundancy = 0.0
                value = lam * relevance - (1 - lam) * redundancy * relevance_scale(pool)
                if value > best_value:
                    best_value = value
                    best_index = position
            doc_id, relevance = remaining.pop(best_index)
            chosen.append((doc_id, relevance))
            chosen_ids.append(doc_id)
        return chosen

    # -- context packing ----------------------------------------------
    def pack_context(
        self, results: Sequence[RetrievedChunk], char_budget: int
    ) -> Tuple[str, List[RetrievedChunk]]:
        """Join the best chunks into a prompt block without exceeding the budget.

        Chunks are added whole - a half-table is worse than no table.
        """
        parts: List[str] = []
        used: List[RetrievedChunk] = []
        total = 0
        for result in results:
            body = result.chunk.prompt_text.strip()
            if not body:
                continue
            cost = len(body) + 6
            if total + cost > char_budget:
                if used:
                    continue
                # Always include at least one chunk, trimmed at a line boundary.
                body = body[: max(0, char_budget - 6)].rsplit("\n", 1)[0]
                cost = len(body) + 6
            parts.append(body)
            used.append(result)
            total += cost
        return "\n\n---\n\n".join(parts), used

    def stats(self) -> Dict:
        return {
            "ready": self.ready,
            "chunks": len(self.chunks),
            "sources": sorted({c.source for c in self.chunks}),
            "dense_index": self.matrix is not None,
            "vector_dim": self.matrix.dim if self.matrix else 0,
            "numpy": HAS_NUMPY,
            "fingerprint": self.fingerprint[:12],
            "build_seconds": round(self.build_seconds, 2),
            "last_error": self.last_error,
        }


def relevance_scale(pool: Sequence[Tuple[int, float]]) -> float:
    """Scale redundancy into the same range as the fused relevance scores."""
    if not pool:
        return 1.0
    return float(pool[0][1]) or 1.0


__all__ = ["KnowledgeBase", "RetrievedChunk", "normalize_query"]
