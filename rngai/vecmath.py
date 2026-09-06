"""Vector maths with an optional NumPy fast path.

The original app pulled in ``torch``, ``transformers``, ``sentence-transformers``
and ``chromadb`` (multiple GB of wheels) purely to do cosine similarity over a
few thousand vectors that are computed by a *remote* API anyway. Everything we
actually need is a dot product, so it lives here: NumPy when available, a pure
Python fallback when it is not, and identical results either way.
"""

from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple

Vector = Sequence[float]

np = None
if not os.environ.get("RNGAI_NO_NUMPY"):
    try:  # pragma: no cover - depends on the host environment
        import numpy as _np

        # Some Windows/MinGW NumPy builds import fine but crash on first use, so
        # exercise it once here rather than mid-request.
        _np.dot(_np.zeros(4, dtype=_np.float32), _np.zeros(4, dtype=_np.float32))
        np = _np
    except Exception:  # pragma: no cover - defensive
        np = None

HAS_NUMPY = np is not None


def normalize(vector: Vector) -> List[float]:
    """Return ``vector`` scaled to unit length (zero vectors are returned as-is)."""
    if np is not None:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr.tolist()
        return (arr / norm).tolist()

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def dot(a: Vector, b: Vector) -> float:
    if np is not None:
        return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))
    return float(sum(x * y for x, y in zip(a, b)))


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity. Cheap when both inputs are already normalised."""
    if np is not None:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom else 0.0

    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return float(sum(x * y for x, y in zip(a, b)) / denom) if denom else 0.0


class Matrix:
    """A stack of unit-length row vectors supporting fast ``similarities``."""

    __slots__ = ("_rows", "_np_rows", "dim")

    def __init__(self, rows: Sequence[Vector]):
        self._rows = [normalize(r) for r in rows]
        self.dim = len(self._rows[0]) if self._rows else 0
        self._np_rows = (
            np.asarray(self._rows, dtype=np.float32) if np is not None and self._rows else None
        )

    def __len__(self) -> int:
        return len(self._rows)

    def row(self, index: int) -> List[float]:
        return self._rows[index]

    def similarities(self, query: Vector) -> List[float]:
        """Cosine similarity of ``query`` against every row, in row order."""
        if not self._rows:
            return []
        q = normalize(query)
        if self._np_rows is not None:
            return (self._np_rows @ np.asarray(q, dtype=np.float32)).tolist()
        return [sum(x * y for x, y in zip(row, q)) for row in self._rows]

    def top_k(self, query: Vector, k: int) -> List[Tuple[int, float]]:
        """The ``k`` highest-scoring rows as ``(index, score)``, best first."""
        scores = self.similarities(query)
        if not scores:
            return []
        k = min(k, len(scores))
        if np is not None and k < len(scores):
            arr = np.asarray(scores, dtype=np.float32)
            # argpartition is O(n) vs O(n log n) for a full sort.
            idx = np.argpartition(-arr, k - 1)[:k]
            idx = idx[np.argsort(-arr[idx])]
            return [(int(i), float(arr[i])) for i in idx]
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:k]]
