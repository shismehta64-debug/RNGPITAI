"""A small BM25 index - the lexical half of the hybrid retriever.

Dense embeddings are great at paraphrase but weak on rare literal tokens: an
email address, a phone number, "INDUX 5.0", a surname. BM25 nails exactly those,
so we run both and fuse the rankings.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .text import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, documents: Sequence[str]):
        self.doc_count = len(documents)
        self.doc_tokens: List[Counter] = []
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = {}

        for doc_id, doc in enumerate(documents):
            counts = Counter(tokenize(doc))
            self.doc_tokens.append(counts)
            self.doc_len.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings.setdefault(term, []).append((doc_id, freq))

        self.avg_len = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0
        self.idf: Dict[str, float] = {}
        for term, posting in self.postings.items():
            df = len(posting)
            # BM25+ style idf: always positive, so common terms never subtract.
            self.idf[term] = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def __len__(self) -> int:
        return self.doc_count

    def search(self, query: str, top_k: int = 30) -> List[Tuple[int, float]]:
        """Return ``(doc_index, score)`` for the best matches, highest first."""
        terms = tokenize(query)
        if not terms or not self.doc_count:
            return []

        scores: Dict[int, float] = {}
        query_counts = Counter(terms)
        for term, query_freq in query_counts.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf.get(term, 0.0)
            # Repeating a term in the query should count, but with damping.
            query_weight = idf * (1.0 + math.log(query_freq))
            for doc_id, freq in posting:
                length = self.doc_len[doc_id] or 1
                denom = freq + K1 * (1 - B + B * length / (self.avg_len or 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + query_weight * (freq * (K1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[Tuple[int, float]]],
    weights: Iterable[float] = (),
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Fuse several ranked lists into one.

    RRF scores by *rank*, not by raw score, which is what makes it safe to
    combine a cosine similarity (0..1) with a BM25 score (unbounded) without any
    fragile normalisation step.
    """
    rankings = list(rankings)
    weight_list = list(weights) or [1.0] * len(rankings)
    if len(weight_list) < len(rankings):
        weight_list += [1.0] * (len(rankings) - len(weight_list))

    fused: Dict[int, float] = {}
    for ranking, weight in zip(rankings, weight_list):
        for rank, (doc_id, _score) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
