"""Text normalisation, tokenisation and domain-aware query expansion."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Set

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9._+-]*", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t ]+")
_BLANKS_RE = re.compile(r"\n{3,}")

# Words that carry no retrieval signal. Deliberately small: an over-eager stop
# list hurts short queries like "who is the HOD of IT".
STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "get", "give", "has", "have", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "tell", "that", "the", "there",
    "these", "this", "to", "want", "was", "were", "what", "when", "which",
    "will", "with", "you", "your",
}

# Domain vocabulary. Expanding the query with these before retrieval is what
# lets "fees for CS" match a chunk that only ever says
# "Computer Science & Engineering tuition".
SYNONYMS = {
    "rngpit": ["rng patel institute of technology", "rng patel", "institute"],
    "cs": ["computer science", "cse", "computer engineering"],
    "cse": ["computer science", "computer science and engineering"],
    "it": ["information technology"],
    "ec": ["electronics and communication"],
    "ee": ["electrical engineering"],
    "me": ["mechanical engineering"],
    "ce": ["civil engineering"],
    "chem": ["chemical engineering"],
    "mba": ["master of business administration", "logistics supply chain management"],
    "bvoc": ["b.voc", "bachelor of vocation"],
    "msc": ["m.sc", "master of science"],
    "hod": ["head of department", "head of the department"],
    "prof": ["professor", "faculty"],
    "fees": ["fee structure", "tuition", "cost"],
    "fee": ["fee structure", "tuition", "cost"],
    "placement": ["placements", "recruiters", "package", "salary", "companies"],
    "placements": ["placement", "recruiters", "package", "lpa"],
    "package": ["salary", "lpa", "ctc", "placement"],
    "admission": ["admissions", "eligibility", "acpc", "entrance", "apply"],
    "hostel": ["accommodation", "residence", "mess"],
    "faculty": ["professor", "teaching staff", "teachers"],
    "staff": ["faculty", "professor"],
    "contact": ["phone", "email", "address"],
    "syllabus": ["curriculum", "subjects", "course structure"],
    "lab": ["laboratory", "labs"],
    "labs": ["laboratory"],
    "club": ["clubs", "student chapter", "committee"],
    "event": ["events", "fest", "hackathon"],
    "library": ["books", "e-resources"],
    "principal": ["director"],
    "seats": ["intake", "capacity"],
    "intake": ["seats", "capacity"],
}

GREETING_RE = re.compile(
    r"^(hi|hey|hello|yo|hola|namaste|good\s+(morning|afternoon|evening|night)|"
    r"how\s+are\s+you|what'?s\s+up|sup|thanks?|thank\s+you|ok(ay)?|cool|nice|"
    r"bye|goodbye|see\s+you)[\s!.,?]*$",
    re.IGNORECASE,
)

IDENTITY_RE = re.compile(
    r"\b(who\s+(made|built|created|developed|designed)\s+(you|this|u)|"
    r"who\s+(are|r)\s+(you|u)|what\s+is\s+your\s+name|your\s+(creator|developer|maker)s?|"
    r"who\s+is\s+behind\s+(this|you))\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines without destroying structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANKS_RE.sub("\n\n", text).strip()


def normalize_query(text: str) -> str:
    """Canonical form used as a cache key: NFKC, lowercase, punctuation-trimmed."""
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n?!.,;:")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, stopwords removed, singular/plural folded."""
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]
    out: List[str] = []
    for token in tokens:
        if token in STOPWORDS or len(token) < 2:
            continue
        out.append(stem(token))
    return out


def stem(token: str) -> str:
    """A deliberately conservative suffix trim - enough to fold plurals."""
    if token.endswith("ies") and len(token) >= 5:
        return token[:-3] + "y"
    for suffix in ("ing", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def expand_query(query: str) -> str:
    """Append domain synonyms so the embedding sees the fuller intent."""
    lowered = normalize_query(query)
    words = set(re.findall(r"[a-z]+", lowered))
    additions: List[str] = []
    seen: Set[str] = set()
    for word in words:
        for synonym in SYNONYMS.get(word, ()):
            if synonym not in seen and synonym not in lowered:
                seen.add(synonym)
                additions.append(synonym)
    if not additions:
        return query.strip()
    return f"{query.strip()} ({', '.join(additions[:8])})"


def is_greeting(query: str) -> bool:
    return bool(GREETING_RE.match((query or "").strip()))


def is_identity_question(query: str) -> bool:
    return bool(IDENTITY_RE.search(query or ""))


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + "..."


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token), used only for budgeting."""
    return max(1, len(text or "") // 4)


def dedupe_preserving_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
