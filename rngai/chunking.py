"""Structure-aware chunking for the knowledge base.

The original chunker split on blank lines and counted words, which shredded
Markdown tables across chunk boundaries and threw away every heading. A chunk
that reads ``| Prof. Vivek C. Joshi | I/C HOD |`` with no surrounding context is
almost impossible to retrieve *and* almost impossible for the model to use.

This chunker instead:

* parses the document into blocks (heading / table / list / paragraph),
* keeps every Markdown table intact (splitting only huge tables, and repeating
  the header row when it must),
* prefixes each chunk with its heading breadcrumb, so a table of IT faculty
  carries "Information Technology > Faculty" into both the embedding and the
  prompt,
* overlaps consecutive chunks by whole blocks rather than by arbitrary lines,
* de-duplicates identical content (``link17.txt`` is ~45% repeated lines).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .text import estimate_tokens, normalize_whitespace

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SETEXT_RE = re.compile(r"^(=+|-{4,})\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_LIST_RE = re.compile(r"^\s*([-*•]|\d+[.)])\s+")


@dataclass
class Block:
    """One logical unit of the source document."""

    kind: str  # heading | table | list | paragraph
    text: str
    level: int = 0  # heading depth, 0 for non-headings
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = estimate_tokens(self.text)


@dataclass
class Chunk:
    """A retrievable passage plus the metadata that makes it useful."""

    id: str
    text: str
    heading_path: List[str] = field(default_factory=list)
    source: str = ""
    kind: str = "text"

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def embedding_text(self) -> str:
        """What we embed: breadcrumb first so topical context is in the vector."""
        if self.heading_path:
            return f"{self.breadcrumb}\n{self.text}"
        return self.text

    @property
    def prompt_text(self) -> str:
        """What we show the model: breadcrumb as a light source header."""
        if self.heading_path:
            return f"### {self.breadcrumb}\n{self.text}"
        return self.text

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "heading_path": self.heading_path,
            "source": self.source,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, raw: Dict) -> "Chunk":
        return cls(
            id=raw["id"],
            text=raw["text"],
            heading_path=list(raw.get("heading_path") or []),
            source=raw.get("source", ""),
            kind=raw.get("kind", "text"),
        )


def _content_key(text: str) -> str:
    """Hash used for de-duplication: whitespace- and case-insensitive."""
    normalized = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def parse_blocks(text: str) -> List[Block]:
    """Split a Markdown-ish document into typed blocks."""
    lines = normalize_whitespace(text).split("\n")
    blocks: List[Block] = []
    buffer: List[str] = []
    buffer_kind = "paragraph"

    def flush() -> None:
        nonlocal buffer, buffer_kind
        if buffer:
            body = "\n".join(buffer).strip()
            if body:
                blocks.append(Block(kind=buffer_kind, text=body))
        buffer = []
        buffer_kind = "paragraph"

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush()
            i += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            flush()
            title = heading.group(2).strip().strip("*_ ").strip()
            if title:
                blocks.append(Block(kind="heading", text=title, level=len(heading.group(1))))
            i += 1
            continue

        # Setext-style heading: a short line underlined with === or ----.
        if (
            i + 1 < len(lines)
            and _SETEXT_RE.match(lines[i + 1].strip())
            and len(stripped) < 90
            and not _TABLE_ROW_RE.match(stripped)
        ):
            flush()
            level = 1 if lines[i + 1].strip().startswith("=") else 2
            blocks.append(Block(kind="heading", text=stripped.strip("*_ "), level=level))
            i += 2
            continue

        if _TABLE_ROW_RE.match(line):
            flush()
            table: List[str] = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                table.append(lines[i].strip())
                i += 1
            blocks.append(Block(kind="table", text="\n".join(table)))
            continue

        kind = "list" if _LIST_RE.match(line) else "paragraph"
        if buffer and kind != buffer_kind:
            flush()
        buffer_kind = kind
        buffer.append(stripped)
        i += 1

    flush()
    return blocks


def _split_prose(block: Block, max_tokens: int) -> List[Block]:
    """Break an oversized paragraph/list block on sentence or line boundaries.

    Some source files contain a single 10,000-character "paragraph". Left whole
    it would be one chunk that both dilutes its own embedding and eats the whole
    prompt budget in one bite.
    """
    if block.tokens <= max_tokens * 1.6:
        return [block]

    units = [line for line in block.text.split("\n") if line.strip()]
    if len(units) <= 1:
        units = re.split(r"(?<=[.!?])\s+", block.text)

    pieces: List[Block] = []
    current: List[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            pieces.append(Block(kind=block.kind, text="\n".join(current)))
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        pieces.append(Block(kind=block.kind, text="\n".join(current)))
    return pieces or [block]


def _split_table(block: Block, max_tokens: int) -> List[Block]:
    """Split an oversized table by rows, repeating the header on each piece."""
    rows = block.text.split("\n")
    if len(rows) <= 3:
        return [block]

    header = rows[0]
    separator = rows[1] if len(rows) > 1 and _TABLE_SEP_RE.match(rows[1]) else None
    body_start = 2 if separator else 1
    preamble = [header] + ([separator] if separator else [])
    preamble_tokens = estimate_tokens("\n".join(preamble))

    pieces: List[Block] = []
    current: List[str] = []
    current_tokens = preamble_tokens
    for row in rows[body_start:]:
        row_tokens = estimate_tokens(row)
        if current and current_tokens + row_tokens > max_tokens:
            pieces.append(Block(kind="table", text="\n".join(preamble + current)))
            current = []
            current_tokens = preamble_tokens
        current.append(row)
        current_tokens += row_tokens
    if current:
        pieces.append(Block(kind="table", text="\n".join(preamble + current)))
    return pieces or [block]


MIN_CHUNK_CHARS = 24
# How much content a heading may "own" before we stop trusting it as an
# ancestor. The corpus is scraped page-by-page and concatenated, so a new page
# often starts at `###` without a fresh `#`; without this, placement tables from
# one department inherited "Science and Humanities > Laboratory Facilities" from
# a page that ended thousands of tokens earlier. A wrong breadcrumb is worse
# than none: it is embedded into the vector and shown to the model.
STALE_HEADING_TOKENS = 3000


def chunk_document(
    text: str,
    source: str,
    target_tokens: int = 320,
    overlap_tokens: int = 60,
    doc_title: Optional[str] = None,
) -> List[Chunk]:
    """Turn one document into heading-aware chunks."""
    blocks = parse_blocks(text)
    heading_stack: List[str] = [doc_title] if doc_title else []
    heading_levels: List[int] = [0] if doc_title else []
    # Tokens of content seen since each stack entry was pushed.
    heading_age: List[int] = [0] * len(heading_stack)

    chunks: List[Chunk] = []
    pending: List[Block] = []
    pending_tokens = 0
    pending_headings: List[str] = list(heading_stack)
    counter = 0

    def current_headings() -> List[str]:
        """The breadcrumb with stale ancestors trimmed off the front."""
        start = 0
        for index, age in enumerate(heading_age):
            if age > STALE_HEADING_TOKENS:
                start = index + 1
            else:
                break
        return heading_stack[start:]

    def emit() -> None:
        nonlocal pending, pending_tokens, counter
        if not pending:
            return
        body = "\n\n".join(b.text for b in pending).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            counter += 1
            kind = "table" if all(b.kind == "table" for b in pending) else "text"
            chunks.append(
                Chunk(
                    id=f"{source}#{counter}",
                    text=body,
                    heading_path=list(pending_headings),
                    source=source,
                    kind=kind,
                )
            )
        pending = []
        pending_tokens = 0

    def carry_overlap() -> None:
        """Keep trailing blocks so consecutive chunks share context."""
        nonlocal pending, pending_tokens, pending_headings
        if overlap_tokens <= 0 or not chunks:
            pending, pending_tokens = [], 0
            pending_headings = list(heading_stack)
            return
        tail: List[Block] = []
        total = 0
        for block in reversed(pending):
            if block.kind == "table" or total + block.tokens > overlap_tokens:
                break
            tail.insert(0, block)
            total += block.tokens
        pending = tail
        pending_tokens = total
        pending_headings = current_headings()

    for block in blocks:
        if block.kind == "heading":
            # A new heading closes the current chunk: chunks should not straddle
            # topics, and the breadcrumb must match the content.
            emit()
            pending, pending_tokens = [], 0
            while heading_levels and heading_levels[-1] >= block.level:
                heading_levels.pop()
                heading_stack.pop()
                heading_age.pop()
            heading_stack.append(block.text)
            heading_levels.append(block.level)
            heading_age.append(0)
            pending_headings = current_headings()
            continue

        for index in range(len(heading_age)):
            heading_age[index] += block.tokens

        if block.kind == "table":
            parts = _split_table(block, target_tokens) if block.tokens > target_tokens else [block]
        else:
            parts = _split_prose(block, target_tokens)

        for part in parts:
            if pending and pending_tokens + part.tokens > target_tokens:
                emit()
                carry_overlap()
            if not pending:
                pending_headings = current_headings()
            pending.append(part)
            pending_tokens += part.tokens
            # A single oversized block still gets its own chunk.
            if pending_tokens >= target_tokens * 1.6:
                emit()
                carry_overlap()

    emit()
    return chunks


def load_corpus(
    data_dir: Path,
    target_tokens: int = 320,
    overlap_tokens: int = 60,
    extra_documents: Optional[Sequence[tuple]] = None,
) -> List[Chunk]:
    """Load and chunk every knowledge file, de-duplicating across sources.

    ``extra_documents`` is a sequence of ``(source_name, text)`` pairs merged in
    ahead of the on-disk files, used for the curated facts that must always be
    retrievable.
    """
    documents: List[tuple] = list(extra_documents or [])

    if data_dir.is_dir():
        paths = sorted(
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".txt", ".markdown"}
        )
        for path in paths:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if raw.strip():
                documents.append((str(path.relative_to(data_dir)).replace("\\", "/"), raw))

    all_chunks: List[Chunk] = []
    seen: Dict[str, str] = {}
    for source, raw in documents:
        for chunk in chunk_document(
            raw,
            source=source,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        ):
            key = _content_key(chunk.embedding_text)
            if key in seen:
                continue
            seen[key] = chunk.id
            all_chunks.append(chunk)
    return all_chunks


def corpus_fingerprint(chunks: Iterable[Chunk]) -> str:
    """Stable hash of the corpus, used to invalidate the persisted index."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.embedding_text.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()
