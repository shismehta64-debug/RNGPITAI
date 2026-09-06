"""Short-term conversation memory.

The original app was strictly single-turn: every request was embedded and
answered on its own, so "what about civil?" retrieved nothing and the model had
no idea what "it" referred to. This keeps a small, bounded, per-session
transcript that feeds both query rewriting and the prompt.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Dict, List, Tuple


class ConversationStore:
    """Bounded in-memory transcript store, evicting least-recently-used sessions."""

    def __init__(self, max_sessions: int = 2000, max_turns: int = 8, ttl_seconds: int = 3600):
        self.max_sessions = max(1, max_sessions)
        self.max_turns = max(2, max_turns)
        self.ttl = ttl_seconds
        self._sessions: "OrderedDict[str, Dict]" = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str, turns: int) -> List[Tuple[str, str]]:
        """The last ``turns`` user/assistant pairs, oldest first."""
        if not session_id:
            return []
        with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return []
            if time.time() - record["updated"] > self.ttl:
                self._sessions.pop(session_id, None)
                return []
            self._sessions.move_to_end(session_id)
            messages: List[Tuple[str, str]] = record["messages"]
            return list(messages[-(turns * 2) :])

    def append(self, session_id: str, role: str, content: str) -> None:
        if not session_id or not content:
            return
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                record = {"messages": [], "updated": time.time()}
                self._sessions[session_id] = record
            record["messages"].append((role, content))
            # Keep only whole recent turns.
            excess = len(record["messages"]) - self.max_turns * 2
            if excess > 0:
                del record["messages"][:excess]
            record["updated"] = time.time()
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)
