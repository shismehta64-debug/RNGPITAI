"""Supabase persistence and analytics.

Two problems in the original are fixed here:

* **Writes blocked the response.** ``save_chat_to_supabase`` ran two round trips
  (a SELECT then an INSERT) *inside* the streaming generator, before the final
  ``done`` frame was emitted, so every user paid the database latency. Writes now
  go to a background worker.
* **Reads pulled entire tables.** ``/api/analytics/stats`` did
  ``select('id, input_tokens, ...')`` with no limit and summed in Python; with
  100k messages that is a multi-megabyte transfer per dashboard refresh. Reads
  are now bounded, and results are cached for a few seconds.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .cache import TTLCache
from .logging_utils import get_logger

log = get_logger("rngai.analytics")

MAX_ROWS = 20_000


def utcnow_iso() -> str:
    """Timezone-aware UTC timestamp (``datetime.utcnow()`` is deprecated)."""
    return datetime.now(timezone.utc).isoformat()


class Analytics:
    """Thin repository over Supabase; every method degrades to a no-op offline."""

    def __init__(self, url: str, key: str, cache_ttl: int = 30):
        self.client = None
        self._cache = TTLCache(cache_ttl)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analytics")
        self._session_uuids: Dict[str, str] = {}
        self._lock = threading.Lock()

        if not (url and key):
            log.info("Supabase not configured - analytics disabled")
            return
        try:
            from supabase import create_client

            self.client = create_client(url, key)
            log.info("Supabase connected")
        except Exception as exc:
            log.error("Could not connect to Supabase: %s", exc)
            self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------ writes
    def _submit(self, fn, *args) -> None:
        if not self.enabled:
            return
        try:
            self._pool.submit(self._guard, fn, *args)
        except RuntimeError:  # pool shutting down
            pass

    @staticmethod
    def _guard(fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:
            log.warning("Background analytics task failed: %s", exc)

    def ensure_session(self, session_id: str, ip: str, user_agent: str) -> None:
        self._submit(self._ensure_session_sync, session_id, ip, user_agent)

    def _ensure_session_sync(self, session_id: str, ip: str, user_agent: str) -> None:
        with self._lock:
            if session_id in self._session_uuids:
                return
        result = (
            self.client.table("chat_sessions")
            .insert(
                {
                    "session_id": session_id,
                    "ip_address": ip,
                    "user_agent": (user_agent or "")[:500] or None,
                    "started_at": utcnow_iso(),
                }
            )
            .execute()
        )
        if result.data:
            with self._lock:
                self._session_uuids[session_id] = result.data[0]["id"]

    def _session_uuid(self, session_id: str) -> Optional[str]:
        with self._lock:
            cached = self._session_uuids.get(session_id)
        if cached:
            return cached
        result = (
            self.client.table("chat_sessions")
            .select("id")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        uuid = result.data[0]["id"]
        with self._lock:
            # Bound the map so a long-lived process cannot grow without limit.
            if len(self._session_uuids) > 5000:
                self._session_uuids.clear()
            self._session_uuids[session_id] = uuid
        return uuid

    def record_message(
        self,
        session_id: str,
        question: str,
        answer: str,
        response_time_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        source: str = "chat",
        cached: bool = False,
    ) -> None:
        self._submit(
            self._record_message_sync,
            session_id,
            question,
            answer,
            response_time_ms,
            input_tokens,
            output_tokens,
            source,
            cached,
        )

    def _record_message_sync(
        self,
        session_id: str,
        question: str,
        answer: str,
        response_time_ms: int,
        input_tokens: int,
        output_tokens: int,
        source: str,
        cached: bool,
    ) -> None:
        uuid = self._session_uuid(session_id)
        if not uuid:
            return
        self.client.table("chat_messages").insert(
            {
                "session_id": uuid,
                "user_question": question[:2000],
                "ai_response": answer[:5000],
                "response_time_ms": response_time_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": 0.0,
                "source": source if not cached else f"{source}:cached",
                "created_at": utcnow_iso(),
            }
        ).execute()

    def record_report(
        self, query: str, response: str, rating: int, reason: str, source: str
    ) -> None:
        self._submit(self._record_report_sync, query, response, rating, reason, source)

    def _record_report_sync(
        self, query: str, response: str, rating: int, reason: str, source: str
    ) -> None:
        self.client.table("reported_issues").insert(
            {
                "query": (query or "")[:2000],
                "ai_response": (response or "")[:5000],
                "rating": rating,
                "reason": (reason or "")[:2000],
                "source": source,
                "created_at": utcnow_iso(),
            }
        ).execute()

    def touch_login(self, admin_id: str) -> None:
        self._submit(
            lambda: self.client.table("admin_users")
            .update({"last_login": utcnow_iso()})
            .eq("id", admin_id)
            .execute()
        )

    def upgrade_password(self, admin_id: str, password_hash: str) -> None:
        self._submit(
            lambda: self.client.table("admin_users")
            .update({"password_hash": password_hash})
            .eq("id", admin_id)
            .execute()
        )

    # ------------------------------------------------------------- reads
    def find_admin(self, username: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        result = (
            self.client.table("admin_users")
            .select("id, username, password_hash")
            .eq("username", username)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def _messages(self, since: Optional[str] = None) -> List[Dict]:
        """Bounded fetch of message rows for aggregation."""
        query = self.client.table("chat_messages").select(
            "created_at, input_tokens, output_tokens, total_cost"
        )
        if since:
            query = query.gte("created_at", since)
        result = query.order("created_at", desc=True).limit(MAX_ROWS).execute()
        return result.data or []

    def stats(self) -> Dict:
        if not self.enabled:
            return {"error": "Database not available"}
        cached = self._cache.get("stats")
        if cached is not None:
            return cached

        counted = (
            self.client.table("chat_messages").select("id", count="exact").limit(1).execute()
        )
        total_questions = counted.count or 0
        sessions = self.client.table("chat_sessions").select("id", count="exact").limit(1).execute()
        total_sessions = sessions.count or 0

        window_start = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
        rows = self._messages(since=window_start)

        today = datetime.now(timezone.utc).date().isoformat()
        total_in = total_out = total_cost = 0
        today_in = today_out = today_cost = 0
        today_questions = 0
        for row in rows:
            tin = row.get("input_tokens") or 0
            tout = row.get("output_tokens") or 0
            cost = row.get("total_cost") or 0
            total_in += tin
            total_out += tout
            total_cost += cost
            if (row.get("created_at") or "").startswith(today):
                today_questions += 1
                today_in += tin
                today_out += tout
                today_cost += cost

        sampled = len(rows) or 1
        payload = {
            "total_questions": total_questions,
            "total_sessions": total_sessions,
            "today_questions": today_questions,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_cost": round(total_cost, 6),
            "today_input_tokens": today_in,
            "today_output_tokens": today_out,
            "today_tokens": today_in + today_out,
            "today_cost": round(today_cost, 6),
            "avg_input_tokens": round(total_in / sampled, 1),
            "avg_output_tokens": round(total_out / sampled, 1),
            "avg_cost_per_message": round(total_cost / sampled, 6),
            "window_days": 30,
        }
        self._cache.put("stats", payload)
        return payload

    def top_questions(self, limit: int = 10) -> Dict:
        if not self.enabled:
            return {"error": "Database not available"}
        key = f"top:{limit}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = (
            self.client.table("chat_messages")
            .select("user_question, created_at")
            .order("created_at", desc=True)
            .limit(5000)
            .execute()
        )
        counts: Dict[str, Dict] = {}
        for row in result.data or []:
            question = (row.get("user_question") or "").strip()
            if not question:
                continue
            key_q = question.lower()
            entry = counts.get(key_q)
            if entry:
                entry["count"] += 1
                if row["created_at"] > entry["last_asked"]:
                    entry["last_asked"] = row["created_at"]
                    entry["question"] = question
            else:
                counts[key_q] = {
                    "question": question,
                    "count": 1,
                    "last_asked": row["created_at"],
                }
        ranked = sorted(counts.values(), key=lambda item: item["count"], reverse=True)[:limit]
        payload = {"questions": ranked}
        self._cache.put(key, payload)
        return payload

    def all_questions(self, page: int = 1, per_page: int = 20, search: str = "") -> Dict:
        if not self.enabled:
            return {"error": "Database not available"}
        page = max(1, page)
        per_page = min(max(1, per_page), 100)
        offset = (page - 1) * per_page

        query = self.client.table("chat_messages").select(
            "id, user_question, ai_response, created_at, response_time_ms", count="exact"
        )
        if search:
            # Escape PostgREST wildcards so a search for "%" is not a full scan.
            escaped = search.replace("%", r"\%").replace("_", r"\_")
            query = query.ilike("user_question", f"%{escaped}%")
        result = query.order("created_at", desc=True).range(offset, offset + per_page - 1).execute()
        total = result.count or 0
        return {
            "questions": result.data or [],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    def token_usage(self) -> Dict:
        if not self.enabled:
            return {"error": "Database not available"}
        cached = self._cache.get("usage")
        if cached is not None:
            return cached

        window_start = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
        rows = self._messages(since=window_start)

        daily: Dict[str, Dict] = {}
        hourly: Dict[str, Dict] = {}
        for row in rows:
            raw = row.get("created_at") or ""
            try:
                created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            tin = row.get("input_tokens") or 0
            tout = row.get("output_tokens") or 0
            cost = row.get("total_cost") or 0
            for bucket, key, label in (
                (daily, created.strftime("%Y-%m-%d"), "date"),
                (hourly, created.strftime("%Y-%m-%d %H:00"), "hour"),
            ):
                entry = bucket.setdefault(
                    key,
                    {
                        label: key,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0,
                        "messages": 0,
                    },
                )
                entry["input_tokens"] += tin
                entry["output_tokens"] += tout
                entry["total_tokens"] += tin + tout
                entry["cost"] += cost
                entry["messages"] += 1

        daily_usage = sorted(daily.values(), key=lambda d: d["date"])[-30:]
        hourly_usage = sorted(hourly.values(), key=lambda d: d["hour"])[-24:]
        days = len(daily_usage) or 1
        avg_tokens = sum(d["total_tokens"] for d in daily_usage) / days
        avg_cost = sum(d["cost"] for d in daily_usage) / days
        avg_messages = sum(d["messages"] for d in daily_usage) / days

        payload = {
            "daily_usage": daily_usage,
            "hourly_usage": hourly_usage,
            "projections": {
                "avg_daily_tokens": round(avg_tokens, 0),
                "avg_daily_cost": round(avg_cost, 4),
                "avg_daily_messages": round(avg_messages, 1),
                "projected_monthly_tokens": round(avg_tokens * 30, 0),
                "projected_monthly_cost": round(avg_cost * 30, 2),
                "projected_monthly_messages": round(avg_messages * 30, 0),
            },
        }
        self._cache.put("usage", payload)
        return payload

    def reports(self, limit: int = 500) -> List[Dict]:
        if not self.enabled:
            return []
        result = (
            self.client.table("reported_issues")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []

    def invalidate(self) -> None:
        self._cache.clear()
