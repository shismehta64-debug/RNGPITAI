"""End-to-end smoke tests through Flask's test client.

No API keys are configured here, so these exercise the offline paths: routing,
auth, rate limiting, validation, security headers and the local fast-path
answers. They are the tests that would have caught the missing
``/api/analytics/generate-summary`` route and the unauthenticated admin actions.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["SECRET_KEY"] = "smoke-test-secret"
os.environ["RNGAI_NO_NUMPY"] = "1"
os.environ.pop("NVIDIA_API_KEY", None)
os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)
os.environ["RATE_LIMIT_CHAT"] = "5/60"

from rngai.security import rate_limiter  # noqa: E402
from rngai.webapp import create_app, services  # noqa: E402


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        services.knowledge.build()
        cls.app = create_app()
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.client = self.app.test_client()
        rate_limiter._buckets.clear()

    # ------------------------------------------------------------- pages
    def test_home_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SINA", response.data)

    def test_security_headers_present(self):
        headers = self.client.get("/").headers
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn("Content-Security-Policy", headers)

    def test_health_reports_state(self):
        payload = self.client.get("/health").get_json()
        self.assertIn(payload["status"], ("healthy", "degraded"))
        self.assertTrue(payload["knowledge"]["chunks"] > 0)
        self.assertFalse(payload["llm_configured"], "no key is set in this test env")

    # -------------------------------------------------------------- chat
    def _stream(self, path, message):
        response = self.client.post(path, json={"message": message})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.data.decode().splitlines() if line.strip()]
        text = "".join(e.get("token", "") for e in events)
        return text, events

    def test_greeting_is_answered_without_any_api_key(self):
        text, events = self._stream("/chat", "hello")
        self.assertIn("SINA", text)
        self.assertTrue(events[-1]["done"])
        self.assertTrue(events[-1]["cached"])

    def test_identity_question_names_the_team(self):
        text, _ = self._stream("/chat", "who made you?")
        self.assertIn("InnoCrew", text)

    def test_voice_reply_has_no_markdown(self):
        text, _ = self._stream("/api/sina-chat", "who built you")
        self.assertNotIn("|", text)
        self.assertNotIn("**", text)

    def test_missing_llm_is_reported_gracefully(self):
        text, events = self._stream("/chat", "what is the fee for computer science")
        self.assertTrue(text.strip(), "the user must always get something back")
        self.assertTrue(events[-1]["done"])

    def test_empty_message_rejected(self):
        self.assertEqual(self.client.post("/chat", json={"message": "   "}).status_code, 400)
        self.assertEqual(self.client.post("/chat", json={}).status_code, 400)

    def test_oversized_message_rejected(self):
        response = self.client.post("/chat", json={"message": "x" * 5000})
        self.assertEqual(response.status_code, 413)

    def test_non_json_body_rejected_cleanly(self):
        response = self.client.post("/chat", data="not json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)

    def test_rate_limit_kicks_in(self):
        statuses = [
            self.client.post("/chat", json={"message": "hi"}).status_code for _ in range(7)
        ]
        self.assertIn(429, statuses, "the chat endpoint must be rate limited")

    # -------------------------------------------------------------- auth
    def test_admin_dashboard_requires_login(self):
        response = self.client.get("/admin/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login", response.headers["Location"])

    def test_admin_apis_return_401_not_a_redirect(self):
        for path in (
            "/api/analytics/stats",
            "/api/analytics/top-questions",
            "/api/analytics/all-questions",
            "/api/analytics/token-usage",
            "/api/admin/reports",
            "/api/nvidia-key/status",
            "/api/debug/status",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 401)

    def test_expensive_mutations_require_auth(self):
        """These were completely unauthenticated in the original app."""
        for path in (
            "/api/embeddings/regenerate",
            "/api/debug/toggle",
            "/api/nvidia-key",
            "/api/cache/clear",
            "/api/analytics/generate-summary",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.post(path, json={}).status_code, 401)

    def test_generate_summary_route_exists(self):
        """The admin dashboard has always called this; it used to 404."""
        self.assertIn(
            "/api/analytics/generate-summary",
            {rule.rule for rule in self.app.url_map.iter_rules()},
        )

    def test_admin_check_is_public_and_says_no(self):
        payload = self.client.get("/api/admin/check").get_json()
        self.assertFalse(payload["is_admin"])

    def test_login_without_database_is_503_not_500(self):
        response = self.client.post(
            "/admin/login", json={"username": "admin", "password": "admin"}
        )
        self.assertEqual(response.status_code, 503)

    def test_login_rejects_blank_credentials(self):
        response = self.client.post("/admin/login", json={"username": "", "password": ""})
        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------------ inputs
    def test_report_validates_rating(self):
        self.assertEqual(self.client.post("/api/report", json={"rating": 9}).status_code, 400)
        self.assertEqual(self.client.post("/api/report", json={}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/report", json={"rating": 4, "query": "q"}).status_code, 200
        )

    def test_tts_validates_input(self):
        self.assertEqual(self.client.post("/api/tts", json={"text": ""}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/tts", json={"text": "x" * 100000}).status_code, 413
        )

    def test_transcribe_without_key_is_503(self):
        self.assertEqual(self.client.post("/api/transcribe").status_code, 503)

    def test_unknown_api_path_returns_json_404(self):
        response = self.client.get("/api/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"], "Not found")

    def test_vrm_model_is_cacheable(self):
        response = self.client.get("/vrm-model")
        if response.status_code == 200:
            self.assertIn("max-age", response.headers.get("Cache-Control", ""))
            self.assertTrue(response.headers.get("ETag"))
        else:
            self.assertEqual(response.status_code, 404)

    def test_session_reset(self):
        self.assertTrue(self.client.post("/api/session/reset").get_json()["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
