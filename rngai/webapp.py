"""Flask application factory and HTTP routes."""

from __future__ import annotations

import json
import time
import uuid
from datetime import timedelta
from typing import Dict, Iterator

import requests
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from . import __version__
from .analytics import Analytics, utcnow_iso
from .cache import ResponseCache
from .chat import ChatService
from .config import config
from .conversations import ConversationStore
from .embeddings import EmbeddingCache, EmbeddingService
from .knowledge import KnowledgeBase
from .logging_utils import get_logger
from .nvidia import NvidiaClient
from .security import (
    api_login_required,
    apply_security_headers,
    client_ip,
    csrf_token,
    hash_password,
    login_required,
    rate_limit,
    require_csrf,
    safe_error,
    start_admin_session,
    verify_password,
)
from .tts import TTSService

log = get_logger("rngai.web")


class Services:
    """Container wiring every component together once at startup."""

    def __init__(self):
        self.client = NvidiaClient(
            config.nvidia_api_keys,
            config.nvidia_base_url,
            connect_timeout=config.llm_connect_timeout_s,
            read_timeout=config.llm_timeout_s,
        )
        self.embedding_cache = EmbeddingCache(config.embedding_cache_path)
        self.embedder = EmbeddingService(
            self.client,
            config.embedding_model,
            self.embedding_cache,
            batch_size=config.embed_batch_size,
            workers=config.embed_workers,
        )
        self.knowledge = KnowledgeBase(config, self.embedder)
        self.response_cache = ResponseCache(
            max_size=config.response_cache_size,
            ttl_seconds=config.response_cache_ttl_s,
            threshold=config.semantic_cache_threshold,
        )
        self.conversations = ConversationStore(max_sessions=config.max_sessions_in_memory)
        self.analytics = Analytics(
            config.supabase_url, config.supabase_key, cache_ttl=config.analytics_cache_ttl_s
        )
        self.tts = TTSService(config.tts_voice, max_chars=config.max_tts_chars)
        self.chat = ChatService(
            config, self.client, self.knowledge, self.response_cache, self.conversations
        )
        self.chat.on_complete = self._record
        self.debug_mode = False
        self.started_at = time.time()

        self.http = requests.Session()

    def _record(self, **kwargs) -> None:
        if kwargs.get("answer"):
            self.analytics.record_message(**kwargs)

    def warm_up(self) -> None:
        self.knowledge.build()


services = Services()


# --------------------------------------------------------------- utilities
def json_body() -> Dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def get_session_id() -> str:
    """Stable per-browser chat session id, created lazily."""
    session_id = session.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["chat_session_id"] = session_id
        services.analytics.ensure_session(
            session_id, client_ip(), request.user_agent.string or ""
        )
    return session_id


def ndjson_stream(events: Iterator[Dict]) -> Response:
    """Wrap an event iterator as a non-buffered NDJSON response."""

    def generate() -> Iterator[str]:
        try:
            for event in events:
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except GeneratorExit:  # client disconnected mid-stream
            raise
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Stream failed: %s", exc)
            yield json.dumps({"error": safe_error(exc), "done": True}) + "\n"

    response = Response(generate(), mimetype="application/x-ndjson")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Accel-Buffering"] = "no"  # do not let nginx buffer the stream
    return response


def read_message() -> tuple:
    """Validate the ``message`` field. Returns ``(message, error_response)``."""
    body = json_body()
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return "", (jsonify({"error": "No message provided"}), 400)
    message = message.strip()
    if len(message) > config.max_message_chars:
        return "", (
            jsonify({"error": f"Message too long (max {config.max_message_chars} characters)"}),
            413,
        )
    return message, None


# ------------------------------------------------------------ app factory
def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(config.base_dir / "templates"),
        static_folder=str(config.base_dir / "static"),
    )
    app.secret_key = config.secret_key
    app.config.update(
        MAX_CONTENT_LENGTH=config.max_upload_bytes,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=config.force_https_cookies,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        JSON_SORT_KEYS=False,
    )

    if config.behind_proxy:
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    # CORS is opt-in. The original enabled it for every origin while also using
    # cookie-based admin sessions, which let any site drive the admin API.
    if config.cors_origins:
        try:
            from flask_cors import CORS

            CORS(app, origins=config.cors_origins, supports_credentials=True)
            log.info("CORS enabled for: %s", ", ".join(config.cors_origins))
        except ImportError:
            log.warning("flask-cors is not installed - CORS_ORIGINS ignored")

    app.after_request(apply_security_headers)
    register_routes(app)
    return app


# ---------------------------------------------------------------- routes
def register_routes(app: Flask) -> None:
    # ---------------------------------------------------------- pages
    @app.route("/")
    def home():
        return render_template("index.html", csrf_token=csrf_token())

    @app.route("/health")
    def health():
        return jsonify(
            {
                "status": "healthy" if services.knowledge.ready else "degraded",
                "version": __version__,
                "uptime_seconds": int(time.time() - services.started_at),
                "llm_configured": services.client.configured,
                "llm_keys": services.client.key_count,
                "model": config.chat_model,
                "embedding_model": config.embedding_model,
                "knowledge": services.knowledge.stats(),
                "response_cache": services.response_cache.stats(),
                "embeddings": services.embedder.stats(),
                "conversations": services.conversations.size(),
                "analytics": services.analytics.enabled,
                "tts": services.tts.available,
                "transcription": bool(config.groq_api_key),
                "debug_mode": services.debug_mode,
            }
        )

    # ----------------------------------------------------------- chat
    @app.route("/chat", methods=["POST"])
    @rate_limit(config.rate_limit_chat, scope="chat")
    def chat():
        message, error = read_message()
        if error:
            return error
        session_id = get_session_id()
        log.info("[chat] %s", message[:120])
        return ndjson_stream(
            services.chat.stream(message, session_id, voice=False, debug=services.debug_mode)
        )

    @app.route("/api/sina-chat", methods=["POST"])
    @rate_limit(config.rate_limit_chat, scope="chat")
    def sina_chat():
        message, error = read_message()
        if error:
            return error
        session_id = get_session_id()
        log.info("[sina] %s", message[:120])
        return ndjson_stream(
            services.chat.stream(message, session_id, voice=True, debug=services.debug_mode)
        )

    @app.route("/api/session/reset", methods=["POST"])
    def reset_session():
        session_id = session.get("chat_session_id")
        if session_id:
            services.conversations.clear(session_id)
        return jsonify({"success": True})

    # ------------------------------------------------------- speech i/o
    @app.route("/api/tts", methods=["POST"])
    @rate_limit(config.rate_limit_tts, scope="tts")
    def tts():
        body = json_body()
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "No text provided"}), 400
        if len(text) > config.max_tts_chars * 4:
            return jsonify({"error": "Text too long"}), 413
        try:
            audio = services.tts.synthesize(text, voice=body.get("voice") or None)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            log.warning("TTS failed: %s", exc)
            return jsonify({"error": safe_error(exc, "Speech synthesis is unavailable")}), 503
        return Response(
            audio,
            mimetype="audio/mpeg",
            headers={
                "Content-Length": str(len(audio)),
                "Cache-Control": "private, max-age=300",
            },
        )

    @app.route("/api/transcribe", methods=["POST"])
    @rate_limit(config.rate_limit_transcribe, scope="transcribe")
    def transcribe():
        if not config.groq_api_key:
            return jsonify({"error": "Transcription is not configured on this server"}), 503
        audio = request.files.get("audio")
        if audio is None:
            return jsonify({"error": "No audio file provided"}), 400

        content_type = (audio.content_type or "").lower()
        if content_type and not content_type.startswith(("audio/", "video/webm")):
            return jsonify({"error": "Unsupported audio format"}), 415

        try:
            response = services.http.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {config.groq_api_key}"},
                files={
                    "file": (
                        audio.filename or "audio.webm",
                        audio.stream,
                        audio.content_type or "audio/webm",
                    ),
                    "model": (None, config.transcribe_model),
                    "language": (None, "en"),
                    "response_format": (None, "json"),
                },
                timeout=(8, 40),
            )
        except requests.RequestException as exc:
            log.warning("Transcription request failed: %s", exc)
            return jsonify({"error": "Transcription service unreachable"}), 503

        if response.status_code != 200:
            log.warning("Groq transcription error %s: %s", response.status_code, response.text[:200])
            # Never forward the upstream body: it can echo the API key context.
            return jsonify({"error": "Transcription failed"}), 502

        text = (response.json().get("text") or "").strip()
        return jsonify({"text": text[: config.max_message_chars]})

    # -------------------------------------------------------- feedback
    @app.route("/api/report", methods=["POST"])
    @rate_limit(config.rate_limit_report, scope="report")
    def report_issue():
        body = json_body()
        try:
            rating = int(body.get("rating", 0))
        except (TypeError, ValueError):
            rating = 0
        if not 1 <= rating <= 5:
            return jsonify({"error": "Rating must be between 1 and 5"}), 400

        source = body.get("source")
        source = source if source in ("chat", "sina") else "chat"
        services.analytics.record_report(
            query=str(body.get("query") or "")[:2000],
            response=str(body.get("response") or "")[:5000],
            rating=rating,
            reason=str(body.get("reason") or "")[:2000],
            source=source,
        )
        return jsonify({"success": True})

    # ----------------------------------------------------------- assets
    @app.route("/vrm-model")
    def vrm_model():
        """Serve the 18 MB avatar with caching and range support.

        Previously this was re-downloaded in full on every page load: no ETag,
        no Last-Modified, no ``Cache-Control``.
        """
        if not config.vrm_path.exists():
            return jsonify({"error": "VRM model not found"}), 404
        response = send_file(
            config.vrm_path,
            mimetype="model/gltf-binary",
            conditional=True,
            max_age=60 * 60 * 24 * 30,
        )
        response.headers["Cache-Control"] = "public, max-age=2592000, immutable"
        return response

    # ------------------------------------------------------------ admin
    @app.route("/admin/login", methods=["GET", "POST"])
    @rate_limit(config.rate_limit_login, scope="login")
    def admin_login():
        if request.method == "GET":
            if session.get("admin_logged_in"):
                return redirect(url_for("admin_dashboard"))
            return render_template("login.html")

        body = json_body()
        username = str(body.get("username") or "").strip()[:120]
        password = str(body.get("password") or "")
        if not username or not password:
            return jsonify({"success": False, "error": "Username and password required"}), 400
        if not services.analytics.enabled:
            return jsonify({"success": False, "error": "Admin database not available"}), 503

        try:
            user = services.analytics.find_admin(username)
        except Exception as exc:
            log.error("Login lookup failed: %s", exc)
            return jsonify({"success": False, "error": "Login is temporarily unavailable"}), 503

        valid = False
        needs_rehash = False
        if user:
            valid, needs_rehash = verify_password(password, user.get("password_hash") or "")

        if not valid:
            # One generic message: never reveal whether the username exists.
            log.warning("Failed admin login for %r from %s", username, client_ip())
            return jsonify({"success": False, "error": "Invalid credentials"}), 401

        start_admin_session(username, user.get("id"))
        services.analytics.touch_login(user["id"])
        if needs_rehash:
            services.analytics.upgrade_password(user["id"], hash_password(password))
            log.info("Upgraded stored password hash for %r", username)

        return jsonify(
            {"success": True, "redirect": "/admin/dashboard", "csrf_token": csrf_token()}
        )

    @app.route("/admin/logout")
    def admin_logout():
        session.clear()
        return redirect(url_for("admin_login"))

    @app.route("/admin/dashboard")
    @login_required
    def admin_dashboard():
        return render_template(
            "admin.html",
            username=session.get("admin_username", "Admin"),
            csrf_token=csrf_token(),
        )

    @app.route("/api/admin/check")
    def admin_check():
        return jsonify(
            {
                "is_admin": bool(session.get("admin_logged_in")),
                "username": session.get("admin_username"),
            }
        )

    # -------------------------------------------------------- analytics
    def _analytics(fn, *args, **kwargs):
        try:
            payload = fn(*args, **kwargs)
        except Exception as exc:
            log.error("Analytics query failed: %s", exc)
            return jsonify({"error": safe_error(exc, "Could not load analytics")}), 500
        if isinstance(payload, dict) and payload.get("error"):
            return jsonify(payload), 503
        return jsonify(payload)

    @app.route("/api/analytics/stats")
    @api_login_required
    def analytics_stats():
        return _analytics(services.analytics.stats)

    @app.route("/api/analytics/top-questions")
    @api_login_required
    def analytics_top_questions():
        limit = min(max(request.args.get("limit", 10, type=int) or 10, 1), 50)
        return _analytics(services.analytics.top_questions, limit)

    @app.route("/api/analytics/all-questions")
    @api_login_required
    def analytics_all_questions():
        return _analytics(
            services.analytics.all_questions,
            request.args.get("page", 1, type=int) or 1,
            request.args.get("per_page", 20, type=int) or 20,
            (request.args.get("search", "") or "").strip()[:120],
        )

    @app.route("/api/analytics/token-usage")
    @api_login_required
    def analytics_token_usage():
        return _analytics(services.analytics.token_usage)

    @app.route("/api/analytics/generate-summary", methods=["POST"])
    @api_login_required
    @require_csrf
    def analytics_summary():
        """AI summary of recent activity.

        The dashboard has always called this endpoint; it never existed in the
        backend, so the button returned a 404 page into the results panel.
        """
        if not services.client.configured:
            return jsonify({"error": "Language model is not configured"}), 503
        try:
            stats = services.analytics.stats()
            top = services.analytics.top_questions(15).get("questions", [])
        except Exception as exc:
            log.error("Summary data fetch failed: %s", exc)
            return jsonify({"error": "Could not load analytics"}), 500

        # The dashboard renders these alongside the summary text.
        meta = {
            "analyzed_chats": stats.get("total_questions", 0),
            "unique_questions": len(top),
            "generated_at": utcnow_iso(),
        }
        if not top:
            return jsonify({"summary": "No questions have been asked yet.", **meta})

        lines = "\n".join(f"- {item['question']} ({item['count']}x)" for item in top)
        prompt = (
            "You are an analytics assistant for a college chatbot. Write a short "
            "Markdown briefing (max 180 words) for the administrator.\n\n"
            f"Total questions: {stats.get('total_questions')}\n"
            f"Sessions: {stats.get('total_sessions')}\n"
            f"Questions today: {stats.get('today_questions')}\n\n"
            f"Most asked questions:\n{lines}\n\n"
            "Cover: what students most want to know, any gaps the college should "
            "publish better, and one concrete recommendation. Use bullets."
        )
        try:
            data = services.client.post_json(
                "/chat/completions",
                {
                    "model": config.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.4,
                },
                read_timeout=40.0,
            )
            summary = (data["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            log.error("Summary generation failed: %s", exc)
            return jsonify({"error": "Could not generate summary"}), 502
        return jsonify({"summary": summary or "No summary produced.", **meta})

    @app.route("/api/admin/reports")
    @api_login_required
    def admin_reports():
        try:
            return jsonify(services.analytics.reports())
        except Exception as exc:
            log.error("Reports query failed: %s", exc)
            return jsonify({"error": "Could not load reports"}), 500

    # ------------------------------------------------------ maintenance
    @app.route("/api/models")
    def models():
        return jsonify(
            {
                "current_model": config.chat_model,
                "embedding_model": config.embedding_model,
                "provider": "nvidia",
                "configured": services.client.configured,
            }
        )

    @app.route("/api/debug/status")
    @api_login_required
    def debug_status():
        return jsonify({"debug_mode": services.debug_mode})

    @app.route("/api/debug/toggle", methods=["POST"])
    @api_login_required
    @require_csrf
    def debug_toggle():
        services.debug_mode = not services.debug_mode
        return jsonify({"success": True, "debug_mode": services.debug_mode})

    @app.route("/api/embeddings/regenerate", methods=["POST"])
    @api_login_required
    @require_csrf
    def regenerate():
        """Rebuild the index. Admin-only: it is slow and costs API quota."""
        force = bool(json_body().get("force"))
        services.response_cache.clear()
        if force:
            services.embedding_cache.clear()
        ok = services.knowledge.build(force=force)
        services.analytics.invalidate()
        payload = {"success": ok, "knowledge": services.knowledge.stats()}
        return (jsonify(payload), 200) if ok else (jsonify(payload), 500)

    @app.route("/api/cache/clear", methods=["POST"])
    @api_login_required
    @require_csrf
    def clear_cache():
        services.response_cache.clear()
        services.analytics.invalidate()
        return jsonify({"success": True})

    @app.route("/api/nvidia-key", methods=["POST"])
    @api_login_required
    @require_csrf
    def set_nvidia_key():
        api_key = str(json_body().get("api_key") or "").strip()
        if not api_key or len(api_key) < 20:
            return jsonify({"success": False, "error": "A valid API key is required"}), 400
        services.client.set_keys([api_key] + [k for k in config.nvidia_api_keys if k != api_key])
        log.info("NVIDIA API key updated at runtime by %s", session.get("admin_username"))
        return jsonify(
            {
                "success": True,
                "message": "API key updated for this process only. Set NVIDIA_API_KEY "
                "in the environment to make it permanent.",
            }
        )

    @app.route("/api/nvidia-key/status")
    @api_login_required
    def nvidia_key_status():
        return jsonify({"configured": services.client.configured, "keys": services.client.key_count})

    # ------------------------------------------------------ error pages
    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return redirect(url_for("home"))

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "Upload too large"}), 413

    @app.errorhandler(429)
    def too_many(_error):
        return jsonify({"error": "Too many requests. Please slow down."}), 429

    @app.errorhandler(500)
    def server_error(error):
        log.exception("Unhandled error: %s", error)
        return jsonify({"error": "Internal server error"}), 500
