"""RNGPIT AI Assistant - application entry point.

The implementation lives in the ``rngai`` package; this file wires it up so that
``python app.py``, ``flask run`` and ``gunicorn app:app`` all work.

    python app.py                      # development server
    gunicorn -w 2 -k gthread -t 120 app:app   # production

Configuration comes entirely from the environment - see ``.env.example``.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# TensorFlow/oneDNN chatter from transitive imports.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from rngai import __version__  # noqa: E402
from rngai.config import config  # noqa: E402
from rngai.logging_utils import get_logger  # noqa: E402
from rngai.webapp import create_app, services  # noqa: E402

log = get_logger("rngai")

app = create_app()


def _banner() -> None:
    log.info("=" * 62)
    log.info("RNGPIT AI Assistant v%s", __version__)
    log.info("=" * 62)
    for warning in config.warnings:
        log.warning(warning)
    log.info("Chat model      : %s", config.chat_model)
    log.info("Embedding model : %s", config.embedding_model)
    log.info("Data directory  : %s", config.data_dir)
    log.info("Cache directory : %s", config.cache_dir)


def bootstrap() -> None:
    """Build the knowledge base. Safe to call more than once."""
    if not services.knowledge.ready:
        services.warm_up()


_banner()
bootstrap()


if __name__ == "__main__":
    log.info("Chatbot : http://localhost:%d", config.port)
    log.info("Admin   : http://localhost:%d/admin/login", config.port)
    if config.debug:
        log.warning("FLASK_DEBUG is on - never enable this on a public host")
    try:
        app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)
    except KeyboardInterrupt:
        log.info("Shutting down")
        sys.exit(0)
