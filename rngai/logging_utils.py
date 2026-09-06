"""Logging setup.

Replaces the scattered ``print()`` calls of the original app with a real
logger, so levels can be tuned per environment and log lines carry timestamps
and module names without hand-formatting each one.
"""

from __future__ import annotations

import logging
import sys

from .config import config

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Windows consoles default to cp1252 and blow up on non-ASCII log records.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, config.log_level, logging.INFO))

    # These libraries are extremely chatty at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "hpack", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
