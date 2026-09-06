FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# curl is only needed for the healthcheck; no build toolchain is required now
# that torch/chromadb are gone, which keeps the image small and the build fast.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.rngai_cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# gthread handles the streaming responses well; each worker builds/loads the
# vector index once at boot from the persisted cache.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 8 --worker-class gthread --timeout 180 --graceful-timeout 30 --access-logfile - app:app"]
