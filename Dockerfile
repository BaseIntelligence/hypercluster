FROM python:3.12-slim AS runtime

# BASE may override CHALLENGE_DATABASE_URL under swarm, but defaults stay
# challenge-owned SQLite on the /data volume (never BASE_DATABASE_URL).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHALLENGE_HOST=0.0.0.0 \
    CHALLENGE_PORT=8000 \
    CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////data/challenge.sqlite3

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

# Vendored Base SDK wheel (offline-friendly; staged by CI/tests from platform/dist).
# When missing, fall back to the hash-pinned release URL declared in pyproject.
COPY docker/vendor/ /tmp/base-wheels/
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && if ls /tmp/base-wheels/base-*.whl >/dev/null 2>&1; then \
         pip install --no-cache-dir /tmp/base-wheels/base-*.whl; \
       fi \
    && pip install --no-cache-dir . \
    && rm -rf /tmp/base-wheels

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); raise SystemExit(0 if data.get('status') == 'ok' else 1)"

CMD ["uvicorn", "hypercluster.app:app", "--host", "0.0.0.0", "--port", "8000"]
