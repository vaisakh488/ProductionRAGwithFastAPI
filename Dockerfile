# ══════════════════════════════════════════════════════════
# Dockerfile — slim image for api, celery_beat, flower
#
# NO torch, NO sentence-transformers, NO CUDA packages.
# Those only belong in Dockerfile.worker (celery_worker).
#
# Expected final image size: ~600–800 MB

# ══════════════════════════════════════════════════════════

# ── Stage 1: builder ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# [R1] curl removed — not used in any builder RUN command.
# curl is only needed in the runtime stage for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# requirements-base.txt has NO sentence-transformers, NO torch
COPY requirements-base.txt .

RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir \
    --prefix=/install/packages \
    -r requirements-base.txt


# ── Stage 2: runtime ──────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install/packages /usr/local

# [R2] Create user BEFORE COPY so --chown can be used directly.
# This avoids the separate chown -R layer which duplicated all file
# inodes and bloated the image size.
RUN useradd -m -u 1000 appuser \
 && mkdir -p /app/pdfs \
 && chown appuser:appuser /app/pdfs

COPY --chown=appuser:appuser . .

USER appuser

# [R3] EXPOSE before CMD — standard Dockerfile convention.
EXPOSE 8000

# Production (docker-compose): overridden with gunicorn (see docker-compose.yml)
# Direct run / dev: single uvicorn process
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]