# Production RAG System — FastAPI + LangGraph + Qdrant

A production-grade Retrieval-Augmented Generation (RAG) system built to handle **5,000+ PDFs** with hybrid retrieval, async ingestion, persistent conversation memory, and full observability. Not a tutorial — a real deployment.

![Architecture](architecture.png)

---

## Table of Contents

- [What makes this production-grade](#what-makes-this-production-grade)
- [Architecture overview](#architecture-overview)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Environment configuration](#environment-configuration)
- [Service endpoints](#service-endpoints)
- [API reference](#api-reference)
- [Ingestion pipeline](#ingestion-pipeline)
- [Retrieval pipeline](#retrieval-pipeline)
- [Authentication and roles](#authentication-and-roles)
- [Monitoring and observability](#monitoring-and-observability)
- [Docker image breakdown](#docker-image-breakdown)
- [Windows vs Linux differences](#windows-vs-linux-differences)
- [Scaling guidance](#scaling-guidance)
- [Troubleshooting](#troubleshooting)
- [Production checklist](#production-checklist)

---

## What makes this production-grade

| Feature | Standard RAG | This system |
|---|---|---|
| Retrieval | Dense-only (cosine) | Dense + Sparse (BM25) + Cross-encoder rerank |
| Ingestion | Sync, blocks API | Async Celery workers, API returns instantly |
| Conversation memory | In-process dict | PostgreSQL checkpoints, survives restarts |
| Concurrency | Single process | 4 Gunicorn workers + N Celery workers |
| Auth | None / basic | JWT + refresh tokens + RBAC + token blocklist |
| Monitoring | None | Prometheus + Grafana + Loki + LangSmith |
| Error recovery | None | Celery retry ×3 with exponential backoff |
| PDF handling | Sequential | Parallel (ThreadPool/ProcessPool) |

---

## Architecture overview

```
Client
  │  JWT Bearer auth · Rate limited · CORS
  ▼
FastAPI (Gunicorn, 4 × UvicornWorker)
  │  /auth  /ingest  /chat  /jobs  /health  /metrics
  ├──► LangGraph agent (query rewrite → retrieve → rerank → generate)
  │      ├── Qdrant       dense retrieval (text-embedding-3-small, 1536-dim)
  │      ├── BM25         sparse retrieval (in-memory, Redis-cached)
  │      └── PostgreSQL   conversation checkpoints (AsyncPostgresSaver)
  │
  ├──► Redis broker ──► Celery worker
  │                        └── Ingest pipeline
  │                              PDF parse → dedup → embed → semantic merge
  │                              → Qdrant upsert → BM25 rebuild → Redis persist
  │
  └──► Observability
         Prometheus :9090 · Grafana :3001 · Loki :3100 · Flower :5555
```

---

## Tech stack

**AI / ML**
- `GPT-4o-mini` — generation and query rewriting
- `text-embedding-3-small` — 1536-dim embeddings, batch 64
- `BAAI/bge-reranker-base` — cross-encoder reranking (worker image)
- `FlashRank ms-marco-MiniLM-L-12-v2` — lightweight reranker fallback (API image)
- `BM25Okapi` — sparse retrieval
- `LangGraph` — agent orchestration with stateful graph
- `LangSmith` — LLM call tracing

**Backend**
- `FastAPI` + `Gunicorn` + `UvicornWorker`
- `Celery` + `Redis` broker
- `asyncpg` — async PostgreSQL connection pooling
- `psycopg3` + `psycopg-pool` — LangGraph checkpointer
- `slowapi` — rate limiting (Redis-backed)

**Storage**
- `Qdrant v1.13` — vector store, COSINE distance, payload indexes
- `PostgreSQL 16` — users, jobs, documents, conversation checkpoints
- `Redis 7.2` — broker, BM25 cache, token blocklist, job state, rate limits

**Observability**
- `Prometheus` — metrics scraping
- `Grafana` — dashboards (auto-provisioned)
- `Loki` — log aggregation
- `Promtail` — log shipping (Linux only)
- `Flower` — Celery task monitoring

**Infrastructure**
- `Docker Compose` — 11-container orchestration
- Multi-stage Dockerfiles — slim API image (~800 MB) + heavy worker image (~2 GB)

---

## Project structure

```
ProductionRAGwithFastAPI/
├── .dockerignore                 # prevents secrets/git from entering images
├── .env                          # secrets — NEVER commit this file
├── .env.example                  # safe template — commit this instead
├── docker-compose.yml            # full stack orchestration
├── Dockerfile                    # slim image: api, celery_beat, flower
├── Dockerfile.worker             # heavy image: celery_worker (torch + reranker)
├── requirements-base.txt         # shared deps (api, beat, flower)
├── requirements-worker.txt       # worker-only: sentence-transformers, torch
├── requirements.txt              # dev reference only (project runs in Docker)
│
├── main.py                       # FastAPI app, lifespan startup, all endpoints
├── auth.py                       # JWT auth, RBAC, user management, token blocklist
├── agent.py                      # LangGraph RAG agent, retrieval + rerank pipeline
├── ingest.py                     # PDF parsing, embedding, Qdrant upsert, BM25
├── tasks.py                      # Celery task definitions, worker process lifecycle
│
├── prometheus.yml                # Prometheus scrape targets
├── promtail-config.yml           # Promtail log scrape config (Linux only)
│
├── grafana/
│   └── provisioning/
│       └── datasources/
│           └── datasources.yml   # auto-provisioned Prometheus + Loki datasources
│
└── pdfs/                         # runtime upload directory — auto-created, gitignored
```

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Docker Desktop | 4.x | Windows or Linux |
| Docker Compose | v2.x | bundled with Docker Desktop |
| OpenAI API key | — | GPT-4o-mini + text-embedding-3-small |
| LangSmith API key | — | optional, for tracing |
| RAM | 8 GB | 12 GB+ recommended for worker |
| Disk | 10 GB free | worker image ~2 GB, model weights ~400 MB |

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ProductionRAGwithFastAPI.git
cd ProductionRAGwithFastAPI
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in every required value — see [Environment configuration](#environment-configuration) for details.

### 3. Build and start all services

```bash
docker compose up --build -d
```

First build takes 10–15 minutes (downloads model weights into the worker image).

### 4. Verify everything is healthy

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

All containers should show `healthy` or `running`. Expected output:

```
NAMES                  STATUS
rag_api                Up 2 minutes (healthy)
rag_celery_worker      Up 2 minutes (healthy)
rag_celery_beat        Up 2 minutes
rag_flower             Up 2 minutes
rag_prometheus         Up 2 minutes
rag_grafana            Up 2 minutes
rag_loki               Up 2 minutes (healthy)
ragdb                  Up 2 minutes (healthy)
redis                  Up 2 minutes (healthy)
qdrant                 Up 2 minutes
```

### 5. Access the services

| Service | URL | Credentials |
|---|---|---|
| API docs | http://localhost:8000/docs | — |
| Grafana | http://localhost:3001 | admin / `GRAFANA_PASSWORD` |
| Flower | http://localhost:5555 | admin / `FLOWER_PASSWORD` |
| Prometheus | http://localhost:9090 | — |

### 6. Start with log shipping (Linux only)

```bash
docker compose --profile linux-monitoring up --build -d
```

---

## Environment configuration

Create `.env` in the project root. **Never commit this file.**

```bash
# ── OpenAI ──────────────────────────────────────────────────
OPENAI_API_KEY=sk-...                    # required

# ── LangSmith tracing (optional) ────────────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=ProductionRAG

# ── Databases ───────────────────────────────────────────────
POSTGRES_PASSWORD=your_strong_password   # used by postgres container

# IMPORTANT: DATABASE_URL must use the SAME password as POSTGRES_PASSWORD
# .env files do NOT support variable interpolation — both must be set manually
DATABASE_URL=postgresql://postgres:your_strong_password@postgres:5432/ragdb

# IMPORTANT: REDIS_URL must use the SAME password as REDIS_PASSWORD
REDIS_PASSWORD=your_strong_redis_password
REDIS_URL=redis://:your_strong_redis_password@redis:6379

# ── Vector store ────────────────────────────────────────────
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=documents

# ── Application ─────────────────────────────────────────────
PDF_DIR=/app/pdfs
RERANKER_BACKEND=baai                    # baai | flashrank

# ── Security ────────────────────────────────────────────────
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=your_64_char_hex_secret

# Set ONCE for first startup, then REMOVE this line.
# Leaving it set resets the admin password on every container restart.
ADMIN_PASSWORD=your_initial_admin_password

ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# ── CORS ────────────────────────────────────────────────────
# Comma-separated list of allowed frontend origins
ALLOWED_ORIGINS=http://localhost:3000

# ── Celery ──────────────────────────────────────────────────
CELERY_WORKER_CONCURRENCY=2             # tune to your RAM and OpenAI tier
GUNICORN_WORKERS=4

# ── Monitoring ──────────────────────────────────────────────
FLOWER_USER=admin
FLOWER_PASSWORD=your_flower_password
GRAFANA_PASSWORD=your_grafana_password

# ── Feature flags ───────────────────────────────────────────
DOCS_ENABLED=false                      # set true in dev, false in production
DEBUG_ENDPOINTS_ENABLED=false           # never true in production
```

### Critical password rules

> ⚠️ `.env` files do NOT support variable interpolation like `${POSTGRES_PASSWORD}`.
> `REDIS_URL` and `DATABASE_URL` must contain the literal password value.
> If you change `POSTGRES_PASSWORD`, you MUST also update `DATABASE_URL` manually.

### After first startup

Once the system starts successfully and you can log in as admin, **remove `ADMIN_PASSWORD` from `.env`**. Leaving it set causes the admin password to reset to that value on every container restart, overwriting any password changes you make through the API.

---

## Service endpoints

### API — `http://localhost:8000`

| Method | Endpoint | Auth | Role | Description |
|---|---|---|---|---|
| `POST` | `/auth/login` | — | — | Login, get JWT + refresh token |
| `POST` | `/auth/refresh` | refresh token | — | Rotate tokens |
| `POST` | `/auth/logout` | Bearer | any | Invalidate token |
| `GET` | `/auth/me` | Bearer | any | Current user info |
| `POST` | `/users` | Bearer | admin | Create user |
| `GET` | `/users` | Bearer | admin | List all users |
| `PATCH` | `/users/{username}/deactivate` | Bearer | admin | Deactivate user |
| `POST` | `/ingest` | Bearer | editor+ | Upload single PDF |
| `POST` | `/ingest/bulk` | Bearer | editor+ | Upload multiple PDFs |
| `GET` | `/jobs/{job_id}` | Bearer | any | Poll ingest job progress |
| `GET` | `/documents` | Bearer | any | List ingested documents |
| `POST` | `/chat` | Bearer | any | Ask a question |
| `POST` | `/chat/stream` | Bearer | any | Streaming chat response |
| `GET` | `/history/{thread_id}` | Bearer | any | Conversation history |
| `GET` | `/health` | — | — | Service health check |
| `GET` | `/metrics` | — | — | Prometheus metrics |

### Rate limits

| Endpoint | Limit |
|---|---|
| `/auth/login` | 5 / minute |
| `/ingest` | 10 / minute |
| `/ingest/bulk` | 5 / minute |
| `/chat` | 20 / minute |
| `/chat/stream` | 20 / minute |
| `/jobs/{job_id}` | 60 / minute |
| `/documents` | 30 / minute |

---

## API reference

### Login

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your_password"
```

Response:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

### Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "job_id": "b7a86d5b-d298-4528-9391-2fc5b5ee29df",
  "message": "Ingestion queued — poll /jobs/{job_id} for progress",
  "files_queued": 1
}
```

### Poll job progress

```bash
curl http://localhost:8000/jobs/b7a86d5b-d298-4528-9391-2fc5b5ee29df \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

Response:
```json
{
  "job_id": "b7a86d5b-...",
  "status": "completed",
  "progress": 100,
  "total_chunks": 847,
  "elapsed_seconds": 42.3
}
```

Status values: `queued` → `parsing` → `embedding` → `upserting` → `completed` | `failed`

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key provisions?", "thread_id": "user-123-session-1"}'
```

Response:
```json
{
  "answer": "Based on the documents...",
  "thread_id": "user-123-session-1"
}
```

### Streaming chat

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the contract", "thread_id": "user-123"}' \
  --no-buffer
```

Returns `text/event-stream` — tokens stream as `data: <token>\n\n`, ends with `data: [DONE]\n\n`.

---

## Ingestion pipeline

When you POST to `/ingest`, here is exactly what happens:

```
1. API validates file (PDF magic bytes, size ≤ 50 MB)
2. File saved to shared volume /app/pdfs/{job_id}_{filename}
3. Job registered in PostgreSQL (status: pending)
4. Task enqueued to Redis broker — API returns job_id immediately
5. Celery worker picks up task:
   ├── Parse PDFs in parallel (ThreadPoolExecutor in worker, ProcessPoolExecutor in CLI)
   ├── Deduplicate chunks by SHA-256 content hash
   ├── Embed all chunks (OpenAI text-embedding-3-small, batch 64, concurrency 4)
   ├── Semantic merge (cosine similarity ≥ 0.88, re-embed merged chunks)
   ├── Upsert to Qdrant (batch 256, wait=True)
   ├── Record document metadata in PostgreSQL
   ├── Rebuild BM25 index from full Qdrant collection
   └── Persist BM25 index to Redis (compressed pickle, zlib level 1)
6. Job state updated to completed in Redis + PostgreSQL
```

### Chunking parameters

| Parameter | Value |
|---|---|
| Chunk size | 1,000 tokens |
| Chunk overlap | 250 tokens |
| Max chunks per document | 2,000 |
| Max upload size | 50 MB |
| Semantic merge threshold | cosine ≥ 0.88 |
| Embedding model | text-embedding-3-small |
| Embedding dimensions | 1,536 |

### Retry behaviour

Failed jobs retry up to 3 times with exponential backoff: 60s → 120s → 240s.
After all retries are exhausted, uploaded files are cleaned up and the job is marked `failed`.

---

## Retrieval pipeline

Every `/chat` request goes through this pipeline:

```
1. Query rewriter    — LLM rewrites the question for better retrieval
2. Dense retrieval   — Qdrant similarity search, top 30 results
3. Sparse retrieval  — BM25Okapi keyword search, top 30 results
4. RRF merge         — Reciprocal Rank Fusion combines both lists
5. Cross-encoder     — BAAI/bge-reranker-base reranks top 30 → top 8
6. Generator         — GPT-4o-mini answers from context + conversation history
```

### Why hybrid retrieval

Dense search captures semantic meaning but misses exact keywords and rare terms.
BM25 sparse search captures exact matches but has no semantic understanding.
RRF merge combines both result sets without needing to tune score thresholds.
The cross-encoder then applies the most accurate (but expensive) reranking to the final candidates.

### Context limits

| Parameter | Value |
|---|---|
| Dense retrieval top-k | 30 |
| Sparse retrieval top-k | 30 |
| RRF candidates passed to reranker | 30 |
| Reranker output (top-k) | 8 |
| Max context characters sent to LLM | 80,000 |

---

## Authentication and roles

The system uses JWT Bearer tokens with a role hierarchy:

```
viewer  →  read-only: chat, history, documents, jobs
editor  →  viewer + ingest PDFs
admin   →  editor + manage users, view metrics
```

### Default admin account

On first startup, an `admin` account is created automatically.
Set `ADMIN_PASSWORD` in `.env` before starting — the system applies it on first boot.
**Remove `ADMIN_PASSWORD` from `.env` after first login.**

### Token lifecycle

```
Login → access token (60 min) + refresh token (7 days)
Refresh → old refresh token blocklisted → new pair issued
Logout → access token blocklisted in Redis until expiry
```

Blocklisted tokens are stored in Redis with TTL matching their remaining lifetime.
All protected endpoints check the blocklist on every request.

### Create additional users

```bash
curl -X POST http://localhost:8000/users \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "email": "alice@example.com", "password": "secure123", "role": "editor"}'
```

---

## Monitoring and observability

### Grafana dashboards

Access Grafana at `http://localhost:3001` (admin / `GRAFANA_PASSWORD`).

Prometheus and Loki datasources are **auto-provisioned** on startup from `grafana/provisioning/datasources/datasources.yml`.

Recommended dashboard IDs to import (Dashboards → New → Import):

| Dashboard | ID | Shows |
|---|---|---|
| FastAPI metrics | `17175` | Request rate, latency, error rate |
| Python app | `7587` | Memory, CPU, GC |
| Celery monitoring | `14788` | Queue depth, task success/failure |
| Redis | `14091` | Memory, hit rate, connections |

### Key Prometheus metrics

```promql
# Request rate by endpoint
rate(http_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Active ingest jobs
rag_ingest_requests
```

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "5.0.0",
  "postgres": "ok",
  "redis": "ok",
  "qdrant": "ok",
  "celery": "ok",
  "bm25_ready": true
}
```

### Log aggregation (Linux only)

Start with the `linux-monitoring` profile to enable Promtail:

```bash
docker compose --profile linux-monitoring up -d
```

Query logs in Grafana → Explore → Loki:

```logql
# All API logs
{container="rag_api"}

# Errors only
{container="rag_api"} |= "ERROR"

# Ingest job progress
{container="rag_celery_worker"} |= "job_id"
```

---

## Docker image breakdown

| Image | Used by | Size | Contents |
|---|---|---|---|
| `Dockerfile` | api, celery_beat, flower | ~800 MB | FastAPI, Celery, asyncpg, FlashRank — **no torch** |
| `Dockerfile.worker` | celery_worker | ~2 GB | Everything above + PyTorch CPU, sentence-transformers, BAAI model |

The split keeps the API image lean. Torch and the BAAI reranker model (~400 MB) live only in the worker image, which is the only container that actually does reranking.

### Build a single service

```bash
# Rebuild only the API (fast, no ML deps)
docker compose up -d --no-deps --build rag_api

# Rebuild only the worker (slow, downloads/verifies model)
docker compose up -d --no-deps --build celery_worker
```

---

## Windows vs Linux differences

| Feature | Windows Docker Desktop | Linux |
|---|---|---|
| PDF parsing | ThreadPoolExecutor | ProcessPoolExecutor (true parallelism) |
| asyncio policy | WindowsSelectorEventLoopPolicy | default |
| Promtail log shipping | ❌ not supported | ✅ `--profile linux-monitoring` |
| Performance | slightly lower | full CPU parallelism |

Both platforms run the full application stack. Promtail is the only Linux-exclusive component and is optional — metrics and dashboards work on both.

---

## Scaling guidance

### Increase API workers

```bash
# In .env
GUNICORN_WORKERS=8

# Or override at runtime
docker compose up -d --no-deps --build rag_api
```

Note: each worker creates its own asyncpg pool (max_size=20). With N workers:
`N × 20 + 10 (celery) + 10 (langgraph) < max_connections (200)`

With `GUNICORN_WORKERS=8`: 8×20 + 20 = 180 connections — within the 200 limit.

### Increase Celery concurrency

```bash
# In .env — tune to RAM and OpenAI rate limits
# Formula: concurrency × 4 (EMBED_CONCURRENCY) × 64 (BATCH_SIZE) × ~6KB ≈ RAM needed
CELERY_WORKER_CONCURRENCY=4
```

### Scale Celery workers horizontally

```bash
# Run multiple worker containers
docker compose up -d --scale celery_worker=3
```

### Tune PostgreSQL

Increase `shared_buffers` and `work_mem` in `docker-compose.yml` for larger deployments:

```yaml
command: >
  postgres
  -c shared_buffers=512MB    # was 256MB
  -c work_mem=32MB           # was 16MB
  -c max_connections=200
```

---

## Troubleshooting

### Container not starting

```bash
# Check logs for any container
docker logs rag_api --tail 100
docker logs rag_celery_worker --tail 100
```

### Redis connection refused / WRONGPASS

`.env` does not support variable interpolation. `REDIS_URL` must contain the literal password:
```bash
# Wrong
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379

# Correct
REDIS_URL=redis://:your_actual_password@redis:6379
```

### PDF upload permission denied

The `/app/pdfs` directory must be owned by `appuser` (uid 1000).
If you see `[Errno 13] Permission denied: '/app/pdfs/...'`, rebuild the image:
```bash
docker compose up -d --no-deps --build rag_api celery_worker
```

### BM25 division by zero on startup

Normal on a fresh deployment with no PDFs ingested yet. The message:
```
[BM25] Collection is empty — index will build after first ingest
```
is expected and non-fatal. Ingest a PDF and the index builds automatically.

### CREATE INDEX CONCURRENTLY error

If you see `psycopg.errors.ActiveSqlTransaction: CREATE INDEX CONCURRENTLY cannot run inside a transaction block`, your `agent.py` is outdated. The fix requires `kwargs={"autocommit": True}` in the `AsyncConnectionPool` constructor. Pull the latest code and rebuild.

### Grafana shows no datasources

The provisioning file must be at exactly:
```
grafana/provisioning/datasources/datasources.yml
```
Common mistake: `provisoning` (typo, missing `i`) or an extra `resources/` subdirectory.

### Celery tasks not running

```bash
# Check worker is alive
docker logs rag_celery_worker --tail 50

# Check Redis broker connection
docker exec -it redis redis-cli --no-auth-warning -a YOUR_REDIS_PASSWORD ping
# Expected: PONG

# Monitor tasks in real-time
# Open http://localhost:5555 (Flower)
```

### Checking PostgreSQL tables

```bash
docker exec -it ragdb psql -U postgres -d ragdb -c "\dt"
docker exec -it ragdb psql -U postgres -d ragdb -c "SELECT * FROM ingestion_jobs ORDER BY started_at DESC LIMIT 5;"
```

---

## Production checklist

Before going live, verify every item:

**Security**
- [ ] `SECRET_KEY` is a random 64-char hex string (not a guessable value)
- [ ] `ADMIN_PASSWORD` removed from `.env` after first boot
- [ ] `POSTGRES_PASSWORD`, `REDIS_PASSWORD` are strong random passwords
- [ ] `DOCS_ENABLED=false` (Swagger UI disabled)
- [ ] `DEBUG_ENDPOINTS_ENABLED=false`
- [ ] `ALLOWED_ORIGINS` set to your actual frontend domain(s)
- [ ] API is behind a reverse proxy (nginx/traefik) — not exposed directly on port 8000
- [ ] `.env` is in `.gitignore` and never committed

**Reliability**
- [ ] All containers show `healthy` in `docker ps`
- [ ] `GET /health` returns `"status": "ok"` with all services green
- [ ] Test ingest a PDF and verify job reaches `completed` status
- [ ] Test `/chat` returns a coherent answer
- [ ] Test token refresh and logout (verify token is blocklisted)

**Observability**
- [ ] Grafana accessible and both datasources connected (green checkmark)
- [ ] At least one dashboard imported and showing data
- [ ] Prometheus targets all `UP` at `http://localhost:9090/targets`

**Data**
- [ ] PostgreSQL volume mounted and persisting across restarts
- [ ] Qdrant volume mounted and persisting across restarts
- [ ] `celery_beat_data` volume exists for persistent Beat schedule

---

## Contributing

Pull requests welcome. Please:
- Keep the slim/heavy image split intact — no ML deps in `requirements-base.txt`
- Add a comment for any non-obvious async pattern
- Test on both Linux and Windows Docker Desktop if possible

---

## License

MIT
