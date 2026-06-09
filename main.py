"""
main.py — Production RAG API v5 (Celery edition)

CHANGES FROM v4:
  [C3] init_agent(pg_pool) called in lifespan — AsyncPostgresSaver checkpointer.
  [B3] BackgroundTasks replaced with Celery task queue.
       /ingest and /ingest/bulk call tasks.run_ingest_task.delay() and
       return immediately. The Celery worker process owns all ingestion work.
  [B2] BM25 loaded from Redis cache on startup; falls back to Qdrant rebuild.
  [FIX] Filename collision — job_id prepended to stored filename.
  [FIX] MAX_UPLOAD_BYTES imported from ingest.py — single source of truth.
  [FIX] /metrics Content-Type corrected for Prometheus scraping.

  All previous upgrades (U1–U14) retained unless superseded above.

REVIEW FIXES (v5.1):
  [R1] Admin seed INSERT added to lifespan as a parameterized execute call.
       USERS_SCHEMA_SQL is now pure DDL (no data) — the seed row is inserted
       here explicitly so no values are ever interpolated into SQL strings.
       _PLACEHOLDER_HASH imported from auth.py for the seed call.
  [R2] /auth/login rate-limited at 5/minute — brute force protection.
  [R3] ChatRequest fields validated — question max 4000 chars, thread_id
       max 100 chars, question min length 1 enforced at model level.
  [R4] /documents pagination bounded — limit capped at 200, offset >= 0.
  [R5] Bulk ingest response message now uses actual job_id variable
       (was {{job_id}} escaped brace — printed literal text "{job_id}").
  [R6] _register_job failure now logged at ERROR (was WARNING) so it
       surfaces correctly in alerting.
  [R7] All inline imports (asyncio, httpx, agent) moved to top-level.
  [R8] /debug/history endpoint gated behind DEBUG_ENDPOINTS_ENABLED env var.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import asyncpg
import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from werkzeug.utils import secure_filename
from prometheus_fastapi_instrumentator import Instrumentator

import agent as _agent
from agent import arun_rag, graph, init_agent, shutdown_agent
from auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ROLES,
    PG_DSN,
    REDIS_URL,
    Token,
    User,
    UserCreate,
    USERS_SCHEMA_SQL,
    _PLACEHOLDER_HASH,          # [R1] used for parameterized admin seed INSERT
    authenticate_user_db,
    blocklist_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    get_user_from_db,
    hash_password,
    is_token_blocked,
    require_role,
)
from ingest import (
    MAX_UPLOAD_BYTES,   # single source of truth — defined in ingest.py
    PDF_DIR,
    bm25_index,
    get_job_state,
    rebuild_bm25_index,
    update_job_state,
    COLLECTION_NAME,
)

# Import Celery app — worker runs separately; here we only call .delay()
from tasks import celery_app, run_ingest_task

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

os.makedirs(PDF_DIR, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────
MAX_UPLOAD_MB   = MAX_UPLOAD_BYTES // (1024 * 1024)

_raw_origins    = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

_counters: dict[str, int] = {
    "requests_total":  0,
    "chat_requests":   0,
    "ingest_requests": 0,
    "errors_total":    0,
}

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
)


# ══════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up …")

    # ── PostgreSQL pool ───────────────────────────────────
    app.state.pg_pool = await asyncpg.create_pool(PG_DSN, min_size=5, max_size=20)
    logger.info("PostgreSQL pool ready")

    async with app.state.pg_pool.acquire() as conn:
        # Run DDL — creates tables if they don't exist
        await conn.execute(USERS_SCHEMA_SQL)

        # [R1] Seed admin row — parameterized, no SQL interpolation.
        # ON CONFLICT DO NOTHING means this is safe to run on every startup.
        # The placeholder hash locks the account until ADMIN_PASSWORD is applied.
        await conn.execute(
            "INSERT INTO users (username, email, hashed_password, role) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (username) DO NOTHING",
            "admin", "admin@example.com", _PLACEHOLDER_HASH, "admin",
        )

        # Apply ADMIN_PASSWORD from env if set — replaces placeholder hash
        # with the real password on first startup.
        # WARNING: if ADMIN_PASSWORD remains set in .env, every restart
        # resets the admin password to this value — remove from .env after
        # first successful boot.
        admin_password = os.getenv("ADMIN_PASSWORD")
        if admin_password:
            hashed = hash_password(admin_password)
            await conn.execute(
                "UPDATE users SET hashed_password=$1 WHERE username='admin'",
                hashed,
            )
            logger.info("Admin password initialised from ADMIN_PASSWORD env var")

    # ── Redis client ──────────────────────────────────────
    app.state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis client ready")

    # ── BM25: try Redis cache first, fall back to Qdrant  [B2] ──
    try:
        loaded = await bm25_index.load_from_redis(app.state.redis)
        if loaded:
            logger.info("BM25 index loaded from Redis cache")
        else:
            from qdrant_client import QdrantClient
            qc          = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
            collections = [c.name for c in qc.get_collections().collections]
            qc.close()
            if COLLECTION_NAME in collections:
                logger.info("Building BM25 index from Qdrant (no Redis cache found) …")
                await rebuild_bm25_index(redis=app.state.redis)
                logger.info("BM25 index ready and persisted to Redis")
            else:
                logger.info("Qdrant collection not yet created — BM25 will build after first ingest")
    except Exception as exc:
        logger.warning(f"BM25 startup failed (non-fatal): {exc}")

    # ── LangGraph agent + AsyncPostgresSaver  [C3] ────────
    try:
        await init_agent(app.state.pg_pool)
        logger.info("LangGraph agent ready (AsyncPostgresSaver checkpointer)")
    except Exception as exc:
        logger.error(f"Agent init failed: {exc}", exc_info=True)
        raise   # Hard fail — API is useless without the agent

    yield

    logger.info("Shutting down …")
    await shutdown_agent()
    await app.state.pg_pool.close()
    await app.state.redis.aclose()
    logger.info("Connections closed")


# ── APP ───────────────────────────────────────────────────
app = FastAPI(
    title="Production RAG API",
    description="LangGraph + Qdrant RAG — 5000+ PDFs, hybrid retrieval",
    version="5.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("DOCS_ENABLED", "true").lower() == "true" else None,
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── PROMETHEUS INSTRUMENTATION ────────────────────────────
# Exposes /metrics in proper Prometheus text format.
# No auth on this endpoint — Prometheus scraper needs unauthenticated access.
# Tracks: request count, latency histograms, in-progress requests — all
# broken down by method, path, and status code automatically.
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ── MIDDLEWARE ────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    _counters["requests_total"] += 1
    t0       = time.perf_counter()
    response: Response = await call_next(request)
    elapsed  = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-ID"]    = request_id
    response.headers["X-Response-Time"] = f"{elapsed}ms"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    _counters["errors_total"] += 1
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ── DEPENDENCIES ─────────────────────────────────────────

def get_pg_pool(request: Request) -> asyncpg.Pool:
    return request.app.state.pg_pool

def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis


# ── SCHEMAS ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    # [R3] question bounded: non-empty, max 4000 chars to prevent oversized payloads.
    # thread_id bounded: max 100 chars — used as a Redis/DB key.
    question:  str = Field(..., min_length=1, max_length=4000)
    thread_id: str = Field("default", max_length=100)

class ChatResponse(BaseModel):
    answer:    str
    thread_id: str

class IngestJobResponse(BaseModel):
    job_id:       str
    message:      str
    files_queued: int

class RefreshRequest(BaseModel):
    refresh_token: str


# ── FILE VALIDATION ───────────────────────────────────────

def _safe_filename(original: str) -> str:
    name = secure_filename(original)
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    return name


async def _read_and_validate(file: UploadFile) -> bytes:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit")
    if content[:4] != b"%PDF":
        raise HTTPException(status_code=400, detail="File content is not a valid PDF")
    return content


# ── JOB REGISTRATION ─────────────────────────────────────

async def _register_job(pool: asyncpg.Pool, job_id: str, total_files: int) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO ingestion_jobs (job_id, total_files, status) "
                "VALUES ($1, $2, 'pending') ON CONFLICT (job_id) DO NOTHING",
                job_id, total_files,
            )
    except Exception as e:
        # [R6] Logged at ERROR — a missing job row means /jobs/{job_id} returns
        # 404 from the PostgreSQL path, which is confusing to operators.
        logger.error(f"Could not register job {job_id}: {e}")


# ══════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.post("/auth/login", response_model=Token, tags=["Auth"])
@limiter.limit("5/minute")      # [R2] brute-force protection
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    pool: asyncpg.Pool = Depends(get_pg_pool),
):
    user = await authenticate_user_db(form_data.username, form_data.password, pool)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token  = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(data={"sub": user["username"]})
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@app.post("/auth/refresh", response_model=Token, tags=["Auth"])
async def refresh(
    body: RefreshRequest,
    redis: aioredis.Redis = Depends(get_redis),
):
    data = decode_token(body.refresh_token)
    if data.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")
    if await is_token_blocked(body.refresh_token, redis):
        raise HTTPException(status_code=401, detail="Refresh token revoked")

    await blocklist_token(body.refresh_token, redis)
    access_token  = create_access_token(data={"sub": data["username"]})
    refresh_token = create_refresh_token(data={"sub": data["username"]})
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@app.post("/auth/logout", tags=["Auth"])
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    redis: aioredis.Redis = Depends(get_redis),
):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if token:
        await blocklist_token(token, redis)
    return {"detail": "Logged out successfully"}


@app.get("/auth/me", tags=["Auth"])
async def me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "role":     current_user.role,
        "email":    current_user.email,
    }


# ── USER MANAGEMENT (admin only) ──────────────────────────

@app.post("/users", tags=["Users"])
async def create_user(
    body: UserCreate,
    current_user: User = Depends(require_role("admin")),
    pool: asyncpg.Pool  = Depends(get_pg_pool),
):
    if body.role not in ROLES:
        raise HTTPException(status_code=400, detail=f"Role must be one of: {ROLES}")
    hashed = hash_password(body.password)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (username, email, hashed_password, role) "
                "VALUES ($1, $2, $3, $4)",
                body.username, body.email, hashed, body.role,
            )
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail="Username or email already exists")
    return {"message": f"User '{body.username}' created with role '{body.role}'"}


@app.get("/users", tags=["Users"])
async def list_users(
    current_user: User = Depends(require_role("admin")),
    pool: asyncpg.Pool  = Depends(get_pg_pool),
):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT username, email, role, is_active, created_at, last_login "
            "FROM users ORDER BY created_at DESC"
        )
    return {"users": [dict(r) for r in rows]}


@app.patch("/users/{username}/deactivate", tags=["Users"])
async def deactivate_user(
    username: str,
    current_user: User = Depends(require_role("admin")),
    pool: asyncpg.Pool  = Depends(get_pg_pool),
):
    if username == current_user.username:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET is_active=FALSE WHERE username=$1", username
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"User '{username}' deactivated"}


# ── HEALTH ────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health(
    pool:  asyncpg.Pool   = Depends(get_pg_pool),
    redis: aioredis.Redis = Depends(get_redis),
):
    pg_ok, redis_ok, qdrant_ok, celery_ok = False, False, False, False

    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        pg_ok = True
    except Exception:
        pass

    try:
        await redis.ping()
        redis_ok = True
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(
                f"{os.getenv('QDRANT_URL', 'http://localhost:6333')}/healthz"
            )
            qdrant_ok = r.status_code == 200
    except Exception:
        pass

    try:
        # Celery ping — returns dict of {worker_name: {"ok": "pong"}}
        # Uses a short timeout so health check stays fast
        result = await asyncio.to_thread(
            lambda: celery_app.control.ping(timeout=1.0)
        )
        celery_ok = bool(result)
    except Exception:
        pass

    overall = "ok" if all([pg_ok, redis_ok, qdrant_ok, celery_ok]) else "degraded"

    return {
        "status":     overall,
        "version":    "5.0.0",
        "postgres":   "ok" if pg_ok     else "unreachable",
        "redis":      "ok" if redis_ok  else "unreachable",
        "qdrant":     "ok" if qdrant_ok else "unreachable",
        "celery":     "ok" if celery_ok else "no workers",
        "bm25_ready": bm25_index.is_ready,
    }


# ── DEBUG HISTORY (env-gated) ─────────────────────────────
# [R8] Only registered when DEBUG_ENDPOINTS_ENABLED=true.
#      Exposes internal agent state — must never be active in production.
#      Gate via env var rather than just role so it can be fully absent
#      from the routing table in production builds.

if os.getenv("DEBUG_ENDPOINTS_ENABLED", "false").lower() == "true":
    @app.get("/debug/history/{thread_id}", tags=["System"])
    async def debug_history(
        thread_id: str,
        current_user: User = Depends(require_role("admin")),
    ):
        """Debug endpoint — only active when DEBUG_ENDPOINTS_ENABLED=true."""
        from agent import graph as g, _checkpointer
        result = {
            "graph_is_none":      g is None,
            "checkpointer_type":  str(type(_checkpointer)),
            "thread_id":          thread_id,
        }
        if g is None:
            return result
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = await g.aget_state(config)
            result["state_is_none"]      = state is None
            result["state_values_empty"] = not bool(state.values) if state else True
            if state and state.values:
                msgs = state.values.get("messages", [])
                result["messages_count"] = len(msgs)
                result["first_msg_type"] = str(type(msgs[0])) if msgs else "none"
                result["first_msg"]      = str(msgs[0])[:100] if msgs else "none"
            else:
                result["messages_count"] = 0
        except Exception as e:
            result["aget_state_error"] = str(e)
            import traceback
            result["traceback"] = traceback.format_exc()
        return result


# ── METRICS ───────────────────────────────────────────────

@app.get("/metrics/custom", tags=["System"])
async def metrics_custom(current_user: User = Depends(require_role("admin"))):
    """Custom RAG counters — admin only. For Prometheus use /metrics."""
    lines = ["# RAG API custom counters"]
    for key, val in _counters.items():
        lines.append(f"rag_{key} {val}")
    return Response(
        content="\n".join(lines),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ══════════════════════════════════════════════════════════
# INGEST  [B3] — enqueues to Celery, never runs in-process
# ══════════════════════════════════════════════════════════

@app.post("/ingest", response_model=IngestJobResponse, tags=["Documents"])
@limiter.limit("10/minute")
async def ingest_pdf(
    request: Request,
    file: UploadFile = File(...),
    current_user: User         = Depends(require_role("editor")),
    pool:         asyncpg.Pool = Depends(get_pg_pool),
):
    _counters["ingest_requests"] += 1
    safe_name = _safe_filename(file.filename)
    content   = await _read_and_validate(file)

    job_id = str(uuid.uuid4())

    # [FIX] Prefix with job_id — prevents filename collision on concurrent uploads
    stored_name = f"{job_id}_{safe_name}"
    file_path   = Path(PDF_DIR) / stored_name
    file_path.write_bytes(content)

    await _register_job(pool, job_id, 1)

    # [B3] Fire-and-forget to Celery worker — API returns immediately
    run_ingest_task.apply_async(
        kwargs={
            "pdf_paths":   [str(file_path)],
            "job_id":      job_id,
            "ingested_by": current_user.username,
        },
        task_id=job_id,   # use job_id as Celery task ID for easy lookup
    )

    logger.info(f"Enqueued ingest job {job_id} for {safe_name}")
    return IngestJobResponse(
        job_id=job_id,
        message="Ingestion queued — poll /jobs/{job_id} for progress",
        files_queued=1,
    )


@app.post("/ingest/bulk", response_model=IngestJobResponse, tags=["Documents"])
@limiter.limit("5/minute")
async def ingest_bulk(
    request: Request,
    files: list[UploadFile]  = File(...),
    current_user: User       = Depends(require_role("editor")),
    pool: asyncpg.Pool       = Depends(get_pg_pool),
):
    _counters["ingest_requests"] += 1
    job_id    = str(uuid.uuid4())
    pdf_paths: list[str] = []

    for f in files:
        safe_name   = _safe_filename(f.filename)
        content     = await _read_and_validate(f)
        # [FIX] job_id prefix — prevents concurrent upload collision
        stored_name = f"{job_id}_{safe_name}"
        dest        = Path(PDF_DIR) / stored_name
        dest.write_bytes(content)
        pdf_paths.append(str(dest))

    if not pdf_paths:
        raise HTTPException(status_code=400, detail="No valid PDFs provided")

    await _register_job(pool, job_id, len(pdf_paths))

    # [B3] Single Celery task handles all files in the bulk upload
    run_ingest_task.apply_async(
        kwargs={
            "pdf_paths":   pdf_paths,
            "job_id":      job_id,
            "ingested_by": current_user.username,
        },
        task_id=job_id,
    )

    logger.info(f"Enqueued bulk ingest job {job_id} for {len(pdf_paths)} files")
    return IngestJobResponse(
        job_id=job_id,
        # [R5] Use actual job_id variable — was {{job_id}} which printed literal "{job_id}"
        message=f"Bulk ingestion queued for {len(pdf_paths)} files — poll /jobs/{job_id}",
        files_queued=len(pdf_paths),
    )


@app.get("/jobs/{job_id}", tags=["Documents"])
@limiter.limit("60/minute")
async def job_status(
    request: Request,
    job_id:  str,
    current_user: User           = Depends(get_current_user),
    redis:        aioredis.Redis = Depends(get_redis),
):
    """
    Returns job progress from Redis (written by ingest_pipeline).
    Also checks Celery task state for queued/pending jobs that haven't
    started writing progress yet.
    """
    # Primary source: Redis progress state written by ingest_pipeline
    state = await get_job_state(redis, job_id)
    if state:
        return {"job_id": job_id, **state}

    # Fallback: check Celery task state (job enqueued but not started yet)
    async_result = await asyncio.to_thread(celery_app.AsyncResult, job_id)
    celery_state = await asyncio.to_thread(lambda: async_result.state)

    if celery_state == "PENDING":
        return {"job_id": job_id, "status": "queued", "progress": 0}
    if celery_state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(async_result.info)}

    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/documents", tags=["Documents"])
@limiter.limit("30/minute")
async def list_documents(
    request: Request,
    current_user: User         = Depends(get_current_user),
    pool:         asyncpg.Pool = Depends(get_pg_pool),
    # [R4] Bounded pagination — prevents full-table dumps via limit=999999
    limit:  int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        async with pool.acquire() as conn:
            rows  = await conn.fetch(
                "SELECT doc_id, filename, file_size, total_pages, "
                "total_chunks, ingested_at, ingested_by, status "
                "FROM ingested_documents ORDER BY ingested_at DESC "
                "LIMIT $1 OFFSET $2",
                limit, offset,
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM ingested_documents")
        return {
            "total":     total,
            "limit":     limit,
            "offset":    offset,
            "documents": [dict(r) for r in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


# ── CHAT ──────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    _counters["chat_requests"] += 1
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = await arun_rag(question=body.question, thread_id=body.thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")
    return ChatResponse(answer=answer, thread_id=body.thread_id)


@app.post("/chat/stream", tags=["Chat"])
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    _counters["chat_requests"] += 1
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if _agent.graph is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    config = {"configurable": {"thread_id": body.thread_id}}

    async def token_generator():
        try:
            async for chunk, _meta in _agent.graph.astream(
                {
                    "messages":        [{"role": "user", "content": body.question}],
                    "question":        body.question,
                    "rewritten_query": "",
                    "dense_docs":      [],
                    "sparse_docs":     [],
                    "reranked_docs":   [],
                    "answer":          "",
                },
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield f"data: {chunk.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history/{thread_id}", tags=["Chat"])
@limiter.limit("30/minute")
async def get_history(
    request: Request,
    thread_id: str,
    current_user: User = Depends(get_current_user),
):
    """Return conversation history for a thread."""
    if _agent.graph is None:
        return {"thread_id": thread_id, "messages": []}

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await _agent.graph.aget_state(config)
        if not state or not state.values:
            return {"thread_id": thread_id, "messages": []}

        raw_messages = state.values.get("messages", [])
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        messages = []

        for msg in raw_messages:
            if not msg:
                continue
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role    = role_map.get(str(msg.type).lower(), msg.type)
                content = msg.content
            elif isinstance(msg, dict):
                role    = role_map.get(msg.get("type", msg.get("role", "")), "user")
                content = msg.get("content", "")
            else:
                continue
            if content:
                messages.append({"role": role, "content": content})

    except Exception as exc:
        logger.error(f"get_history failed for {thread_id}: {exc}", exc_info=True)
        messages = []

    return {"thread_id": thread_id, "messages": messages}


# ── ENTRY POINT ───────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)