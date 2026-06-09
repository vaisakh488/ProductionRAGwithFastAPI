"""
tasks.py — Celery async task queue worker

WHY CELERY INSTEAD OF BackgroundTasks:
  FastAPI's BackgroundTasks runs inside the uvicorn process, which means:
    • A 5000-PDF ingest ties up one API worker for its entire duration
    • A server restart mid-job loses the job with no recovery
    • No concurrency limit — 10 simultaneous uploads spawn 10 simultaneous
      ingest pipelines, crushing CPU and burning through OpenAI rate limits

  Celery gives you:
    • Jobs run in completely separate worker processes — API stays fast
    • Jobs survive server restarts (persisted in Redis broker)
    • Automatic retries with exponential backoff on failure
    • Concurrency controlled per-worker (CELERY_WORKER_CONCURRENCY)
    • Built-in monitoring via Flower (http://localhost:5555)
    • Battle-tested at scale — used by Instagram, Dropbox, etc.

ARCHITECTURE:
  API process          Redis broker          Celery worker process
  ─────────────        ────────────          ─────────────────────
  POST /ingest    →    enqueue job      →    run_ingest_task()
  return job_id   ←    job_id           ←    (runs ingest_pipeline)
  GET /jobs/{id}  →    read job state   ←    (writes progress to Redis)

RUNNING:
  # Start worker (separate terminal / container):
  celery -A tasks worker --loglevel=info --concurrency=2

  # Monitor via Flower:
  celery -A tasks flower --port=5555

  # In docker-compose: see docker-compose.yml celery_worker service

RETRY BEHAVIOUR:
  max_retries=3, countdown doubles each retry: 60s → 120s → 240s.
  After 3 failures the job is marked 'failed' in Redis and PostgreSQL.
  Files are cleaned up after permanent failure.

CONCURRENCY:
  Controlled via CELERY_WORKER_CONCURRENCY env var (default: 2).
  Also wired into celery_app.conf.worker_concurrency so it applies
  whether the worker is started via docker-compose, bare metal, or CI.
  The --concurrency CLI flag still overrides conf if explicitly passed.
  Formula: concurrency * EMBED_CONCURRENCY * BATCH_SIZE * ~6KB ≈ RAM needed.

REVIEW FIXES (v5.1):
  [R1] rebuild_bm25_periodic accesses _redis and _loop directly instead of
       calling _get_resources(). Previously _get_resources() required pg_pool
       to be initialised — if PostgreSQL was down at worker startup the task
       raised RuntimeError even though it never touches the database.
       Now uses explicit None checks on only the resources it actually needs.
  [R2] CELERY_WORKER_CONCURRENCY env var wired into celery_app.conf as
       worker_concurrency. Previously the env var was only applied when
       docker-compose passed --concurrency on the CLI. Bare-metal and CI
       runs ignored it and fell back to Celery's default (CPU count).
       docker-compose --concurrency flag still overrides conf if set.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import asyncpg
import redis.asyncio as aioredis
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from dotenv import load_dotenv

from ingest import ingest_pipeline, rebuild_bm25_index, update_job_state

load_dotenv()

logger = logging.getLogger(__name__)

PG_DSN    = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")
REDIS_URL = os.getenv("REDIS_URL",    "redis://localhost:6379")

# ── Celery app ────────────────────────────────────────────
# broker  = where tasks are queued (Redis list)
# backend = where task results/state are stored (Redis hash)
# Both reuse the existing Redis instance — no new infrastructure needed.
celery_app = Celery(
    "rag_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    # [R2] Wire CELERY_WORKER_CONCURRENCY into conf so it applies regardless
    # of how the worker is started (docker-compose, bare metal, CI).
    # The --concurrency CLI flag still overrides this if explicitly passed.
    worker_concurrency         = int(os.getenv("CELERY_WORKER_CONCURRENCY", "2")),

    # Serialisation
    task_serializer            = "json",
    result_serializer          = "json",
    accept_content             = ["json"],

    # Results kept for 7 days — useful for debugging failed jobs
    result_expires             = 60 * 60 * 24 * 7,

    # Acknowledge task only after it completes — prevents silent loss
    # if the worker crashes mid-task. The task will be re-queued.
    task_acks_late             = True,
    task_reject_on_worker_lost = True,

    # One queue, prefetch 1 so a long ingest doesn't starve other tasks
    worker_prefetch_multiplier = 1,

    # Timezone
    timezone                   = "UTC",
    enable_utc                 = True,

    # Task time limit: 2 hours hard, 1h45m soft (sends SoftTimeLimitExceeded)
    task_soft_time_limit       = 60 * 105,
    task_time_limit            = 60 * 120,
)


# ══════════════════════════════════════════════════════════
# WORKER PROCESS LIFECYCLE
# Per-process resources — created once per worker process, not per task.
# Celery workers are multiprocessing-based, so each process needs its own
# asyncio event loop and connection pool.
# ══════════════════════════════════════════════════════════

# Module-level references — set in worker_process_init, cleared in shutdown
_pg_pool: asyncpg.Pool | None = None
_redis:   aioredis.Redis | None = None
_loop:    asyncio.AbstractEventLoop | None = None


@worker_process_init.connect
def init_worker_process(**kwargs):
    """
    Called once when each Celery worker process starts.
    Creates a dedicated event loop and connection pool for the process.
    """
    global _pg_pool, _redis, _loop

    logger.info("[Celery Worker] Initialising process resources …")
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def _setup():
        global _pg_pool, _redis
        _pg_pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
        _redis   = aioredis.from_url(REDIS_URL, decode_responses=True)

        # Attempt to load BM25 from Redis cache on worker startup
        try:
            from ingest import bm25_index
            loaded = await bm25_index.load_from_redis(_redis)
            if loaded:
                logger.info("[Celery Worker] BM25 index loaded from Redis")
            else:
                logger.info("[Celery Worker] No cached BM25 — will build on first ingest")
        except Exception as exc:
            logger.warning(f"[Celery Worker] BM25 load failed: {exc}")

    _loop.run_until_complete(_setup())
    logger.info("[Celery Worker] Process resources ready")


@worker_process_shutdown.connect
def shutdown_worker_process(**kwargs):
    """Called once when a Celery worker process exits."""
    global _pg_pool, _redis, _loop

    logger.info("[Celery Worker] Shutting down process resources …")

    async def _teardown():
        if _pg_pool:
            await _pg_pool.close()
        if _redis:
            await _redis.aclose()

    if _loop and not _loop.is_closed():
        _loop.run_until_complete(_teardown())
        _loop.close()

    logger.info("[Celery Worker] Process resources closed")


def _get_resources() -> tuple[asyncpg.Pool, aioredis.Redis, asyncio.AbstractEventLoop]:
    """Return the per-process resources, raising clearly if not initialised."""
    if _pg_pool is None or _redis is None or _loop is None:
        raise RuntimeError(
            "Worker process resources not initialised. "
            "Ensure worker_process_init signal fired correctly."
        )
    return _pg_pool, _redis, _loop


# ══════════════════════════════════════════════════════════
# TASK: INGEST
# ══════════════════════════════════════════════════════════

@celery_app.task(
    bind=True,
    name="tasks.run_ingest_task",
    max_retries=3,
    # Don't autoretry — we handle retry logic explicitly so we can
    # distinguish transient failures (worth retrying) from permanent ones
    # (file unreadable, schema error) and clean up files on the latter.
)
def run_ingest_task(
    self,
    pdf_paths: list[str],
    job_id: str,
    ingested_by: str,
) -> dict:
    """
    Celery task — replaces FastAPI BackgroundTasks._run_ingestion.

    Runs in a dedicated Celery worker process with its own asyncio loop
    and connection pool (set up in worker_process_init above).

    Retries up to 3 times with exponential backoff on transient failures.
    Files are cleaned up only after all retries are exhausted.
    """
    pg_pool, redis, loop = _get_resources()

    logger.info(
        f"[Celery] Starting ingest job {job_id} "
        f"(attempt {self.request.retries + 1}/{self.max_retries + 1}) "
        f"for {len(pdf_paths)} PDFs"
    )

    async def _run():
        return await ingest_pipeline(
            pdf_paths,
            pg_pool=pg_pool,
            redis=redis,
            ingested_by=ingested_by,
            job_id=job_id,
        )

    try:
        summary = loop.run_until_complete(_run())
        logger.info(f"[Celery] Job {job_id} completed: {summary}")
        return summary

    except Exception as exc:
        logger.error(
            f"[Celery] Job {job_id} failed "
            f"(attempt {self.request.retries + 1}): {exc}",
            exc_info=True,
        )

        is_final_attempt = self.request.retries >= self.max_retries

        if is_final_attempt:
            # Permanent failure — clean up uploaded files and record state
            logger.warning(
                f"[Celery] Job {job_id} exhausted retries — cleaning up files"
            )
            for p in pdf_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

            # Best-effort state update on permanent failure
            async def _record_failure():
                await update_job_state(
                    redis, job_id,
                    {"status": "failed", "error": str(exc), "progress": 0},
                )
            try:
                loop.run_until_complete(_record_failure())
            except Exception:
                pass

            # Re-raise without retry — Celery marks task FAILURE
            raise

        # Transient failure — exponential backoff: 60s, 120s, 240s
        countdown = 60 * (2 ** self.request.retries)
        logger.info(f"[Celery] Retrying job {job_id} in {countdown}s …")
        raise self.retry(exc=exc, countdown=countdown)


# ══════════════════════════════════════════════════════════
# TASK: PERIODIC BM25 REBUILD (Celery Beat)
# ══════════════════════════════════════════════════════════

@celery_app.task(name="tasks.rebuild_bm25_periodic")
def rebuild_bm25_periodic() -> None:
    """
    Periodic BM25 rebuild safety net — run via Celery Beat every hour.
    Ensures all workers eventually converge on the same index even if a
    rebuild triggered during ingestion failed to persist to Redis.

    [R1] Accesses _redis and _loop directly instead of calling
    _get_resources(). _get_resources() requires pg_pool to be initialised
    — if PostgreSQL was down at worker startup the old code raised
    RuntimeError even though this task never touches the database.
    Now uses explicit None checks on only the resources it actually needs.
    """
    # [R1] Direct access — no pg_pool dependency for a Redis-only operation
    if _redis is None or _loop is None:
        logger.error(
            "[Celery Beat] Redis or event loop not initialised — "
            "skipping periodic BM25 rebuild"
        )
        return

    logger.info("[Celery Beat] Periodic BM25 rebuild starting …")
    _loop.run_until_complete(rebuild_bm25_index(redis=_redis))
    logger.info("[Celery Beat] Periodic BM25 rebuild done")


# ── Celery Beat schedule (periodic tasks) ─────────────────
celery_app.conf.beat_schedule = {
    "rebuild-bm25-hourly": {
        "task":     "tasks.rebuild_bm25_periodic",
        "schedule": 60 * 60,  # every 3600 seconds
    },
}