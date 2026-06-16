"""
ingest.py — Production RAG Ingestion Engine v5

"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import uuid
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ── Windows compatibility ─────────────────────────────────
# On Windows, multiprocessing uses 'spawn' which re-imports every module
# in each subprocess. This breaks ProcessPoolExecutor when called from
# Celery (no __main__ guard). We fall back to ThreadPoolExecutor on Windows
# — PDF parsing is I/O + PyPDF CPU work; threads are adequate for dev.
# On Linux (production) ProcessPoolExecutor is used for true parallelism.
IS_WINDOWS = sys.platform == "win32"

# ── Windows asyncio event loop policy ────────────────────
# Python 3.8+ defaults to ProactorEventLoop on Windows, which has known
# issues with asyncpg and some threading operations.
# SelectorEventLoop is stable across all platforms.
# Must be set before any event loop is created.
if IS_WINDOWS:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def _run_in_thread(func, *args):
    """
    Safe replacement for asyncio.to_thread() that works on both Windows
    (Celery solo pool / SelectorEventLoop) and Linux (prefork pool).

    asyncio.to_thread() is loop.run_in_executor(None, ...) under the hood.
    Calling run_in_executor directly avoids any hidden differences between
    ProactorEventLoop and SelectorEventLoop thread pool behaviour.
    """
    loop = asyncio.get_running_loop()
    import functools
    return await loop.run_in_executor(None, functools.partial(func, *args))

import asyncpg
import numpy as np
import redis.asyncio as aioredis
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── CONFIG ────────────────────────────────────────────────
PDF_DIR         = os.getenv("PDF_DIR", "./pdfs")
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
PG_DSN          = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379")

EMBEDDING_DIM        = 1536
CHUNK_SIZE           = 1000
CHUNK_OVERLAP        = 250
BATCH_SIZE           = 64
MAX_WORKERS          = min(8, (os.cpu_count() or 4))
EMBED_CONCURRENCY    = 4
MAX_CHUNKS_PER_DOC   = 2000
MAX_UPLOAD_BYTES     = 50 * 1024 * 1024   # single source of truth — imported by main.py

SEMANTIC_MERGE_THRESHOLD = 0.88


BM25_REDIS_KEY    = "bm25:index"
BM25_BUILT_AT_KEY = "bm25:built_at"

# ── EMBEDDINGS ────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ══════════════════════════════════════════════════════════
# BM25 SINGLETON
# ══════════════════════════════════════════════════════════

class _BM25Index:
    """
    Asyncio-safe singleton BM25 index with Redis persistence.  [B2]

    Workers that start after an ingestion job can call load_from_redis()
    to get the pre-built index without scrolling Qdrant again.
    The asyncio.Lock is created lazily inside a coroutine so it is always
    bound to the correct running event loop.  [F7]
    """

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi
        self._BM25Okapi = BM25Okapi
        self._index = None
        self._docs: list[dict] = []
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def build(self, qdrant_client: QdrantClient, collection_name: str) -> None:
        """Scroll Qdrant, build BM25 index off the event loop.  [F3]"""
        async with self._get_lock():
            logger.info("[BM25] Scrolling full collection …")
            all_points: list = []
            offset = None

            while True:
                batch, offset = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1_000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                all_points.extend(batch)
                if offset is None:
                    break

            logger.info(f"[BM25] {len(all_points)} points. Building index …")

            # Guard: BM25Okapi([]) raises ZeroDivisionError on empty corpus.
            # Normal on fresh deployment before any PDFs are ingested.
            # Leave index as None — is_ready returns False and sparse
            # retrieval skips gracefully until first ingest completes.
            if not all_points:
                logger.info("[BM25] Collection is empty — index will build after first ingest")
                return

            texts = [p.payload.get("page_content", "") for p in all_points]
            metas = [p.payload for p in all_points]
            tokenized = [t.lower().split() for t in texts]

            BM25Okapi = self._BM25Okapi
            index = await _run_in_thread(BM25Okapi, tokenized)

            self._index = index
            self._docs = [{"page_content": t, "metadata": m} for t, m in zip(texts, metas)]
            logger.info(f"[BM25] Ready: {len(self._docs)} documents")

    async def load_from_redis(self, redis: aioredis.Redis) -> bool:
        """
        [B2 + R1] Load a previously serialised index from Redis.

        [R1] Opens its own bytes-mode Redis client so the result is always
        bytes regardless of the caller's decode_responses setting.  The
        previous latin-1 encode workaround was fragile — any byte outside
        the latin-1 range would silently corrupt the pickle data.
        Now mirrors save_bm25_to_redis: both use decode_responses=False.

        Returns True if the load succeeded, False if no cached index exists.
        """
        async with self._get_lock():
            try:
                # decode_responses setting for binary pickle data.
                bytes_redis = aioredis.from_url(REDIS_URL, decode_responses=False)
                try:
                    raw = await bytes_redis.get(BM25_REDIS_KEY)
                finally:
                    await bytes_redis.aclose()

                if raw is None:
                    return False

                # raw is guaranteed bytes from decode_responses=False client
                obj = pickle.loads(zlib.decompress(raw))
                self._index = obj["index"]
                self._docs  = obj["docs"]
                logger.info(f"[BM25] Loaded from Redis: {len(self._docs)} documents")
                return True
            except Exception as exc:
                logger.warning(f"[BM25] Redis load failed (will rebuild): {exc}")
                return False

    def search(self, query: str, top_k: int = 20) -> list[Document]:
        if self._index is None or not self._docs:
            logger.warning("[BM25] Index not built yet — returning empty")
            return []
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_i  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            Document(
                page_content=self._docs[i]["page_content"],
                metadata=self._docs[i]["metadata"],
            )
            for i in top_i if scores[i] > 0
        ]

    @property
    def is_ready(self) -> bool:
        return self._index is not None


bm25_index = _BM25Index()


# ── BM25 Redis persistence helpers  [B2] ─────────────────

async def save_bm25_to_redis(redis: aioredis.Redis) -> None:
    """
    [B2] Serialise the current BM25 index to Redis so other workers can
    load it without rebuilding.  Uses a bytes-mode Redis client to avoid
    encoding issues with binary pickle data.
    """
    if not bm25_index.is_ready:
        logger.warning("[BM25] Cannot save — index not built yet")
        return
    try:
        obj  = {"index": bm25_index._index, "docs": bm25_index._docs}
        data = zlib.compress(pickle.dumps(obj), level=1)  # level=1 — fast compress

        # We need a bytes-mode client for binary data
        bytes_redis = aioredis.from_url(REDIS_URL, decode_responses=False)
        try:
            await bytes_redis.set(BM25_REDIS_KEY, data)
            await bytes_redis.set(
                BM25_BUILT_AT_KEY,
                __import__("datetime").datetime.utcnow().isoformat(),
            )
        finally:
            await bytes_redis.aclose()

        logger.info(
            f"[BM25] Saved to Redis: {len(bm25_index._docs)} docs, "
            f"{len(data) / 1024:.1f} KB compressed"
        )
    except Exception as exc:
        logger.error(f"[BM25] Save to Redis failed: {exc}", exc_info=True)


async def rebuild_bm25_index(redis: aioredis.Redis | None = None) -> None:
    """
    [F13 + B2] Rebuild BM25 from Qdrant and optionally persist to Redis.
    Pass redis=app.state.redis in production; omit for CLI / tests.
    """
    client = QdrantClient(url=QDRANT_URL)
    try:
        await bm25_index.build(client, COLLECTION_NAME)
        if redis is not None:
            await save_bm25_to_redis(redis)
    finally:
        client.close()


# ══════════════════════════════════════════════════════════
# 1. PURE-CPU PDF PARSING WORKER  [F6 + F9]
# ══════════════════════════════════════════════════════════

def _parse_single_pdf(pdf_path: str) -> list[dict]:
    """
    CPU-bound worker (ProcessPoolExecutor).
    No OpenAI / network calls inside.
    Real page numbers preserved.  [F9]
    print() is intentional here — logging is not available inside
    ProcessPoolExecutor workers without extra setup.
    """
    import hashlib
    import uuid
    from pathlib import Path

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    path   = Path(pdf_path)
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, path.name))

    try:
        loader = PyPDFLoader(str(path))
        pages  = loader.load()
    except Exception as exc:
        print(f"[ERROR] Failed to parse {path.name}: {exc}")
        return []

    if not pages:
        print(f"[WARN] No pages extracted from {path.name}")
        return []

    total_pages = len(pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
    )

    chunks: list[dict] = []
    for page in pages:
        page_text = page.page_content.strip()
        if not page_text:
            continue
        page_num = page.metadata.get("page", 0) + 1   # 1-based  [F9]

        for text in splitter.split_text(page_text):
            if not text.strip():
                continue
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            chunks.append(
                {
                    "page_content": text,
                    "metadata": {
                        "doc_id":       doc_id,
                        "source":       path.name,
                        "page":         page_num,
                        "chunk_index":  len(chunks),
                        "chunk_size":   len(text),
                        "content_hash": content_hash,
                        "file_size":    path.stat().st_size,
                        "total_pages":  total_pages,
                        "chunk_type":   "recursive",
                    },
                }
            )
            if len(chunks) >= MAX_CHUNKS_PER_DOC:
                print(
                    f"[WARN] {path.name}: chunk cap ({MAX_CHUNKS_PER_DOC}) reached "
                    f"at page {page_num}/{total_pages}"
                )
                return chunks
    return chunks


# ══════════════════════════════════════════════════════════
# 2. PARALLEL PARSE  [F12]
# ══════════════════════════════════════════════════════════

def _is_in_daemon_process() -> bool:
    """
    Detect if we are running inside a Celery worker process.

    Celery prefork workers are Python daemon processes.
    Daemon processes cannot spawn child processes — attempting to use
    ProcessPoolExecutor inside one raises:
        'daemonic processes are not allowed to have children'

    We check multiprocessing.current_process().daemon which is True
    inside every Celery prefork worker process.
    """
    import multiprocessing
    return multiprocessing.current_process().daemon


async def parse_pdfs_parallel(pdf_paths: list[str]) -> list[dict]:
    """
    Parse PDFs in parallel.

    Uses ThreadPoolExecutor when:
      - Running on Windows (no fork — spawn causes reimport issues)
      - Running inside a Celery worker (daemon process cannot spawn children)

    Uses ProcessPoolExecutor when:
      - Running on Linux outside a daemon process (CLI, tests, direct call)
        for true CPU parallelism across cores.

    PyPDF parsing is mostly I/O-bound (file reads + text decoding) so
    threads are nearly as fast as processes for typical PDF sizes.
    """
    loop = asyncio.get_running_loop()

    # Force threads if: Windows OR inside a Celery daemon worker process
    use_threads = IS_WINDOWS or _is_in_daemon_process()

    if use_threads:
        executor_cls = ThreadPoolExecutor
        worker_count = min(4, max(1, len(pdf_paths)))
        logger.info(
            f"  PDF parsing: ThreadPoolExecutor × {worker_count} "
            f"({'Windows' if IS_WINDOWS else 'Celery daemon process'})"
        )
    else:
        executor_cls = ProcessPoolExecutor
        worker_count = MAX_WORKERS
        logger.info(f"  PDF parsing: ProcessPoolExecutor × {worker_count}")

    with executor_cls(max_workers=worker_count) as executor:
        tasks   = [loop.run_in_executor(executor, _parse_single_pdf, p) for p in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    flat: list[dict] = []
    for path, result in zip(pdf_paths, results):
        if isinstance(result, Exception):
            logger.error(f"Worker failed on {path}: {result}")
        else:
            flat.extend(result)
            logger.info(f"  Parsed {Path(path).name} → {len(result)} chunks")
    return flat


# ══════════════════════════════════════════════════════════
# 3. ASYNC BATCH EMBEDDINGS
# ══════════════════════════════════════════════════════════

async def _embed_batch(texts: list[str]) -> list[list[float]]:
    for attempt in range(5):
        try:
            return await _run_in_thread(embeddings.embed_documents, texts)
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(f"Embedding attempt {attempt + 1} failed: {exc}. Retry in {wait}s")
            await asyncio.sleep(wait)
    raise RuntimeError("Embedding failed after 5 attempts")


async def embed_all_chunks(chunks: list[dict]) -> list[tuple[dict, list[float]]]:
    batches   = [chunks[i: i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)
    results: dict[int, list[tuple[dict, list[float]]]] = {}

    async def _process(idx: int, batch: list[dict]) -> None:
        async with semaphore:
            texts   = [c["page_content"] for c in batch]
            vectors = await _embed_batch(texts)
            results[idx] = list(zip(batch, vectors))

    await asyncio.gather(*[_process(i, b) for i, b in enumerate(batches)])

    flat: list[tuple[dict, list[float]]] = []
    for i in range(len(batches)):
        if i in results:
            flat.extend(results[i])
        else:
            logger.warning(f"Batch {i} missing from embed results — skipped")

    logger.info(f"  Embedded {len(flat)} chunks total")
    return flat


# ══════════════════════════════════════════════════════════
# 3b. SEMANTIC MERGE PASS  [C1 + C2 — hash snapshot + full metadata copy]
#
# FIX C1: `original_vecs` is now built from a snapshot taken BEFORE the
#   merge loop.  The loop mutates cur_chunk in-place (changing its
#   content_hash), which previously made the original hash disappear from
#   the lookup → chunks were silently dropped.
#
# FIX C2: Instead of `dict(chunk)` (shallow copy — metadata still shared),
#   every new chunk dict is fully reconstructed so the metadata dict is
#   always a new object.
# ══════════════════════════════════════════════════════════

def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom  = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _clone_chunk(chunk: dict, override: dict | None = None) -> dict:
    """
    [C2] Return a fully independent copy of a chunk dict.
    metadata is always a new dict — never a reference to the original.
    override keys are applied to the metadata copy.
    """
    meta = {**chunk["metadata"]}   # new dict, not a reference
    if override:
        meta.update(override)
    return {"page_content": chunk["page_content"], "metadata": meta}


async def _semantic_merge(
    embedded: list[tuple[dict, list[float]]],
    threshold: float = SEMANTIC_MERGE_THRESHOLD,
) -> list[tuple[dict, list[float]]]:
    """
    Merge consecutive semantically-identical chunks and RE-EMBED the
    merged text.  [F1]
    Chunks from different source files are never merged.

    [C1] Snapshot original hash→vec BEFORE the loop so mutations during
         merging never cause a lookup miss.
    [C2] All new chunk dicts are created via _clone_chunk() — no shared
         metadata references.
    """
    if not embedded:
        return embedded

    # ── C1: snapshot BEFORE any mutation ─────────────────
    original_vecs: dict[str, list[float]] = {
        chunk["metadata"]["content_hash"]: vec
        for chunk, vec in embedded
    }

    pending_merges: list[tuple[dict, str]] = []
    cur_chunk = _clone_chunk(embedded[0][0])   # C2: independent copy
    cur_vec   = embedded[0][1]
    cur_text  = cur_chunk["page_content"]

    for raw_next_chunk, next_vec in embedded[1:]:
        next_chunk = _clone_chunk(raw_next_chunk)  # C2: independent copy
        same_doc   = next_chunk["metadata"]["source"] == cur_chunk["metadata"]["source"]

        if same_doc and _cosine(cur_vec, next_vec) >= threshold:
            combined = cur_text + "\n\n" + next_chunk["page_content"]
            if len(combined) <= CHUNK_SIZE * 2:
                combined_hash = hashlib.sha256(combined.encode()).hexdigest()
                cur_chunk = _clone_chunk(
                    cur_chunk,
                    override={
                        "content_hash": combined_hash,
                        "chunk_size":   len(combined),
                        "chunk_type":   "semantic_merged",
                    },
                )
                cur_chunk["page_content"] = combined
                cur_text = combined
                continue

        pending_merges.append((cur_chunk, cur_text))
        cur_chunk = next_chunk
        cur_vec   = next_vec
        cur_text  = cur_chunk["page_content"]

    pending_merges.append((cur_chunk, cur_text))

    # ── Re-embed merged chunks ────────────────────────────
    texts_to_embed = [
        text for chunk, text in pending_merges
        if chunk["metadata"].get("chunk_type") == "semantic_merged"
    ]
    re_embedded: dict[str, list[float]] = {}
    if texts_to_embed:
        logger.info(f"  Re-embedding {len(texts_to_embed)} merged chunks …")
        vecs = await _embed_batch(texts_to_embed)
        for text, vec in zip(texts_to_embed, vecs):
            h = hashlib.sha256(text.encode()).hexdigest()
            re_embedded[h] = vec

    # ── Reassemble ────────────────────────────────────────

    merged: list[tuple[dict, list[float]]] = []
    for chunk, _text in pending_merges:
        h   = chunk["metadata"]["content_hash"]
        vec = re_embedded.get(h) or original_vecs.get(h)
        if vec is None:
            logger.warning(f"No vector found for chunk {h[:8]} — skipping")
            continue
        merged.append((chunk, vec))

    logger.info(
        f"  Semantic merge: {len(embedded)} → {len(merged)} chunks "
        f"(threshold={threshold})"
    )
    return merged


# ══════════════════════════════════════════════════════════
# 4. QDRANT UPSERT
# ══════════════════════════════════════════════════════════

async def ensure_collection_async(client: AsyncQdrantClient) -> None:
    collections = [c.name for c in (await client.get_collections()).collections]
    if COLLECTION_NAME not in collections:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        sync_client = QdrantClient(url=QDRANT_URL)
        try:
            for field_name in ("source", "doc_id", "page", "content_hash"):
                sync_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        finally:
            sync_client.close()
        logger.info(f"Created Qdrant collection + indexes: {COLLECTION_NAME}")
    else:
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists")


async def upsert_to_qdrant(
    client: AsyncQdrantClient,
    embedded: list[tuple[dict, list[float]]],
) -> int:
    points = []
    for chunk, vector in embedded:
        meta     = chunk["metadata"]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, meta["content_hash"]))
        payload  = {**meta, "page_content": chunk["page_content"]}
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    for i in range(0, len(points), 256):
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i: i + 256],
            wait=True,
        )
        logger.info(f"  Upserted batch {i // 256 + 1}/{(len(points) - 1) // 256 + 1}")

    return len(points)


# ══════════════════════════════════════════════════════════
# 5. POSTGRESQL SCHEMA + HELPERS
# ══════════════════════════════════════════════════════════

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingested_documents (
    doc_id        TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_size     BIGINT,
    total_pages   INT,
    total_chunks  INT,
    ingested_at   TIMESTAMPTZ DEFAULT now(),
    ingested_by   TEXT,
    status        TEXT DEFAULT 'completed'
);
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id        TEXT PRIMARY KEY,
    started_at    TIMESTAMPTZ DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    total_files   INT,
    total_chunks  INT,
    status        TEXT DEFAULT 'pending',
    error_message TEXT
);
"""


async def ensure_pg_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)


async def record_document(
    pool: asyncpg.Pool,
    doc_id: str,
    filename: str,
    file_size: int,
    total_pages: int,
    total_chunks: int,
    ingested_by: str = "system",
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ingested_documents
                (doc_id, filename, file_size, total_pages, total_chunks, ingested_by)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (doc_id) DO UPDATE
                SET total_chunks = EXCLUDED.total_chunks,
                    ingested_at  = now()
            """,
            doc_id, filename, file_size, total_pages, total_chunks, ingested_by,
        )


async def update_job_pg(
    pool: asyncpg.Pool,
    job_id: str,
    total_files: int,
    total_chunks: int,
    status: str = "completed",
    error: str | None = None,
) -> None:
    """UPDATE only — INSERT is owned by main.py / ARQ task.  [F2]"""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ingestion_jobs
               SET finished_at   = now(),
                   total_files   = $2,
                   total_chunks  = $3,
                   status        = $4,
                   error_message = $5
             WHERE job_id = $1
            """,
            job_id, total_files, total_chunks, status, error,
        )


# ══════════════════════════════════════════════════════════
# 6. REDIS JOB STATE
# ══════════════════════════════════════════════════════════

async def update_job_state(
    redis: aioredis.Redis, job_id: str, state: dict
) -> None:

    await redis.setex(f"job:{job_id}", 3_600, json.dumps(state))


async def get_job_state(
    redis: aioredis.Redis, job_id: str
) -> dict | None:

    raw = await redis.get(f"job:{job_id}")
    return json.loads(raw) if raw else None


# ══════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ══════════════════════════════════════════════════════════

async def ingest_pipeline(
    pdf_paths: list[str],
    pg_pool: asyncpg.Pool,
    redis: aioredis.Redis,
    ingested_by: str = "system",
    job_id: str | None = None,
) -> dict[str, Any]:
    """
    Full ingestion pipeline.

    pg_pool and redis MUST be passed in by the caller (either main.py's
    lifespan pool / ARQ worker context, or the CLI wrapper below).
    This function never creates its own connections.  [B4]
    """
    job_id = job_id or str(uuid.uuid4())
    t0     = time.perf_counter()

    qdrant = AsyncQdrantClient(url=QDRANT_URL)

    try:
        await ensure_pg_schema(pg_pool)
        await ensure_collection_async(qdrant)

        await update_job_state(redis, job_id, {"status": "parsing", "progress": 0})

        # ── Step 1: Parse ──────────────────────────────────
        logger.info(f"[{job_id}] Parsing {len(pdf_paths)} PDFs …")
        all_chunks = await parse_pdfs_parallel(pdf_paths)

        if not all_chunks:
            raise RuntimeError("No chunks produced — all PDFs may be empty or unreadable")

        # ── Step 2: Dedup ──────────────────────────────────
        seen: set[str] = set()
        unique: list[dict] = []
        for c in all_chunks:
            h = c["metadata"]["content_hash"]
            if h not in seen:
                seen.add(h)
                unique.append(c)
        logger.info(f"[{job_id}] After dedup: {len(unique)} unique chunks")

        await update_job_state(redis, job_id, {"status": "embedding", "progress": 10})

        # ── Step 3: Embed ──────────────────────────────────
        embedded = await embed_all_chunks(unique)

        # ── Step 4: Semantic merge   ─────────────
        embedded = await _semantic_merge(embedded)

        await update_job_state(redis, job_id, {"status": "upserting", "progress": 80})

        # ── Step 5: Upsert to Qdrant ──────────────────────
        upserted = await upsert_to_qdrant(qdrant, embedded)

        # ── Step 6: Record document metadata ──────────────
        docs_meta: dict[str, dict] = {}
        for chunk, _ in embedded:
            m = chunk["metadata"]
            if m["doc_id"] not in docs_meta:
                docs_meta[m["doc_id"]] = {
                    "filename":    m["source"],
                    "file_size":   m["file_size"],
                    "total_pages": m["total_pages"],
                    "chunks":      0,
                }
            docs_meta[m["doc_id"]]["chunks"] += 1

        for doc_id, info in docs_meta.items():
            await record_document(
                pg_pool, doc_id, info["filename"], info["file_size"],
                info["total_pages"], info["chunks"], ingested_by,
            )

        # ── Step 7: Rebuild + persist BM25  ──────────
        # Rebuild before capturing elapsed so the reported time reflects
        # the full pipeline including BM25.
        logger.info(f"[{job_id}] Rebuilding BM25 index …")
        await rebuild_bm25_index(redis=redis)

        elapsed = time.perf_counter() - t0

        await update_job_pg(pg_pool, job_id, len(pdf_paths), upserted)
        await update_job_state(
            redis, job_id,
            {
                "status":          "completed",
                "progress":        100,
                "total_chunks":    upserted,
                "elapsed_seconds": round(elapsed, 2),
            },
        )

        summary = {
            "job_id":            job_id,
            "files_processed":   len(pdf_paths),
            "total_chunks":      upserted,
            "unique_docs":       len(docs_meta),
            "elapsed_seconds":   round(elapsed, 2),
            "chunks_per_second": round(upserted / elapsed, 1) if elapsed > 0 else 0,
        }
        logger.info(f"[{job_id}] ✅ Done: {summary}")
        return summary

    except Exception as exc:
        logger.error(f"[{job_id}] ❌ Pipeline failed: {exc}", exc_info=True)
        try:
            await update_job_pg(pg_pool, job_id, len(pdf_paths), 0, "failed", str(exc))
            await update_job_state(redis, job_id, {"status": "failed", "error": str(exc)})
        except Exception as inner:
            logger.error(f"[{job_id}] Failed to record failure state: {inner}")
        raise

    finally:
        await qdrant.close()


# ══════════════════════════════════════════════════════════
# 8. SYNC WRAPPERS (backward compact / CLI)
# ══════════════════════════════════════════════════════════

def _ensure_collection_sync(client: QdrantClient) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        try:
            client.create_collection(...)
        except Exception as exc:
            collections_now = [c.name for c in client.get_collections().collections]
            if COLLECTION_NAME not in collections_now:
                raise  # real error — re-raise
            logger.debug("Collection already created by another worker")


def get_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)
    _ensure_collection_sync(client)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="page_content",
        metadata_payload_key= None,
    )


def ingest_pdfs(pdf_dir: str) -> list[Document]:
    paths = [
        str(Path(pdf_dir) / f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    ]
    raw = asyncio.run(parse_pdfs_parallel(paths))
    return [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in raw
    ]


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else PDF_DIR
    os.makedirs(target, exist_ok=True)
    paths = [str(Path(target) / f) for f in os.listdir(target) if f.endswith(".pdf")]
    if not paths:
        print("No PDFs found.")
        sys.exit(0)

    async def _cli() -> None:
        # CLI creates its own pool — ingest_pipeline never creates one  [B4]
        pool  = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
        redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            summary = await ingest_pipeline(paths, pg_pool=pool, redis=redis)
            print("\nIngestion Summary:")
            for k, v in summary.items():
                print(f"  {k}: {v}")
        finally:
            await pool.close()
            await redis.aclose()

    asyncio.run(_cli())