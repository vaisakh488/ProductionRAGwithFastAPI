"""
agent.py — Production LangGraph RAG Agent v5

"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import sys
import time
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable
from typing_extensions import TypedDict

from ingest import bm25_index, get_vectorstore

load_dotenv()

# ── LOGGING ───────────────────────────────────────────────
# [R3] Module-level logger — consistent with all other modules.
# Replaces all print() calls throughout this file.
logger = logging.getLogger(__name__)

# ── Windows event loop policy ─────────────────────────────
# Must match the policy set in ingest.py — both files share the same
# process (uvicorn) so only needs to be set once, but setting it in both
# is safe and makes each file self-contained.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def _run_in_thread(func, *args):
    """Safe asyncio.to_thread replacement — works on Windows SelectorEventLoop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args))


# ── CONFIG ────────────────────────────────────────────────
RERANKER_BACKEND  = os.getenv("RERANKER_BACKEND", "baai").lower()
TOP_K             = 30
RERANK_TOP_K      = 8
MAX_CONTEXT_CHARS = 80_000

# ── LLM ───────────────────────────────────────────────────
# [R4] timeout=60 — fail fast instead of hanging indefinitely on slow
#      OpenAI responses.  max_retries=2 — automatic retry on transient
#      errors (rate limits, 5xx) before raising to the caller.
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60,
    max_retries=2,
)

# ── VECTOR STORE (lazy — initialised in init_agent) ───────
# [C3] vectorstore is no longer a module-level global that blocks the
# event loop at import time.  It is initialised once during lifespan.
_vectorstore = None

def _init_vectorstore(retries: int = 5, delay: float = 3.0):
    for attempt in range(1, retries + 1):
        try:
            return get_vectorstore()
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Qdrant unavailable after {retries} attempts: {exc}"
                ) from exc
            # [R3] was print() — use logger so this appears in structured logs
            logger.warning(
                f"Qdrant not ready (attempt {attempt}/{retries}), "
                f"retrying in {delay}s …"
            )
            time.sleep(delay)


def get_vectorstore_instance():
    """Return the shared vectorstore, raising clearly if init_agent() wasn't called."""
    if _vectorstore is None:
        raise RuntimeError(
            "Agent not initialised — call await init_agent(pg_pool) from lifespan first"
        )
    return _vectorstore


# ── RERANKER ──────────────────────────────────────────────

def _load_reranker():
    """
    Load the reranker model.

    In the API container (slim image — no sentence_transformers/torch):
      RERANKER_BACKEND=baai will automatically fall back to flashrank
      since sentence_transformers is not installed.

    In the Celery worker (heavy image — has sentence_transformers):
      RERANKER_BACKEND=baai loads the full BAAI cross-encoder.

    flashrank is installed in both images (it's lightweight, no torch).
    """
    if RERANKER_BACKEND == "baai":
        try:
            from sentence_transformers import CrossEncoder
            # [R3] was print() — use logger
            logger.info("Loading reranker: BAAI/bge-reranker-base …")
            model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
            logger.info("BAAI reranker ready")
            return "baai", model
        except ImportError:
            # sentence_transformers not available (slim API image)
            # fall through to flashrank below
            logger.warning(
                "sentence_transformers not available — falling back to FlashRank"
            )

    if RERANKER_BACKEND in ("baai", "flashrank"):
        try:
            from flashrank import Ranker
            # [R3] was print()
            logger.info("Loading reranker: FlashRank ms-marco-MiniLM-L-12-v2 …")
            model = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2",
                cache_dir="/tmp/flashrank",
            )
            logger.info("FlashRank reranker ready")
            return "flashrank", model
        except Exception as exc:
            raise RuntimeError(
                f"FlashRank failed to load: {exc}"
            ) from exc

    raise RuntimeError(
        f"Unknown RERANKER_BACKEND='{RERANKER_BACKEND}'. "
        "Valid options: baai | flashrank"
    )

# Lazy — loaded on first rerank call, not at import time.
# This allows the slim API image (no sentence_transformers) to import
# agent.py successfully.
_reranker_name: str | None = None
_reranker_model = None


def _get_reranker():
    """Load reranker on first call, cache for subsequent calls."""
    global _reranker_name, _reranker_model
    if _reranker_name is None:
        _reranker_name, _reranker_model = _load_reranker()
    return _reranker_name, _reranker_model


# ── STATE ─────────────────────────────────────────────────
class State(TypedDict):
    messages:        Annotated[list, add_messages]
    question:        str
    rewritten_query: str
    dense_docs:      list[Document]
    sparse_docs:     list[Document]
    reranked_docs:   list[Document]
    answer:          str


# ── HELPERS ───────────────────────────────────────────────
def _doc_hash(doc: Document) -> str:
    return doc.metadata.get("content_hash") or str(hash(doc.page_content[:100]))


# ── TRACED ASYNC FUNCTIONS ────────────────────────────────

@traceable(name="1. Query Rewriter", run_type="llm")
async def _rewrite_query(question: str) -> str:
    system = SystemMessage(content=(
        "You are a query optimization expert. "
        "Rewrite the user's question to maximise document retrieval quality. "
        "Make it specific, keyword-rich, and remove conversational filler. "
        "Return ONLY the rewritten query, nothing else."
    ))
    response = await llm.ainvoke([system, {"role": "user", "content": question}])
    return response.content.strip()


@traceable(name="2. Dense Retriever", run_type="retriever")
async def _dense_retrieve(query: str) -> list[Document]:
    """[F9] Plain similarity search — MMR removed (not in qdrant-client 1.13.x)."""
    retriever = get_vectorstore_instance().as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return await retriever.ainvoke(query)


@traceable(name="3. Sparse Retriever (BM25)", run_type="retriever")
def _sparse_retrieve(query: str) -> list[Document]:
    if not bm25_index.is_ready:
        # [R3] was print() — use logger.warning so this surfaces in log aggregators
        logger.warning("BM25 index not ready — sparse retrieval skipped")
        return []
    return bm25_index.search(query, top_k=TOP_K)


@traceable(name="4. RRF Merge + Reranker", run_type="chain")
async def _rerank(
    question: str,
    dense_docs: list[Document],
    sparse_docs: list[Document],
) -> list[Document]:
    rrf_scores:  dict[str, float]    = {}
    doc_by_hash: dict[str, Document] = {}

    def rrf_update(docs: list[Document], k: int = 60) -> None:
        for rank, doc in enumerate(docs):
            h = _doc_hash(doc)
            doc_by_hash[h] = doc
            rrf_scores[h]  = rrf_scores.get(h, 0.0) + 1.0 / (rank + k)

    rrf_update(dense_docs)
    rrf_update(sparse_docs)

    sorted_hashes = sorted(rrf_scores, key=lambda h: rrf_scores[h], reverse=True)
    candidates    = [doc_by_hash[h] for h in sorted_hashes][:30]

    if not candidates:
        return []

    reranker_name, _ = _get_reranker()
    if reranker_name == "baai":
        return await _run_in_thread(_rerank_baai, question, candidates)
    return await _run_in_thread(_rerank_flashrank, question, candidates)


@traceable(name="5. Generator", run_type="llm")
async def _generate(
    question: str,
    reranked_docs: list[Document],
    chat_history: list | None = None,
) -> str:
    """
    Generate an answer using retrieved docs and conversation history.

    chat_history contains previous turns from state["messages"] so the
    model can reference what was said earlier in the conversation.
    The last message (current question) is excluded — it's passed as
    the user turn after the system prompt.
    """
    if not reranked_docs:
        return (
            "I couldn't find relevant information in the document store. "
            "Please ensure the relevant PDFs have been ingested."
        )

    context_parts: list[Document] = []
    total_chars = 0
    for doc in reranked_docs:
        chunk_len = len(doc.page_content)
        if total_chars + chunk_len > MAX_CONTEXT_CHARS:
            break
        context_parts.append(doc)
        total_chars += chunk_len

    if len(context_parts) < len(reranked_docs):
        # [R3] was print() — use logger.warning
        logger.warning(
            f"Context truncated: {len(context_parts)}/{len(reranked_docs)} docs "
            f"({total_chars} chars)"
        )

    context = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')} | "
        f"Page: {doc.metadata.get('page', '?')}\n{doc.page_content}"
        for doc in context_parts
    )

    system = SystemMessage(content=(
        "You are a precise document assistant. "
        "Answer using ONLY the provided context. "
        "If the context contains a Schedule or table with specific amounts "
        "or figures, extract and present them clearly. "
        "Cite sources inline like [DPDP.pdf, p.X]. "
        "If the context does not contain the answer, say so clearly. "
        "You have access to the conversation history — use it to provide "
        "consistent, contextually aware answers.\n\n"
        f"Context:\n{context}"
    ))

    # Build message list: system + conversation history + current question
    messages_to_send = [system]

    # Include prior conversation turns (exclude the current question — added below)
    if chat_history:
        for msg in chat_history[:-1]:   # skip last message (current question)
            if hasattr(msg, "type") and hasattr(msg, "content") and msg.content:
                role_map = {"human": "user", "ai": "assistant", "system": "system"}
                role     = role_map.get(msg.type, msg.type)
                messages_to_send.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict) and msg.get("content"):
                role = msg.get("role") or msg.get("type", "user")
                role_map = {"human": "user", "ai": "assistant"}
                role = role_map.get(role, role)
                messages_to_send.append({"role": role, "content": msg["content"]})

    # Current question
    messages_to_send.append({"role": "user", "content": question})

    response = await llm.ainvoke(messages_to_send)
    return response.content


# ── Reranker helpers (sync, run via asyncio.to_thread) ────

def _rerank_baai(question: str, candidates: list[Document]) -> list[Document]:
    _, reranker_model = _get_reranker()
    pairs  = [[question, doc.page_content[:512]] for doc in candidates]
    scores = reranker_model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:RERANK_TOP_K]]


def _rerank_flashrank(question: str, candidates: list[Document]) -> list[Document]:
    from flashrank import RerankRequest
    _, reranker_model = _get_reranker()
    passages = [{"id": i, "text": doc.page_content[:512]} for i, doc in enumerate(candidates)]
    results  = reranker_model.rerank(RerankRequest(query=question, passages=passages))
    return [candidates[r["id"]] for r in results[:RERANK_TOP_K]]


# ── NODES ─────────────────────────────────────────────────

async def query_rewriter_node(state: State) -> dict:
    rewritten = await _rewrite_query(state["question"])
    # [R3] was print() — use logger.info
    logger.info(f"Rewritten query: {rewritten}")
    return {"rewritten_query": rewritten}


async def dense_retriever_node(state: State) -> dict:
    docs = await _dense_retrieve(state["rewritten_query"])
    # [R3] was print()
    logger.info(f"Dense retrieval: {len(docs)} docs")
    return {"dense_docs": docs}


async def sparse_retriever_node(state: State) -> dict:
    docs = await _run_in_thread(_sparse_retrieve, state["rewritten_query"])
    # [R3] was print()
    logger.info(f"Sparse (BM25): {len(docs)} docs")
    return {"sparse_docs": docs}


def retrieval_merger_node(state: State) -> dict:
    # [R3] was print()
    logger.info(
        f"Merger: dense={len(state['dense_docs'])} "
        f"sparse={len(state['sparse_docs'])} docs"
    )
    return {}


async def reranker_node(state: State) -> dict:
    reranked = await _rerank(state["question"], state["dense_docs"], state["sparse_docs"])
    rn, _ = _get_reranker()
    # [R3] was print()
    logger.info(f"[{rn}] reranked → {len(reranked)} docs")
    return {"reranked_docs": reranked}


async def generator_node(state: State) -> dict:
    # Pass full message history so _generate can include prior turns
    answer = await _generate(
        state["question"],
        state["reranked_docs"],
        chat_history=state.get("messages", []),
    )
    return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}


# ── GRAPH FACTORY ─────────────────────────────────────────

def _build_graph():
    builder = StateGraph(State)
    builder.add_node("query_rewriter",   query_rewriter_node)
    builder.add_node("dense_retriever",  dense_retriever_node)
    builder.add_node("sparse_retriever", sparse_retriever_node)
    builder.add_node("retrieval_merger", retrieval_merger_node)
    builder.add_node("reranker",         reranker_node)
    builder.add_node("generator",        generator_node)

    builder.add_edge(START,              "query_rewriter")
    builder.add_edge("query_rewriter",   "dense_retriever")
    builder.add_edge("query_rewriter",   "sparse_retriever")
    builder.add_edge("dense_retriever",  "retrieval_merger")
    builder.add_edge("sparse_retriever", "retrieval_merger")
    builder.add_edge("retrieval_merger", "reranker")
    builder.add_edge("reranker",         "generator")
    builder.add_edge("generator",        END)
    return builder


# ── GRAPH (compiled lazily in init_agent) ─────────────────
# [C3] graph is no longer a module-level global with MemorySaver.
# It is set by init_agent() during lifespan, after pg_pool is ready.
graph = None

# Module-level checkpointer — kept open for application lifetime
_checkpointer = None

# [R1] Single definition — duplicate assignment removed.
# Holds the psycopg AsyncConnectionPool opened in init_agent()
# so shutdown_agent() can close it cleanly.
_checkpointer_cm = None


async def init_agent(pg_pool) -> None:
    """
    [C3] Called once from main.py lifespan after pg_pool is created.

    langgraph-checkpoint-postgres uses psycopg3 internally — NOT asyncpg.
    This version-aware init handles both the old generator-based API and
    the newer context-manager-based API for from_conn_string().
    """
    global _vectorstore, graph, _checkpointer, _checkpointer_cm

    import os
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # 1. Vectorstore
    _vectorstore = await _run_in_thread(_init_vectorstore)

    # 2. Connect using psycopg3 async connection pool directly.
    # This is the most stable approach across all versions of
    # langgraph-checkpoint-postgres — pass a live psycopg connection
    pg_dsn = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@postgres:5432/ragdb"
    )

    # Use psycopg AsyncConnectionPool — the underlying driver this
    # library actually uses internally
    from psycopg_pool import AsyncConnectionPool

    # autocommit=True is REQUIRED for AsyncPostgresSaver.setup().
    # setup() runs CREATE INDEX CONCURRENTLY which PostgreSQL forbids
    # inside a transaction block. psycopg opens connections with
    # autocommit=False by default (implicit transaction on every statement),
    # which causes:
    #   psycopg.errors.ActiveSqlTransaction:
    #   CREATE INDEX CONCURRENTLY cannot run inside a transaction block
    # Passing autocommit=True via kwargs fixes this for every connection
    # in the pool. Regular checkpoint reads/writes work correctly with
    # autocommit=True — psycopg manages explicit transactions when needed.
    _pool = AsyncConnectionPool(
        conninfo=pg_dsn,
        min_size=2,
        max_size=10,
        open=False,   # don't open in constructor — open explicitly below
        kwargs={"autocommit": True},
    )
    await _pool.open()

    _checkpointer    = AsyncPostgresSaver(_pool)
    _checkpointer_cm = _pool   # store for shutdown

    # 3. Create checkpoint tables
    # setup() runs CREATE TYPE / CREATE TABLE / CREATE INDEX CONCURRENTLY.
    # With multiple gunicorn workers all running lifespan simultaneously,
    # workers race to call setup(). The first succeeds; the rest hit
    # UniqueViolation on the checkpoint_blobs type that was just created.
    # We catch that specific error and treat it as a no-op — the schema
    # is already in place from the winning worker so all workers are safe
    # to proceed to graph compilation.
    # autocommit=True (set above) is required for CREATE INDEX CONCURRENTLY.
    try:
        await _checkpointer.setup()
    except Exception as exc:
        err = str(exc)
        if "already exists" in err or "UniqueViolation" in err:
            logger.info("Checkpoint schema already initialised by another worker — skipping setup")
        else:
            raise

    # 4. Compile graph
    graph = _build_graph().compile(checkpointer=_checkpointer)

    # [R3] was print()
    logger.info("Agent initialised: AsyncPostgresSaver checkpointer ready")


async def shutdown_agent() -> None:
    """Called from main.py lifespan on shutdown to close checkpointer pool."""
    global _checkpointer, _checkpointer_cm
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.close()
        except Exception:
            pass
        _checkpointer_cm = None
        _checkpointer    = None


# ── PUBLIC API ────────────────────────────────────────────

async def arun_rag(question: str, thread_id: str = "default") -> str:
    """
    Async entry point — use from FastAPI. Never blocks event loop.

    Only the new user message is passed into ainvoke. LangGraph loads
    the existing checkpoint state (conversation history) from PostgreSQL
    via AsyncPostgresSaver and merges the new message in via the
    add_messages reducer on the messages field.

    Passing the full state dict on every call was overwriting non-messages
    fields correctly but the pattern is cleaner this way and makes the
    checkpoint merge behaviour explicit.
    """
    if graph is None:
        raise RuntimeError("Agent not initialised — await init_agent(pg_pool) first")
    config = {"configurable": {"thread_id": thread_id}, "run_name": "RAG Graph"}
    result = await graph.ainvoke(
        {
            "messages":        [{"role": "user", "content": question}],
            "question":        question,
            "rewritten_query": "",
            "dense_docs":      [],
            "sparse_docs":     [],
            "reranked_docs":   [],
            "answer":          "",
        },
        config=config,
    )
    return result["answer"]


def run_rag(question: str, thread_id: str = "default") -> str:
    """Sync wrapper — for CLI / tests only. Do NOT call from async context."""
    return asyncio.run(arun_rag(question, thread_id))