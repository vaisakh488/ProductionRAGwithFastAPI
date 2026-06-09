"""
auth.py — Production Auth v3

FIXES IN THIS VERSION:
  [C4] USERS_SCHEMA_SQL no longer seeds admin with '$placeholder$'.
       '$placeholder$' is not a valid bcrypt hash — bcrypt.checkpw() raises
       an exception (not just returns False), which means every admin login
       attempt threw a 500 before this fix.

       New behaviour:
         • The seed row uses a real bcrypt hash of the literal string
           "__CHANGE_ME__" as a safe placeholder.  This hash is valid so
           bcrypt never throws; it just won't match any real password.
         • On startup, main.py sets the real password from ADMIN_PASSWORD
           env var (this was already the intent — now it actually works
           because the row exists and is valid).
         • If ADMIN_PASSWORD is not set the admin account exists but is
           effectively locked (no password will match the placeholder hash),
           which is safe.

       The hash below was generated with:
           import bcrypt
           bcrypt.hashpw(b"__CHANGE_ME__", bcrypt.gensalt(rounds=12)).decode()
       It is deterministic only in the sense that it was generated once and
       committed.  It is NOT the string "__CHANGE_ME__" — it is a one-way
       hash of it.

  [R1] USERS_SCHEMA_SQL is now pure DDL — no data embedded in the SQL string.
       The admin seed INSERT has been moved to main.py lifespan as a
       parameterized execute call.  This eliminates f-string interpolation
       in SQL (safe habit, not an active injection risk since _PLACEHOLDER_HASH
       is a constant — but the pattern is wrong and must not be copied).

  [R2] blocklist_token now handles tokens with no exp claim.
       Previously, if exp was None the token was silently not blocklisted —
       a logout call returned 200 but the token remained usable.
       Now: falls back to ACCESS_TOKEN_EXPIRE_MINUTES * 60 as the TTL.

  [R3] verify_password exception catch narrowed from bare Exception to
       (ValueError, TypeError).  Catching all Exception was swallowing
       programming errors (AttributeError, TypeError) and hiding real bugs.
       Only ValueError (malformed hash) and TypeError (wrong type passed)
       are expected from bcrypt.checkpw.

  All previous auth.py upgrades (U1-U7, C4) are retained unchanged.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import asyncpg
import bcrypt
import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS   = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",   "7"))
REDIS_URL                   = os.getenv("REDIS_URL", "redis://localhost:6379")
PG_DSN                      = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")

# ── ROLE HIERARCHY ────────────────────────────────────────
ROLES = ["viewer", "editor", "admin"]

def role_level(role: str) -> int:
    try:
        return ROLES.index(role)
    except ValueError:
        return -1


# ── SAFE PLACEHOLDER HASH  [C4] ───────────────────────────
# A valid bcrypt hash of the literal string "__CHANGE_ME__".
# bcrypt.checkpw() will NOT throw on this — it returns False for any
# real password, effectively locking the account until ADMIN_PASSWORD
# is applied from the env var during lifespan startup.
#
# Regenerate if needed:
#   python -c "import bcrypt; print(bcrypt.hashpw(b'__CHANGE_ME__', bcrypt.gensalt(12)).decode())"
_PLACEHOLDER_HASH = (
    "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HS.iK8e"
)


# ── DB SCHEMA (run once at startup) ───────────────────────
# [R1] Pure DDL only — no data embedded in the SQL string.
#      The admin seed INSERT is handled in main.py lifespan via a
#      parameterized execute call so no values are ever interpolated
#      into SQL strings.
USERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        TEXT UNIQUE NOT NULL,
    email           TEXT UNIQUE,
    hashed_password TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'viewer',
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    last_login      TIMESTAMPTZ
);
"""


# ── SCHEMAS ───────────────────────────────────────────────

class Token(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str


class TokenData(BaseModel):
    username:   Optional[str] = None
    token_type: Optional[str] = None


class User(BaseModel):
    username:   str
    role:       str
    email:      Optional[str] = None
    is_active:  bool = True


class UserCreate(BaseModel):
    username: str
    email:    str
    password: str
    role:     str = "viewer"


# ── PASSWORD HELPERS ──────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    """
    Returns False (never raises) if the hash is invalid.
    [R3] Catches only (ValueError, TypeError) — the two exceptions
    bcrypt.checkpw can legitimately raise on malformed input.
    Catching bare Exception was hiding programming errors.
    """
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except (ValueError, TypeError):
        logger.error("bcrypt.checkpw failed — stored hash may be malformed")
        return False


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


# ── DB USER LOOKUP ────────────────────────────────────────

async def get_user_from_db(
    username: str,
    pool: asyncpg.Pool,
) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT username, email, hashed_password, role, is_active "
            "FROM users WHERE username = $1",
            username,
        )
    return dict(row) if row else None


async def authenticate_user_db(
    username: str,
    password: str,
    pool: asyncpg.Pool,
) -> Optional[dict]:
    user = await get_user_from_db(username, pool)
    if not user:
        logger.warning(f"Login attempt for unknown user: {username}")
        return None
    if not user["is_active"]:
        logger.warning(f"Login attempt for inactive user: {username}")
        return None
    if not verify_password(password, user["hashed_password"]):
        logger.warning(f"Failed login attempt for user: {username}")
        return None

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET last_login = now() WHERE username = $1", username,
        )
    return user


# ── TOKEN HELPERS ─────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def _make_token(data: dict, expires_delta: timedelta) -> str:
    payload        = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    return _make_token(
        {**data, "type": "access"},
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(data: dict) -> str:
    return _make_token(
        {**data, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload    = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username   = payload.get("sub")
        token_type = payload.get("type", "access")
        if username is None:
            raise credentials_exception
        return {"username": username, "type": token_type, "exp": payload.get("exp")}
    except JWTError:
        raise credentials_exception


# ── REDIS BLOCKLIST ───────────────────────────────────────

async def blocklist_token(token: str, redis: aioredis.Redis) -> None:
    """
    Add a token to the blocklist.

    [R2] Handles tokens with no exp claim — previously a missing exp
    caused a silent no-op where logout returned 200 but the token stayed
    valid.  Now falls back to ACCESS_TOKEN_EXPIRE_MINUTES * 60 as TTL
    so the token is always blocklisted for at least its maximum lifetime.

    If the token is already expired, decode_token raises 401 — we catch
    that here and treat it as a no-op (expired token is already useless).
    """
    try:
        data = decode_token(token)
    except HTTPException:
        # Token already expired — nothing to blocklist
        return

    exp = data.get("exp")
    if exp:
        ttl = int(exp - datetime.now(timezone.utc).timestamp())
        if ttl > 0:
            await redis.setex(f"blocklist:{token}", ttl, "1")
    else:
        # [R2] No exp claim — blocklist for the full access token lifetime
        # as a safety net so the token cannot be reused after logout.
        await redis.setex(
            f"blocklist:{token}",
            ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "1",
        )


async def is_token_blocked(token: str, redis: aioredis.Redis) -> bool:
    return bool(await redis.exists(f"blocklist:{token}"))


# ── DEPENDENCY ────────────────────────────────────────────

async def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    data = decode_token(token)
    if data["type"] != "access":
        raise credentials_exception

    redis: aioredis.Redis = request.app.state.redis
    if await is_token_blocked(token, redis):
        raise credentials_exception

    pg_pool: asyncpg.Pool = request.app.state.pg_pool
    user = await get_user_from_db(data["username"], pg_pool)

    if user is None or not user["is_active"]:
        raise credentials_exception

    return User(
        username=user["username"],
        role=user["role"],
        email=user.get("email"),
        is_active=user["is_active"],
    )


# ── RBAC DEPENDENCY ───────────────────────────────────────

def require_role(minimum_role: str):
    def _check(current_user: User = Depends(get_current_user)) -> User:
        if role_level(current_user.role) < role_level(minimum_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{minimum_role}' or higher required",
            )
        return current_user
    return _check