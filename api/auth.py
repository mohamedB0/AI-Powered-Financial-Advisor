"""
api/auth.py
Authentication utilities for the Financial Advisor API.

Provides:
  - Password hashing (bcrypt)
  - Google ID token verification
  - Redis-backed server-side session management
  - FastAPI dependency for protected routes
"""

import json
import os
import secrets
from typing import Optional

import bcrypt
import redis as redis_lib
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

_ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

# Redis connection — reuses the same Redis instance as Celery
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
USE_UPSTASH = os.getenv("USE_UPSTASH", "false").lower() == "true"

# Sessions live in Redis DB 1 (Celery uses DB 0 by default) to avoid key collisions
REDIS_SESSION_DB = int(os.getenv("REDIS_SESSION_DB", "1"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))  # default 24 h
SESSION_COOKIE_NAME = "session_id"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# ── Session Cookie settings ──────────────────────────────────────────────────
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower()

# Keep this alias so that any existing import of AUTH_COOKIE_NAME still works
AUTH_COOKIE_NAME = SESSION_COOKIE_NAME


# ── Redis Client ──────────────────────────────────────────────────────────────

def _build_redis_client() -> redis_lib.Redis:
    """Build a Redis client for session storage."""
    if USE_UPSTASH and REDIS_HOST and REDIS_HOST not in ("local", "localhost"):
        # Cloud Redis (Upstash) — uses rediss:// (TLS)
        url = (
            f"rediss://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_SESSION_DB}"
            "?ssl_cert_reqs=CERT_NONE"
        )
        return redis_lib.from_url(url, decode_responses=True)

    return redis_lib.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_SESSION_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


_redis: Optional[redis_lib.Redis] = None


def get_redis() -> redis_lib.Redis:
    """Return a lazily-initialised Redis client (singleton per process)."""
    global _redis
    if _redis is None:
        _redis = _build_redis_client()
    return _redis


# ── Password Hashing ─────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


# ── Google OAuth Verification ─────────────────────────────────────────────────


def verify_google_token(id_token_str: str) -> dict:
    """
    Verify a Google ID token and return the user info payload.

    Returns dict with keys: sub, email, name, picture, email_verified
    Raises HTTPException 401 if token is invalid.
    """
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_CLIENT_ID not configured on server.",
        )

    try:
        idinfo = id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )

        if idinfo["iss"] not in ("accounts.google.com", "https://accounts.google.com"):
            raise ValueError("Invalid issuer.")

        return {
            "sub": idinfo["sub"],
            "email": idinfo.get("email", ""),
            "name": idinfo.get("name", ""),
            "picture": idinfo.get("picture", ""),
            "email_verified": idinfo.get("email_verified", False),
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google token: {str(e)}",
        )


# ── Redis Session Management ──────────────────────────────────────────────────


def create_session(user_id: int, email: str, role: str) -> str:
    """
    Create a new server-side session stored in Redis.

    Generates a cryptographically secure random session ID, stores the user
    payload as JSON in Redis with a TTL, and returns the session ID to be
    placed in the client cookie.

    Args:
        user_id: Database user ID
        email:   User email address
        role:    User role ('user' or 'admin')

    Returns:
        Opaque session ID string (to be stored in httpOnly cookie).
    """
    session_id = secrets.token_urlsafe(32)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
    }
    try:
        get_redis().setex(
            f"session:{session_id}",
            SESSION_TTL_SECONDS,
            json.dumps(payload),
        )
    except redis_lib.RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Session store unavailable: {exc}",
        )
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """
    Retrieve session data from Redis.

    Returns the user payload dict, or None if the session does not exist
    or has expired.
    """
    try:
        data = get_redis().get(f"session:{session_id}")
    except redis_lib.RedisError:
        return None
    return json.loads(data) if data else None


def refresh_session(session_id: str) -> None:
    """Reset the TTL on an active session (sliding expiry)."""
    try:
        get_redis().expire(f"session:{session_id}", SESSION_TTL_SECONDS)
    except redis_lib.RedisError:
        pass  # Non-critical — session will still expire naturally


def delete_session(session_id: str) -> None:
    """
    Destroy a session immediately (real server-side logout).

    After this call the session ID is invalid even if the cookie is still
    present on the client.
    """
    try:
        get_redis().delete(f"session:{session_id}")
    except redis_lib.RedisError:
        pass  # Best-effort — cookie will expire on its own


def delete_all_user_sessions(user_id: int) -> int:
    """
    Invalidate ALL active sessions for a given user (e.g. on password change).

    Scans Redis for session keys and deletes those whose payload contains the
    matching user_id.  Returns the number of sessions deleted.

    NOTE: SCAN is O(N) over all keys in the session DB. Acceptable for typical
    user counts; consider a secondary index for very large deployments.
    """
    r = get_redis()
    deleted = 0
    try:
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="session:*", count=100)
            for key in keys:
                raw = r.get(key)
                if raw:
                    try:
                        payload = json.loads(raw)
                        if payload.get("sub") == str(user_id):
                            r.delete(key)
                            deleted += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
            if cursor == 0:
                break
    except redis_lib.RedisError:
        pass
    return deleted


# ── FastAPI Dependencies ──────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None,
) -> dict:
    """
    FastAPI dependency that looks up the session from the httpOnly cookie
    (or a Bearer token carrying the session ID for programmatic API clients).

    Usage:
        @router.post("/protected")
        def my_endpoint(user: dict = Depends(get_current_user)):
            user_id = user["sub"]
            ...

    Returns:
        Session payload dict with keys: sub, email, role
    """
    session_id = None

    # 1. Bearer header (programmatic / API clients)
    if credentials is not None:
        session_id = credentials.credentials

    # 2. httpOnly cookie (browser clients)
    elif request is not None:
        session_id = request.cookies.get(SESSION_COOKIE_NAME)

    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_session(session_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Sliding window: reset TTL on every authenticated request
    refresh_session(session_id)
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """
    FastAPI dependency that requires admin role.

    Usage:
        @router.delete("/dangerous")
        def admin_only(user: dict = Depends(require_admin)):
            ...
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return user
