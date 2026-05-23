"""
api/auth.py
Google OAuth + JWT session authentication for the Financial Advisor API.

Provides:
  - Google ID token verification
  - JWT session token creation & verification
  - FastAPI dependency for protected routes
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET or JWT_SECRET == "CHANGE_ME_IN_PRODUCTION":
    import warnings
    warnings.warn(
        "\n⚠️  JWT_SECRET is not set or uses the default value!\n"
        "   Anyone can forge authentication tokens.\n"
        "   Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\"\n"
        "   Then add JWT_SECRET=<value> to your .env file.\n",
        stacklevel=2,
    )
    if not JWT_SECRET:
        import secrets as _s
        JWT_SECRET = _s.token_urlsafe(32)

JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

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

        # Verify the issuer
        if idinfo["iss"] not in ("accounts.google.com", "https://accounts.google.com"):
            raise ValueError("Invalid issuer.")

        return {
            "sub": idinfo["sub"],  # Google unique user ID
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


# ── JWT Session Token ─────────────────────────────────────────────────────────


def create_access_token(user_id: int, email: str, role: str = "user") -> str:
    """
    Create a signed JWT session token.

    Payload:
        sub: user_id (as string per JWT spec)
        email: user email
        role: user role (user/admin)
        exp: expiration timestamp
        iat: issued at timestamp
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT session token.

    Raises:
        HTTPException 401 if token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired. Please login again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


# ── FastAPI Dependencies ──────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    FastAPI dependency that extracts and validates the JWT from the
    Authorization header.

    Usage:
        @router.post("/protected")
        def my_endpoint(user: dict = Depends(get_current_user)):
            user_id = user["sub"]
            ...

    Returns:
        Decoded JWT payload dict with keys: sub, email, role, exp, iat
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


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
