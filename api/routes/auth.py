"""
api/routes/auth.py
Authentication routes: Email/Password + Google OAuth.

POST /auth/register — Create a new user account
POST /auth/login    — Authenticate with email + password
POST /auth/google   — Authenticate with a Google ID token
GET  /auth/me       — Get current user profile (protected)
POST /auth/logout   — Destroy the server-side session (real revocation)
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, Request, status
from pydantic import BaseModel, EmailStr, Field

from api.auth import (
    AUTH_COOKIE_NAME,
    AUTH_COOKIE_SAMESITE,
    AUTH_COOKIE_SECURE,
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    create_session,
    delete_session,
    get_current_user,
    hash_password,
    verify_password,
    verify_google_token,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Request/Response Models ───────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleAuthRequest(BaseModel):
    """Frontend sends the Google ID token obtained from Google Sign-In."""
    id_token: str


class SessionResponse(BaseModel):
    """Returned on successful login/register.  No token is exposed to JS."""
    token_type: str = "session"
    expires_in_seconds: int
    user: dict


class ProfileResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    picture: Optional[str]
    role: str
    created_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _set_session_cookie(response: Response, session_id: str) -> None:
    """
    Store the session ID in an httpOnly cookie.

    The session ID is opaque — it carries no user data itself. The actual
    user payload lives in Redis server-side.
    """
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite=AUTH_COOKIE_SAMESITE,
        secure=AUTH_COOKIE_SECURE,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        samesite=AUTH_COOKIE_SAMESITE,
        secure=AUTH_COOKIE_SECURE,
    )


def _get_db():
    from storage.relational_db import RelationalDB
    return RelationalDB()


def _get_user_by_email(email: str):
    """Query the users table for a user by email."""
    db = _get_db()
    rows = db.execute_sql(
        "SELECT id, email, full_name, picture, google_sub, role, is_active, hashed_password, created_at "
        "FROM users WHERE email = :email LIMIT 1",
        {"email": email},
    )
    return rows[0] if rows else None


def _get_user_by_google_sub(google_sub: str):
    """Query the users table for a user by Google subject ID."""
    db = _get_db()
    rows = db.execute_sql(
        "SELECT id, email, full_name, picture, google_sub, role, is_active, created_at "
        "FROM users WHERE google_sub = :google_sub LIMIT 1",
        {"google_sub": google_sub},
    )
    return rows[0] if rows else None


def _create_google_user(email: str, full_name: str, picture: str, google_sub: str) -> dict:
    """Create a new user from Google OAuth data."""
    db = _get_db()

    # First user becomes admin
    user_count = db.execute_sql("SELECT COUNT(*) as cnt FROM users")
    role = "admin" if user_count[0]["cnt"] == 0 else "user"

    db.execute_sql(
        "INSERT INTO users (email, full_name, picture, google_sub, role, is_active, created_at) "
        "VALUES (:email, :full_name, :picture, :google_sub, :role, 1, :created_at)",
        {
            "email": email,
            "full_name": full_name,
            "picture": picture,
            "google_sub": google_sub,
            "role": role,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return _get_user_by_email(email)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/register", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, response: Response):
    """Create a new user account with email + password."""
    from api.tracing import trace_stage

    with trace_stage("auth_register", {"auth.method": "email", "auth.email": req.email}) as span:
        existing = _get_user_by_email(req.email)
        if existing:
            span.set_attribute("auth.result", "conflict")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email already exists.",
            )

        db = _get_db()

        # First user becomes admin
        user_count = db.execute_sql("SELECT COUNT(*) as cnt FROM users")
        role = "admin" if user_count[0]["cnt"] == 0 else "user"

        hashed = hash_password(req.password)

        db.execute_sql(
            "INSERT INTO users (email, hashed_password, full_name, role, is_active, created_at) "
            "VALUES (:email, :hashed_password, :full_name, :role, 1, :created_at)",
            {
                "email": req.email,
                "hashed_password": hashed,
                "full_name": req.full_name or "",
                "role": role,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        user = _get_user_by_email(req.email)
        if not user:
            span.set_attribute("auth.result", "error")
            raise HTTPException(status_code=500, detail="User creation failed.")

        session_id = create_session(user["id"], user["email"], user["role"])
        _set_session_cookie(response, session_id)

        span.set_attribute("auth.result", "success")
        span.set_attribute("auth.user_id", user["id"])
        span.set_attribute("auth.role", user["role"])

    return SessionResponse(
        expires_in_seconds=SESSION_TTL_SECONDS,
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user.get("full_name", ""),
            "picture": user.get("picture", ""),
            "role": user["role"],
        },
    )


@router.post("/login", response_model=SessionResponse)
def login(req: LoginRequest, response: Response):
    """Authenticate with email + password and create a server-side session."""
    from api.tracing import trace_stage

    with trace_stage("auth_login", {"auth.method": "email", "auth.email": req.email}) as span:
        user = _get_user_by_email(req.email)
        if not user:
            span.set_attribute("auth.result", "invalid_credentials")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        if not user.get("is_active", 1):
            span.set_attribute("auth.result", "disabled")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled. Contact an administrator.",
            )

        hashed = user.get("hashed_password", "")
        if not hashed or not verify_password(req.password, hashed):
            span.set_attribute("auth.result", "invalid_credentials")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        # Update last_login
        db = _get_db()
        try:
            db.execute_sql(
                "UPDATE users SET last_login = :now WHERE id = :id",
                {"now": datetime.now(timezone.utc).isoformat(), "id": user["id"]},
            )
        except Exception:
            pass

        session_id = create_session(user["id"], user["email"], user["role"])
        _set_session_cookie(response, session_id)

        span.set_attribute("auth.result", "success")
        span.set_attribute("auth.user_id", user["id"])
        span.set_attribute("auth.user_name", user.get("full_name", ""))
        span.set_attribute("auth.role", user["role"])

    return SessionResponse(
        expires_in_seconds=SESSION_TTL_SECONDS,
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user.get("full_name", ""),
            "picture": user.get("picture", ""),
            "role": user["role"],
        },
    )


@router.post("/google", response_model=SessionResponse)
def google_auth(req: GoogleAuthRequest, response: Response):
    """Authenticate using a Google ID token."""
    from api.tracing import trace_stage

    with trace_stage("auth_google", {"auth.method": "google"}) as span:
        google_info = verify_google_token(req.id_token)

        if not google_info.get("email_verified"):
            span.set_attribute("auth.result", "email_not_verified")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Google email is not verified.",
            )

        email = google_info["email"]
        google_sub = google_info["sub"]
        span.set_attribute("auth.email", email)

        # Try to find existing user by Google sub ID
        user = _get_user_by_google_sub(google_sub)

        if not user:
            # Try by email (link existing account)
            user = _get_user_by_email(email)

            if user:
                # Link the Google sub to the existing user
                db = _get_db()
                db.execute_sql(
                    "UPDATE users SET google_sub = :google_sub, picture = :picture WHERE id = :id",
                    {"google_sub": google_sub, "picture": google_info.get("picture", ""), "id": user["id"]},
                )
                user = _get_user_by_email(email)
                span.set_attribute("auth.action", "linked_google")
            else:
                # Create new user
                user = _create_google_user(
                    email=email,
                    full_name=google_info.get("name", ""),
                    picture=google_info.get("picture", ""),
                    google_sub=google_sub,
                )
                span.set_attribute("auth.action", "created_new_user")
        else:
            span.set_attribute("auth.action", "existing_user")

        if not user:
            span.set_attribute("auth.result", "error")
            raise HTTPException(status_code=500, detail="User creation failed.")

        if not user.get("is_active", 1):
            span.set_attribute("auth.result", "disabled")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled. Contact an administrator.",
            )

        # Update last_login
        db = _get_db()
        try:
            db.execute_sql(
                "UPDATE users SET last_login = :now WHERE id = :id",
                {"now": datetime.now(timezone.utc).isoformat(), "id": user["id"]},
            )
        except Exception:
            pass  # Non-critical

        session_id = create_session(user["id"], user["email"], user["role"])
        _set_session_cookie(response, session_id)

        span.set_attribute("auth.result", "success")
        span.set_attribute("auth.user_id", user["id"])
        span.set_attribute("auth.user_name", user.get("full_name", ""))
        span.set_attribute("auth.role", user["role"])

    return SessionResponse(
        expires_in_seconds=SESSION_TTL_SECONDS,
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user.get("full_name", ""),
            "picture": user.get("picture", ""),
            "role": user["role"],
        },
    )


@router.get("/me", response_model=ProfileResponse)
def get_profile(current_user: dict = Depends(get_current_user)):
    """
    Get the authenticated user's profile.
    Requires a valid session cookie or Bearer session ID header.
    """
    user = _get_user_by_email(current_user["email"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    return ProfileResponse(
        id=user["id"],
        email=user["email"],
        full_name=user.get("full_name"),
        picture=user.get("picture"),
        role=user["role"],
        created_at=str(user.get("created_at", "")),
    )


@router.post("/logout")
def logout(request: Request, response: Response):
    """
    Destroy the server-side session in Redis and clear the session cookie.

    Unlike the old JWT approach, this truly revokes the session immediately —
    the session ID is deleted from Redis so it cannot be reused even if
    the cookie is somehow retained.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        delete_session(session_id)
    _clear_session_cookie(response)
    return {"detail": "Logged out."}
