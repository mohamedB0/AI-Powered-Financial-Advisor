"""
api/routes/auth.py
Authentication routes: Email/Password + Google OAuth.

POST /auth/register — Create a new user account
POST /auth/login    — Authenticate with email + password
POST /auth/google   — Authenticate with a Google ID token
GET  /auth/me       — Get current user profile (protected)
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from api.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
    verify_google_token,
    JWT_EXPIRATION_HOURS,
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


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int
    user: dict


class ProfileResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    picture: Optional[str]
    role: str
    created_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_db():
    from storage.relational_db import RelationalDB
    return RelationalDB()


def _get_user_by_email(email: str):
    """Query the users table for a user by email."""
    db = _get_db()
    rows = db.execute_sql(
        "SELECT id, email, full_name, picture, google_sub, role, is_active, created_at "
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


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
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

        token = create_access_token(user["id"], user["email"], user["role"])

        span.set_attribute("auth.result", "success")
        span.set_attribute("auth.user_id", user["id"])
        span.set_attribute("auth.role", user["role"])

    return TokenResponse(
        access_token=token,
        expires_in_hours=JWT_EXPIRATION_HOURS,
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user.get("full_name", ""),
            "picture": user.get("picture", ""),
            "role": user["role"],
        },
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """Authenticate with email + password and receive a JWT token."""
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

        token = create_access_token(user["id"], user["email"], user["role"])

        span.set_attribute("auth.result", "success")
        span.set_attribute("auth.user_id", user["id"])
        span.set_attribute("auth.user_name", user.get("full_name", ""))
        span.set_attribute("auth.role", user["role"])

    return TokenResponse(
        access_token=token,
        expires_in_hours=JWT_EXPIRATION_HOURS,
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user.get("full_name", ""),
            "picture": user.get("picture", ""),
            "role": user["role"],
        },
    )


@router.post("/google", response_model=TokenResponse)
def google_auth(req: GoogleAuthRequest):
    """
    Authenticate using a Google ID token.
    """
    from api.tracing import trace_stage

    with trace_stage("auth_google", {"auth.method": "google"}) as span:
        # Verify the Google ID token
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

        # Generate session JWT
        token = create_access_token(user["id"], user["email"], user["role"])

        span.set_attribute("auth.result", "success")
        span.set_attribute("auth.user_id", user["id"])
        span.set_attribute("auth.user_name", user.get("full_name", ""))
        span.set_attribute("auth.role", user["role"])

    return TokenResponse(
        access_token=token,
        expires_in_hours=JWT_EXPIRATION_HOURS,
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
    Requires a valid Bearer token.
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
