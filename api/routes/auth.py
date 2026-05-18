"""
api/routes/auth.py
Authentication routes: register, login, profile.

POST /auth/register — Create a new user account
POST /auth/login    — Authenticate and receive a JWT token
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


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int
    user: dict


class ProfileResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
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
        "SELECT id, email, hashed_password, full_name, role, is_active, created_at "
        "FROM users WHERE email = :email LIMIT 1",
        {"email": email},
    )
    return rows[0] if rows else None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """
    Create a new user account and return a JWT token.

    - Email must be unique.
    - Password is hashed with bcrypt before storage.
    - First registered user automatically gets the 'admin' role.
    """
    # Check if email already exists
    existing = _get_user_by_email(req.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    db = _get_db()

    # First user becomes admin
    user_count = db.execute_sql("SELECT COUNT(*) as cnt FROM users")
    role = "admin" if user_count[0]["cnt"] == 0 else "user"

    hashed = hash_password(req.password)

    # Insert new user
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

    # Fetch the created user to get their ID
    user = _get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=500, detail="User creation failed.")

    # Generate token
    from api.auth import JWT_EXPIRATION_HOURS

    token = create_access_token(user["id"], user["email"], user["role"])

    return TokenResponse(
        access_token=token,
        expires_in_hours=JWT_EXPIRATION_HOURS,
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user.get("full_name", ""),
            "role": user["role"],
        },
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """
    Authenticate with email + password and receive a JWT token.
    """
    user = _get_user_by_email(req.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.get("is_active", 1):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact an administrator.",
        )

    if not verify_password(req.password, user["hashed_password"]):
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
        pass  # Non-critical

    # Generate token
    from api.auth import JWT_EXPIRATION_HOURS

    token = create_access_token(user["id"], user["email"], user["role"])

    return TokenResponse(
        access_token=token,
        expires_in_hours=JWT_EXPIRATION_HOURS,
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user.get("full_name", ""),
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
        role=user["role"],
        created_at=str(user.get("created_at", "")),
    )
