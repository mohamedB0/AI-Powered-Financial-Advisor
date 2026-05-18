"""
api/routes/conversations.py
Chat history CRUD — conversations and messages per user.

GET    /conversations              — List user's conversations
POST   /conversations              — Create a new conversation
GET    /conversations/{id}         — Get conversation with all messages
PUT    /conversations/{id}         — Update conversation title
DELETE /conversations/{id}         — Delete a conversation
POST   /conversations/{id}/messages — Add a message to a conversation
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from api.auth import get_current_user

router = APIRouter(prefix="/conversations", tags=["Chat History"])


# ── Models ────────────────────────────────────────────────────────────────────


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    route: Optional[str] = None
    created_at: str


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_message: Optional[str] = None


class ConversationDetail(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    messages: List[MessageOut]


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Chat"


class UpdateConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class AddMessageRequest(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)
    route: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_db():
    from storage.relational_db import RelationalDB
    return RelationalDB()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/", response_model=List[ConversationSummary])
def list_conversations(
    limit: int = Query(50, ge=1, le=200),
    user: dict = Depends(get_current_user),
):
    """List all conversations for the authenticated user, most recent first."""
    db = _get_db()
    user_id = int(user["sub"])

    rows = db.execute_sql(
        """
        SELECT c.id, c.title, c.created_at, c.updated_at,
               COUNT(m.id) as message_count,
               (SELECT content FROM messages
                WHERE conversation_id = c.id
                ORDER BY created_at DESC LIMIT 1) as last_message
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id = :user_id
        GROUP BY c.id, c.title, c.created_at, c.updated_at
        ORDER BY c.updated_at DESC
        LIMIT :limit
        """,
        {"user_id": user_id, "limit": limit},
    )

    return [
        ConversationSummary(
            id=r["id"],
            title=r["title"] or "New Chat",
            created_at=str(r["created_at"] or ""),
            updated_at=str(r["updated_at"] or ""),
            message_count=r["message_count"],
            last_message=(r["last_message"] or "")[:100],
        )
        for r in rows
    ]


@router.post("/", response_model=ConversationDetail, status_code=status.HTTP_201_CREATED)
def create_conversation(
    req: CreateConversationRequest,
    user: dict = Depends(get_current_user),
):
    """Create a new empty conversation."""
    db = _get_db()
    user_id = int(user["sub"])
    now = datetime.now(timezone.utc).isoformat()

    db.execute_sql(
        """
        INSERT INTO conversations (user_id, title, created_at, updated_at)
        VALUES (:user_id, :title, :now, :now)
        """,
        {"user_id": user_id, "title": req.title or "New Chat", "now": now},
    )

    # Get the created conversation
    rows = db.execute_sql(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = :user_id ORDER BY id DESC LIMIT 1",
        {"user_id": user_id},
    )

    if not rows:
        raise HTTPException(status_code=500, detail="Failed to create conversation.")

    conv = rows[0]
    return ConversationDetail(
        id=conv["id"],
        title=conv["title"],
        created_at=str(conv["created_at"]),
        updated_at=str(conv["updated_at"]),
        messages=[],
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: int,
    user: dict = Depends(get_current_user),
):
    """Get a conversation with all its messages."""
    db = _get_db()
    user_id = int(user["sub"])

    # Verify ownership
    conv_rows = db.execute_sql(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = :id AND user_id = :user_id",
        {"id": conversation_id, "user_id": user_id},
    )
    if not conv_rows:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    conv = conv_rows[0]

    # Fetch messages
    msg_rows = db.execute_sql(
        "SELECT id, role, content, route, created_at FROM messages WHERE conversation_id = :conv_id ORDER BY created_at ASC",
        {"conv_id": conversation_id},
    )

    messages = [
        MessageOut(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            route=m.get("route"),
            created_at=str(m["created_at"] or ""),
        )
        for m in msg_rows
    ]

    return ConversationDetail(
        id=conv["id"],
        title=conv["title"],
        created_at=str(conv["created_at"]),
        updated_at=str(conv["updated_at"]),
        messages=messages,
    )


@router.put("/{conversation_id}", response_model=ConversationSummary)
def update_conversation(
    conversation_id: int,
    req: UpdateConversationRequest,
    user: dict = Depends(get_current_user),
):
    """Update conversation title."""
    db = _get_db()
    user_id = int(user["sub"])
    now = datetime.now(timezone.utc).isoformat()

    # Verify ownership
    conv_rows = db.execute_sql(
        "SELECT id FROM conversations WHERE id = :id AND user_id = :user_id",
        {"id": conversation_id, "user_id": user_id},
    )
    if not conv_rows:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    db.execute_sql(
        "UPDATE conversations SET title = :title, updated_at = :now WHERE id = :id",
        {"title": req.title, "now": now, "id": conversation_id},
    )

    return ConversationSummary(
        id=conversation_id,
        title=req.title,
        created_at="",
        updated_at=now,
        message_count=0,
        last_message=None,
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    user: dict = Depends(get_current_user),
):
    """Delete a conversation and all its messages."""
    db = _get_db()
    user_id = int(user["sub"])

    # Verify ownership
    conv_rows = db.execute_sql(
        "SELECT id FROM conversations WHERE id = :id AND user_id = :user_id",
        {"id": conversation_id, "user_id": user_id},
    )
    if not conv_rows:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    # Delete messages first (CASCADE should handle this, but be explicit)
    db.execute_sql("DELETE FROM messages WHERE conversation_id = :id", {"id": conversation_id})
    db.execute_sql("DELETE FROM conversations WHERE id = :id", {"id": conversation_id})


@router.post("/{conversation_id}/messages", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
def add_message(
    conversation_id: int,
    req: AddMessageRequest,
    user: dict = Depends(get_current_user),
):
    """Add a message to a conversation."""
    db = _get_db()
    user_id = int(user["sub"])
    now = datetime.now(timezone.utc).isoformat()

    # Verify ownership
    conv_rows = db.execute_sql(
        "SELECT id FROM conversations WHERE id = :id AND user_id = :user_id",
        {"id": conversation_id, "user_id": user_id},
    )
    if not conv_rows:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    # Insert message
    db.execute_sql(
        """
        INSERT INTO messages (conversation_id, role, content, route, created_at)
        VALUES (:conv_id, :role, :content, :route, :now)
        """,
        {
            "conv_id": conversation_id,
            "role": req.role,
            "content": req.content,
            "route": req.route,
            "now": now,
        },
    )

    # Update conversation timestamp + auto-title from first user message
    title_update = ""
    if req.role == "user":
        # Check if this is the first user message — use it as conversation title
        count_rows = db.execute_sql(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = :id AND role = 'user'",
            {"id": conversation_id},
        )
        if count_rows and count_rows[0]["cnt"] <= 1:
            auto_title = req.content[:80].strip()
            if len(req.content) > 80:
                auto_title += "..."
            db.execute_sql(
                "UPDATE conversations SET title = :title, updated_at = :now WHERE id = :id",
                {"title": auto_title, "now": now, "id": conversation_id},
            )
        else:
            db.execute_sql(
                "UPDATE conversations SET updated_at = :now WHERE id = :id",
                {"now": now, "id": conversation_id},
            )
    else:
        db.execute_sql(
            "UPDATE conversations SET updated_at = :now WHERE id = :id",
            {"now": now, "id": conversation_id},
        )

    # Fetch created message
    msg_rows = db.execute_sql(
        "SELECT id, role, content, route, created_at FROM messages WHERE conversation_id = :conv_id ORDER BY id DESC LIMIT 1",
        {"conv_id": conversation_id},
    )

    m = msg_rows[0]
    return MessageOut(
        id=m["id"],
        role=m["role"],
        content=m["content"],
        route=m.get("route"),
        created_at=str(m["created_at"] or ""),
    )
