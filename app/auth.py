"""Authentication and user management.

Each user gets an API key. Keys are scoped to tiers (free/basic/pro).
"""

import uuid
import hashlib
import hmac
import secrets
from app.database import get_connection
from app.models import CreateUserRequest, UserResponse


def generate_api_key() -> str:
    """Generate a cryptographically random API key."""
    return f"aim_{secrets.token_hex(24)}"


def create_user(req: CreateUserRequest) -> UserResponse:
    """Create a new user with an API key."""
    conn = get_connection()
    api_key = generate_api_key()

    conn.execute(
        "INSERT INTO users (id, api_key, tier) VALUES (?, ?, ?)",
        (req.user_id, api_key, req.tier),
    )
    conn.commit()

    user = conn.execute(
        "SELECT id, api_key, tier, request_count FROM users WHERE id = ?",
        (req.user_id,),
    ).fetchone()

    conn.close()
    return UserResponse(
        user_id=user["id"],
        api_key=user["api_key"],
        tier=user["tier"],
        request_count=user["request_count"],
    )


def validate_api_key(api_key: str) -> dict | None:
    """Validate an API key and return user info, or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, tier, request_count FROM users WHERE api_key = ?",
        (api_key,),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "user_id": row["id"],
        "tier": row["tier"],
        "request_count": row["request_count"],
    }


def get_or_create_agent(agent_id: str, name: str, user_id: str) -> dict:
    """Get an existing agent or create one if it doesn't exist."""
    conn = get_connection()

    agent = conn.execute(
        "SELECT id, name, user_id FROM agents WHERE id = ? AND user_id = ?",
        (agent_id, user_id),
    ).fetchone()

    if agent is None:
        conn.execute(
            "INSERT INTO agents (id, user_id, name) VALUES (?, ?, ?)",
            (agent_id, user_id, name),
        )
        conn.commit()
        agent = conn.execute(
            "SELECT id, name, user_id FROM agents WHERE id = ?",
            (agent_id,),
        ).fetchone()

    conn.close()
    return {"agent_id": agent["id"], "name": agent["name"], "user_id": agent["user_id"]}
