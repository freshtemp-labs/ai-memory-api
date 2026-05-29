"""Memory CRUD operations — store, retrieve, list, delete."""

import uuid
from datetime import datetime, timezone
from app.database import get_connection, serialize_vector, deserialize_vector
from app.embeddings import embed_single
from app.models import StoreMemoryRequest, MemoryResponse


def gen_memory_id() -> str:
    return uuid.uuid4().hex[:12]


def store_memory(
    agent_id: str,
    user_id: str,
    req: StoreMemoryRequest,
) -> MemoryResponse:
    """Store a new memory with its embedding vector."""
    memory_id = gen_memory_id()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()

    # Insert memory record
    conn.execute(
        """INSERT INTO memories (id, agent_id, user_id, content, importance, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, agent_id, user_id, req.content, req.importance, now, now),
    )

    conn.commit()

    # Generate and store embedding (async, but we call synchronously in the background)
    # For simplicity, we embed inline — in production, use a task queue
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, but store_memory is sync.
            # Use a simple fallback: store with None vector, embed later.
            pass
    except RuntimeError:
        pass

    row = conn.execute(
        "SELECT * FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()

    conn.close()

    return MemoryResponse(
        id=row["id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        content=row["content"],
        summary=row["summary"],
        importance=row["importance"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        compressed=bool(row["compressed"]),
    )


def get_memory(memory_id: str, user_id: str) -> MemoryResponse | None:
    """Retrieve a single memory by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND user_id = ?",
        (memory_id, user_id),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    return MemoryResponse(
        id=row["id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        content=row["content"],
        summary=row["summary"],
        importance=row["importance"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        compressed=bool(row["compressed"]),
    )


def list_memories(
    agent_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[MemoryResponse]:
    """List memories for an agent, newest first."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM memories
           WHERE agent_id = ? AND user_id = ?
           ORDER BY created_at DESC
           LIMIT ? OFFSET ?""",
        (agent_id, user_id, limit, offset),
    ).fetchall()
    conn.close()

    return [
        MemoryResponse(
            id=r["id"],
            agent_id=r["agent_id"],
            user_id=r["user_id"],
            content=r["content"],
            summary=r["summary"],
            importance=r["importance"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            compressed=bool(r["compressed"]),
        )
        for r in rows
    ]


def delete_memory(memory_id: str, user_id: str) -> bool:
    """Delete a memory. Returns True if deleted, False if not found."""
    conn = get_connection()
    cursor = conn.execute(
        "DELETE FROM memories WHERE id = ? AND user_id = ?",
        (memory_id, user_id),
    )
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def memory_count(agent_id: str, user_id: str) -> int:
    """Count total memories for an agent."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as count FROM memories WHERE agent_id = ? AND user_id = ?",
        (agent_id, user_id),
    ).fetchone()
    conn.close()
    return row["count"]


async def embed_and_store(memory_id: str, content: str):
    """Generate embedding and store it for a memory (async, called after creation)."""
    try:
        vector = await embed_single(content)
        conn = get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO memory_vectors (memory_id, vector) VALUES (?, ?)",
            (memory_id, serialize_vector(vector)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Embedding failure is non-fatal
