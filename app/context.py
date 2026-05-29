"""Context injection — retrieve and format memories as context for AI agents.

Returns a formatted string suitable for injection into an LLM prompt.
"""

import asyncio
from app.database import get_connection, deserialize_vector
from app.embeddings import embed_single, cosine_similarity
from app.models import ContextInjectRequest, MemoryResponse


async def inject_context(
    agent_id: str,
    user_id: str,
    req: ContextInjectRequest,
) -> str:
    """Retrieve relevant memories and format them as injectable context.

    Returns a string like:

        [Relevant Memories]
        - (2024-01-15) User prefers Python over JavaScript.
        - (2024-01-10) Project uses PostgreSQL, not MySQL.

    The result is capped to max_tokens (roughly 4 chars per token).
    """
    # Get relevant memories via semantic search
    memories = await _get_relevant_memories(agent_id, user_id, req.query)

    # Format as context string
    lines = ["[Relevant Memories]"]
    char_limit = req.max_tokens * 3  # rough estimate: 3 chars per token
    current_chars = len(lines[0])

    for mem in memories:
        line = f"- ({mem.created_at[:10]}) {mem.content}"
        if mem.summary and mem.summary != "compressed":
            line = f"- ({mem.created_at[:10]}) [Summary: {mem.summary}]"

        if current_chars + len(line) > char_limit:
            break

        lines.append(line)
        current_chars += len(line) + 1

    return "\n".join(lines)


async def _get_relevant_memories(
    agent_id: str,
    user_id: str,
    query: str,
    limit: int = 20,
) -> list[MemoryResponse]:
    """Get memories most relevant to the query."""
    conn = get_connection()

    # Try vector search first
    try:
        query_vector = await embed_single(query)

        rows = conn.execute(
            """SELECT mv.memory_id, mv.vector, m.*
               FROM memory_vectors mv
               JOIN memories m ON mv.memory_id = m.id
               WHERE m.agent_id = ? AND m.user_id = ?
                 AND m.compressed = 0
               LIMIT 500""",
            (agent_id, user_id),
        ).fetchall()

        if rows:
            scored = []
            for row in rows:
                vec = deserialize_vector(row["vector"])
                sim = cosine_similarity(query_vector, vec)
                scored.append((sim, row))

            scored.sort(key=lambda x: x[0], reverse=True)
            rows = [r for _, r in scored[:limit]]
    except Exception:
        # Fallback to recent memories
        rows = conn.execute(
            """SELECT * FROM memories
               WHERE agent_id = ? AND user_id = ? AND compressed = 0
               ORDER BY created_at DESC
               LIMIT ?""",
            (agent_id, user_id, limit),
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
