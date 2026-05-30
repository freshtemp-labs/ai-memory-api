"""Semantic search over memories using vector similarity."""

import asyncio
import logging
from app.database import get_connection, deserialize_vector
from app.embeddings import embed_single, cosine_similarity
from app.models import SearchMemoryRequest, SearchResult, MemoryResponse

logger = logging.getLogger(__name__)


async def search_memories(
    agent_id: str,
    user_id: str,
    req: SearchMemoryRequest,
) -> list[SearchResult]:
    """Search memories by semantic similarity to the query.

    Falls back to keyword search if no vectors are available.
    """
    # Try semantic search first
    try:
        query_vector = await embed_single(req.query)
    except Exception as exc:
        logger.warning("embed_single failed in search for agent %s: %s", agent_id, exc)
        # Fallback to keyword search
        return _keyword_search(agent_id, user_id, req)

    # Get all vectors for this agent
    conn = get_connection()
    rows = conn.execute(
        """SELECT mv.memory_id, mv.vector, m.id, m.agent_id, m.user_id,
                  m.content, m.summary, m.importance, m.created_at, m.updated_at, m.compressed
           FROM memory_vectors mv
           JOIN memories m ON mv.memory_id = m.id
           WHERE m.agent_id = ? AND m.user_id = ?
           LIMIT 500""",
        (agent_id, user_id),
    ).fetchall()
    conn.close()

    if not rows:
        return _keyword_search(agent_id, user_id, req)

    # Compute similarity scores
    scored = []
    for row in rows:
        vec = deserialize_vector(row["vector"])
        sim = cosine_similarity(query_vector, vec)
        if sim >= req.min_similarity:
            scored.append((sim, row))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top-N
    results = []
    for sim, row in scored[: req.limit]:
        results.append(
            SearchResult(
                memory=MemoryResponse(
                    id=str(row["id"]),
                    agent_id=str(row["agent_id"]),
                    user_id=str(row["user_id"]),
                    content=row["content"],
                    summary=row["summary"],
                    importance=row["importance"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    compressed=bool(row["compressed"]),
                ),
                similarity=round(sim, 4),
            )
        )

    return results


def _keyword_search(
    agent_id: str,
    user_id: str,
    req: SearchMemoryRequest,
) -> list[SearchResult]:
    """Fallback: simple keyword search using LIKE."""
    conn = get_connection()

    # Split query into words for LIKE matching
    words = req.query.lower().split()
    conditions = " AND ".join(["LOWER(content) LIKE ?" for _ in words])
    params = [agent_id, user_id] + [f"%{w}%" for w in words] + [req.limit]

    rows = conn.execute(
        f"""SELECT * FROM memories
           WHERE agent_id = ? AND user_id = ?
             AND ({conditions})
           LIMIT ?""",
        tuple(params),
    ).fetchall()
    conn.close()

    return [
        SearchResult(
            memory=MemoryResponse(
                id=r["id"],
                agent_id=r["agent_id"],
                user_id=r["user_id"],
                content=r["content"],
                summary=r["summary"],
                importance=r["importance"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                compressed=bool(r["compressed"]),
            ),
            similarity=1.0,  # Keyword match, no real similarity score
        )
        for r in rows
    ]
