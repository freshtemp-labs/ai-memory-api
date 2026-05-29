"""Memory compression — automatically summarise old memories.

When an agent exceeds the compression threshold, the oldest N memories
are compressed into a single summary memory.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from app.database import get_connection
from app.models import CompressionResult
from app.config import settings


async def compress_memories(
    agent_id: str,
    user_id: str,
    target_count: int = 30,
) -> CompressionResult:
    """Compress oldest memories into summaries when over threshold.

    Strategy: Take the oldest uncompressed memories, batch them into groups,
    and create short summaries for each group. The original memories are
    marked as compressed (kept for record) and the summary becomes a new
    "active" memory.
    """
    conn = get_connection()

    # Count uncompressed memories
    row = conn.execute(
        "SELECT COUNT(*) as count FROM memories WHERE agent_id = ? AND user_id = ? AND compressed = 0",
        (agent_id, user_id),
    ).fetchone()
    total = row["count"]

    if total <= target_count:
        conn.close()
        return CompressionResult(
            original_count=total,
            new_count=total,
            compressed=0,
            summaries=[],
        )

    # Get oldest uncompressed memories to compress
    excess = total - target_count
    batch_size = min(excess, settings.compression_batch_size)

    rows = conn.execute(
        """SELECT id, content FROM memories
           WHERE agent_id = ? AND user_id = ? AND compressed = 0
           ORDER BY created_at ASC
           LIMIT ?""",
        (agent_id, user_id, batch_size),
    ).fetchall()

    # Create a summary memory
    summary_content = _generate_summary([r["content"] for r in rows])

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    summary_id = uuid.uuid4().hex[:12]

    conn.execute(
        """INSERT INTO memories (id, agent_id, user_id, content, summary, importance, created_at, updated_at, compressed)
           VALUES (?, ?, ?, ?, ?, 0.5, ?, ?, 0)""",
        (summary_id, agent_id, user_id, summary_content, None, now, now),
    )

    # Mark originals as compressed
    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"UPDATE memories SET compressed = 1, summary = 'compressed' WHERE id IN ({placeholders})",
        ids,
    )

    conn.commit()

    # Get new count
    new_count = conn.execute(
        "SELECT COUNT(*) as count FROM memories WHERE agent_id = ? AND user_id = ? AND compressed = 0",
        (agent_id, user_id),
    ).fetchone()["count"]

    conn.close()

    summaries = [
        {"summary_id": summary_id, "original_ids": ids, "summary": summary_content}
    ]

    return CompressionResult(
        original_count=total,
        new_count=new_count,
        compressed=len(ids),
        summaries=summaries,
    )


def _generate_summary(contents: list[str]) -> str:
    """Generate a simple summary from a list of memory contents.

    In production, this would call an LLM. For the MVP, we concatenate
    truncated versions with timestamps.
    """
    if not contents:
        return "(no memories)"

    # Simple concatenation-based summarisation
    truncated = []
    for content in contents:
        # Take first ~80 chars of each memory
        if len(content) > 80:
            truncated.append(content[:77] + "...")
        else:
            truncated.append(content)

    combined = " | ".join(truncated)
    if len(combined) > 500:
        combined = combined[:497] + "..."

    return f"[Compressed batch of {len(contents)} memories] {combined}"
