"""Rate limiting and pricing tier enforcement."""

import time
from datetime import datetime, timezone
from app.database import get_connection
from app.config import settings


def _utc_now() -> datetime:
    """Return a timezone-aware UTC datetime for consistent comparisons."""
    return datetime.now(timezone.utc)


def check_rate_limit(user_id: str, tier: str) -> tuple[bool, str]:
    """Check if a user is within their rate limit.

    Returns (allowed: bool, message: str).
    """
    conn = get_connection()
    user = conn.execute(
        "SELECT request_count, request_reset FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()

    if user is None:
        conn.close()
        return False, "User not found"

    # Determine limit and reset window
    if tier == "pro":
        conn.close()
        return True, ""  # Unlimited

    now = _utc_now()
    reset_time_str = user["request_reset"]
    if reset_time_str is None:
        reset_time = _utc_now()
    else:
        reset_time = datetime.fromisoformat(reset_time_str)
        if reset_time.tzinfo is None:
            reset_time = reset_time.replace(tzinfo=timezone.utc)

    # Check if we should reset the counter
    reset = False
    if tier == "free":
        # Daily reset
        if (now - reset_time).total_seconds() > 86400:
            reset = True
        limit = settings.free_tier_limit
    elif tier == "basic":
        # Monthly reset
        if (now - reset_time).total_seconds() > 2592000:  # ~30 days
            reset = True
        limit = settings.basic_tier_limit
    else:
        limit = 999999

    if reset:
        conn.execute(
            "UPDATE users SET request_count = 0, request_reset = ? WHERE id = ?",
            (now.strftime("%Y-%m-%d %H:%M:%S+00:00"), user_id),
        )
        conn.commit()

    count = user["request_count"]
    if count >= limit:
        conn.close()
        return False, f"Rate limit exceeded ({limit} requests). Upgrade at https://aimemory.dev/pricing"

    # Increment counter
    conn.execute(
        "UPDATE users SET request_count = request_count + 1 WHERE id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()

    return True, ""


def record_request(user_id: str):
    """Record an API request for billing/analytics."""
    pass  # Future: log to analytics DB
