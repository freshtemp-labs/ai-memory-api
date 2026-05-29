"""Middleware for API key authentication and rate limiting."""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.auth import validate_api_key
from app.rate_limit import check_rate_limit


# Public endpoints that don't need authentication
PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that extracts and validates API keys from requests.

    API keys can be provided in either:
    - Authorization: Bearer <key> header
    - X-API-Key: <key> header
    """

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths and OPTIONS
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        # Skip if request already has user info (set by dependency)
        if hasattr(request.state, "user_id"):
            return await call_next(request)

        # Extract API key
        api_key = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        if not api_key:
            api_key = request.headers.get("X-API-Key", "")

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing API key. Use Authorization: Bearer <key> or X-API-Key header."},
            )

        # Validate
        user = validate_api_key(api_key)
        if user is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"},
            )

        # Rate limit check
        allowed, message = check_rate_limit(user["user_id"], user["tier"])
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": message},
            )

        # Attach user info to request state
        request.state.user_id = user["user_id"]
        request.state.tier = user["tier"]

        return await call_next(request)
