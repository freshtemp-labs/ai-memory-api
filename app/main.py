"""
AI Memory API — Hosted Memory Service for AI Agents.

Core features:
- Vectorized memory storage (OpenAI or local embeddings)
- Semantic search over memories
- Automatic memory compression
- Context injection for agent prompts
- Multi-user/multi-agent isolation
- Tiered pricing with rate limiting
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Header, Query
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import init_db
from app.models import (
    CreateUserRequest, CreateAgentRequest,
    StoreMemoryRequest, SearchMemoryRequest,
    CompressRequest, ContextInjectRequest,
    MemoryResponse, SearchResult, UserResponse, AgentResponse,
    CompressionResult,
)
from app.auth import create_user, validate_api_key, get_or_create_agent
from app.memory import (
    store_memory, get_memory, list_memories,
    delete_memory, memory_count, embed_and_store,
)
from app.search import search_memories
from app.compression import compress_memories
from app.context import inject_context


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# --- App ---

app = FastAPI(
    title="AI Memory API",
    description="Hosted memory service for AI agents — store, search, and inject memories.",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiter (global)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Auth Dependency ---

async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Extract and validate user from request headers."""
    # Check if middleware already set user
    if hasattr(request.state, "user_id"):
        return {
            "user_id": request.state.user_id,
            "tier": request.state.tier,
        }

    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    elif x_api_key:
        api_key = x_api_key

    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    user = validate_api_key(api_key)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user


# --- Health ---

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai-memory-api", "version": "0.1.0"}


# --- Users ---

@app.post("/users", response_model=UserResponse)
async def create_new_user(req: CreateUserRequest):
    """Create a new user account. Returns an API key."""
    return create_user(req)


# --- Agents ---

@app.post("/agents", response_model=AgentResponse)
async def create_agent(
    req: CreateAgentRequest,
    user: dict = Depends(get_current_user),
):
    """Register a new agent under your account."""
    agent = get_or_create_agent(req.agent_id, req.name, user["user_id"])
    return AgentResponse(
        agent_id=agent["agent_id"],
        name=agent["name"],
        user_id=agent["user_id"],
    )


# --- Memories ---

@app.post("/agents/{agent_id}/memories", response_model=MemoryResponse)
async def store_agent_memory(
    agent_id: str,
    req: StoreMemoryRequest,
    user: dict = Depends(get_current_user),
):
    """Store a new memory for an agent."""
    # Ensure agent exists
    agent = get_or_create_agent(agent_id, agent_id, user["user_id"])

    memory = store_memory(agent_id, user["user_id"], req)

    # Trigger background embedding
    asyncio.create_task(embed_and_store(memory.id, req.content))

    # Check if compression is needed
    count = memory_count(agent_id, user["user_id"])
    if count > settings.compression_threshold:
        asyncio.create_task(
            compress_memories(agent_id, user["user_id"], settings.compression_threshold // 2)
        )

    return memory


@app.get("/agents/{agent_id}/memories", response_model=list[MemoryResponse])
async def list_agent_memories(
    agent_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
):
    """List memories for an agent, newest first."""
    return list_memories(agent_id, user["user_id"], limit, offset)


@app.get("/agents/{agent_id}/memories/{memory_id}", response_model=MemoryResponse)
async def get_agent_memory(
    agent_id: str,
    memory_id: str,
    user: dict = Depends(get_current_user),
):
    """Retrieve a single memory by ID."""
    memory = get_memory(memory_id, user["user_id"])
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.delete("/agents/{agent_id}/memories/{memory_id}")
async def delete_agent_memory(
    agent_id: str,
    memory_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a memory."""
    deleted = delete_memory(memory_id, user["user_id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted"}


# --- Search ---

@app.post("/agents/{agent_id}/search", response_model=list[SearchResult])
async def search_agent_memories(
    agent_id: str,
    req: SearchMemoryRequest,
    user: dict = Depends(get_current_user),
):
    """Semantic search over an agent's memories."""
    return await search_memories(agent_id, user["user_id"], req)


# --- Compression ---

@app.post("/agents/{agent_id}/compress", response_model=CompressionResult)
async def compress_agent_memories(
    agent_id: str,
    req: CompressRequest,
    user: dict = Depends(get_current_user),
):
    """Compress old memories into summaries."""
    return await compress_memories(agent_id, user["user_id"], req.target_count)


# --- Context Injection ---

@app.post("/agents/{agent_id}/context")
async def get_context(
    agent_id: str,
    req: ContextInjectRequest,
    user: dict = Depends(get_current_user),
):
    """Get formatted context from relevant memories for injection into an agent prompt."""
    context = await inject_context(agent_id, user["user_id"], req)
    return {"context": context, "query": req.query}


# --- Stats ---

@app.get("/agents/{agent_id}/stats")
async def agent_stats(
    agent_id: str,
    user: dict = Depends(get_current_user),
):
    """Get stats for an agent."""
    total = memory_count(agent_id, user["user_id"])
    return {
        "agent_id": agent_id,
        "total_memories": total,
        "compression_threshold": settings.compression_threshold,
        "tier": user["tier"],
    }
