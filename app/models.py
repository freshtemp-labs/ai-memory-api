"""Pydantic models for the Memory API."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import uuid


def gen_id() -> str:
    return uuid.uuid4().hex[:12]


# --- Request Models ---

class CreateUserRequest(BaseModel):
    user_id: str = Field(default_factory=gen_id)
    tier: str = "free"  # free, basic, pro


class CreateAgentRequest(BaseModel):
    agent_id: str = Field(default_factory=gen_id)
    name: str
    user_id: str | None = None


class StoreMemoryRequest(BaseModel):
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class SearchMemoryRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class CompressRequest(BaseModel):
    target_count: int = Field(default=30, ge=5, le=100)


class ContextInjectRequest(BaseModel):
    query: str
    max_tokens: int = Field(default=2000, ge=100, le=32000)


# --- Response Models ---

class MemoryResponse(BaseModel):
    id: str
    agent_id: str
    user_id: str
    content: str
    summary: Optional[str] = None
    importance: float
    created_at: str
    updated_at: str
    compressed: bool = False


class SearchResult(BaseModel):
    memory: MemoryResponse
    similarity: float


class UserResponse(BaseModel):
    user_id: str
    api_key: str
    tier: str
    request_count: int


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    user_id: str


class CompressionResult(BaseModel):
    original_count: int
    new_count: int
    compressed: int
    summaries: list[dict]
