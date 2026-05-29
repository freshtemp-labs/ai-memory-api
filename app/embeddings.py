"""Vector embedding generation.

Supports three modes:
- openai: Uses OpenAI text-embedding-3-small
- local: Uses sentence-transformers (all-MiniLM-L6-v2)
- none: Returns zero vectors (no semantic search)
"""

from __future__ import annotations

import time
import httpx
import numpy as np
from app.config import settings

# Lazy-loaded local model
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _local_model = SentenceTransformer(settings.local_embedding_model)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _local_model


async def embed_openai(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via OpenAI API."""
    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-..."):
        raise ValueError("OPENAI_API_KEY not configured")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_embedding_model,
                "input": texts,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]


def embed_local(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using local sentence-transformers model."""
    model = _get_local_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_dummy(texts: list[str]) -> list[list[float]]:
    """Generate dummy zero-vectors (no semantic search)."""
    return [[0.0] * settings.embedding_dim for _ in texts]


async def embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using the configured provider."""
    if isinstance(texts, str):
        texts = [texts]

    provider = settings.embedding_provider
    if provider == "openai":
        return await embed_openai(texts)
    elif provider == "local":
        return embed_local(texts)
    else:
        return embed_dummy(texts)


async def embed_single(text: str) -> list[float]:
    """Generate a single embedding."""
    results = await embed([text])
    return results[0]


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
