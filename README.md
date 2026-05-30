# AI Memory API

Hosted memory service for AI Agents — store, search, compress, and inject memories.

## Features

- **Vectorized Memory Storage** — OpenAI or local sentence-transformers embeddings
- **Semantic Search** — Find memories by meaning, not just keywords
- **Automatic Compression** — Summarise old memories when over threshold
- **Context Injection** — Format relevant memories for LLM prompt injection
- **Multi-User/Multi-Agent Isolation** — Each agent's memories are isolated
- **Tiered Pricing** — Free (100 req/day), Basic (10K req/month), Pro (unlimited)
- **Docker Deployment** — Single `docker compose up`

## Quick Start

```bash
# Clone
git clone https://github.com/freshtemp-labs/ai-memory-api.git
cd ai-memory-api

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key if using OpenAI embeddings

# Run with Docker
docker compose up -d

# Or run locally
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs at http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/users` | Create user (returns API key) |
| POST | `/agents` | Register an agent |
| POST | `/agents/{id}/memories` | Store a memory |
| GET | `/agents/{id}/memories` | List memories |
| GET | `/agents/{id}/memories/{mid}` | Get a memory | 
| DELETE | `/agents/{id}/memories/{mid}` | Delete a memory |
| POST | `/agents/{id}/search` | Semantic search |
| POST | `/agents/{id}/compress` | Compress old memories |
| POST | `/agents/{id}/context` | Get context for injection |
| GET | `/agents/{id}/stats` | Agent stats |

## Authentication

All endpoints (except /health and /users) require an API key:

```
Authorization: Bearer aim_xxxxxxxxxxxx
# or
X-API-Key: aim_xxxxxxxxxxxx
```

Get your API key by creating a user: `POST /users`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `none` | `openai`, `local`, or `none` |
| `OPENAI_API_KEY` | - | Required if provider=openai |
| `DATABASE_PATH` | `data/ai-memory.db` | SQLite database path |
| `COMPRESSION_THRESHOLD` | `50` | Memories before auto-compression |
| `FREE_TIER_LIMIT` | `100` | Daily request limit for free tier |
| `BASIC_TIER_LIMIT` | `10000` | Monthly request limit for basic tier |
