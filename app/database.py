"""SQLite database layer for memory storage with vector support.

Uses sqlite-vec for vector similarity search (lightweight, no external deps).
"""

import sqlite3
import json
import struct
import os
from pathlib import Path
from app.config import settings


def _get_db_path() -> str:
    """Resolve the database path, creating parent directories if needed."""
    path = Path(settings.database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_connection() -> sqlite3.Connection:
    """Get a new SQLite connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(_get_db_path())
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            api_key TEXT UNIQUE NOT NULL,
            tier TEXT NOT NULL DEFAULT 'free',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            request_count INTEGER NOT NULL DEFAULT 0,
            request_reset TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL REFERENCES agents(id),
            user_id TEXT NOT NULL REFERENCES users(id),
            content TEXT NOT NULL,
            summary TEXT,
            importance REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            compressed INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_memories_agent
            ON memories(agent_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_user
            ON memories(user_id, created_at);

        CREATE TABLE IF NOT EXISTS memory_vectors (
            memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            vector BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            tier TEXT NOT NULL DEFAULT 'free',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            is_active INTEGER NOT NULL DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_api_keys_user
            ON api_keys(user_id);
    """)
    conn.commit()

    # Enable sqlite-vec if available
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (ImportError, Exception):
        pass  # sqlite-vec not available, fall back to cosine in Python

    conn.close()


def serialize_vector(vec: list[float]) -> bytes:
    """Serialize a float32 list to a compact binary blob."""
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_vector(blob: bytes) -> list[float]:
    """Deserialize binary blob back to float32 list."""
    return list(struct.unpack(f"{len(blob)//4}f", blob))
