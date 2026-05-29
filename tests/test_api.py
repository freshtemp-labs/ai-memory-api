"""Tests for AI Memory API."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Reset DB before each test — drop all data for clean isolation."""
    init_db()
    # In tests we clean data so previous runs don't leak
    from app.database import get_connection
    conn = get_connection()
    conn.executescript("""
        DELETE FROM memory_vectors;
        DELETE FROM memories;
        DELETE FROM agents;
        DELETE FROM api_keys;
        DELETE FROM users;
    """)
    conn.commit()
    conn.close()


@pytest.fixture
def user():
    """Create a test user and return its API key."""
    resp = client.post("/users", json={"tier": "free"})
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def auth_header(user):
    """Return Authorization header for the test user."""
    return {"Authorization": f"Bearer {user['api_key']}"}


class TestHealth:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200


class TestUsers:
    def test_create_user_free(self):
        resp = client.post("/users", json={"tier": "free"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"].startswith("aim_")
        assert data["tier"] == "free"
        assert "user_id" in data

    def test_create_user_basic(self):
        resp = client.post("/users", json={"tier": "basic"})
        assert resp.status_code == 200
        assert resp.json()["tier"] == "basic"


class TestAgents:
    def test_create_agent(self, auth_header):
        resp = client.post(
            "/agents",
            json={"agent_id": "test-agent-1", "name": "Test Agent"},
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "test-agent-1"

    def test_create_agent_no_auth(self):
        resp = client.post("/agents", json={"agent_id": "bad"})
        assert resp.status_code == 401


class TestMemories:
    def test_store_and_get(self, auth_header):
        # Store memory
        resp = client.post(
            "/agents/test-agent/memories",
            json={"content": "User prefers dark mode."},
            headers=auth_header,
        )
        assert resp.status_code == 200
        mem = resp.json()
        assert mem["content"] == "User prefers dark mode."
        memory_id = mem["id"]

        # Get memory
        resp = client.get(
            f"/agents/test-agent/memories/{memory_id}",
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "User prefers dark mode."

    def test_list_memories(self, auth_header):
        # Store a few
        for i in range(5):
            client.post(
                "/agents/test-agent/memories",
                json={"content": f"Memory {i}"},
                headers=auth_header,
            )

        resp = client.get(
            "/agents/test-agent/memories",
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 5

    def test_delete_memory(self, auth_header):
        resp = client.post(
            "/agents/test-agent/memories",
            json={"content": "Delete me"},
            headers=auth_header,
        )
        memory_id = resp.json()["id"]

        resp = client.delete(
            f"/agents/test-agent/memories/{memory_id}",
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_memory_not_found(self, auth_header):
        resp = client.get(
            "/agents/test-agent/memories/nonexistent",
            headers=auth_header,
        )
        assert resp.status_code == 404


class TestSearch:
    def test_search_fallback_keyword(self, auth_header):
        # Store memories
        client.post(
            "/agents/test-agent/memories",
            json={"content": "The user loves Python."},
            headers=auth_header,
        )
        client.post(
            "/agents/test-agent/memories",
            json={"content": "The user hates JavaScript."},
            headers=auth_header,
        )

        resp = client.post(
            "/agents/test-agent/search",
            json={"query": "Python", "limit": 5, "min_similarity": 0.1},
            headers=auth_header,
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert any("Python" in r["memory"]["content"] for r in results)


class TestCompression:
    def test_compress_below_threshold(self, auth_header):
        resp = client.post(
            "/agents/test-agent/compress",
            json={"target_count": 30},
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert resp.json()["compressed"] == 0


class TestContext:
    def test_context_injection(self, auth_header):
        client.post(
            "/agents/test-agent/memories",
            json={"content": "User prefers concise responses."},
            headers=auth_header,
        )

        resp = client.post(
            "/agents/test-agent/context",
            json={"query": "What does the user prefer?", "max_tokens": 500},
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert "context" in resp.json()
        assert "concise" in resp.json()["context"].lower()
