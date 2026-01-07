"""Tests for the Assistant API endpoints."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from mlflow.server.assistant.api import assistant_router, SESSION_DIR, _require_localhost
from mlflow.server.assistant.providers.base import AssistantProvider


class MockProvider(AssistantProvider):
    """Mock provider for testing."""

    @property
    def name(self) -> str:
        return "mock_provider"

    def is_available(self) -> bool:
        return True

    def load_config(self) -> dict:
        return {}

    async def run(self, prompt: str, session_id: str | None = None):
        yield {"type": "message", "data": {"text": "Hello from mock"}}
        yield {"type": "done", "data": {"status": "complete", "session_id": "mock-session-123"}}


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear session storage before each test."""
    import shutil

    if SESSION_DIR.exists():
        shutil.rmtree(SESSION_DIR)
    yield
    if SESSION_DIR.exists():
        shutil.rmtree(SESSION_DIR)


@pytest.fixture
def client():
    """Create test client with mock provider and bypassed localhost check."""
    app = FastAPI()
    app.include_router(assistant_router)

    # Override localhost dependency to allow TestClient requests
    async def mock_require_localhost():
        pass

    app.dependency_overrides[_require_localhost] = mock_require_localhost

    with patch("mlflow.server.assistant.api._provider", MockProvider()):
        yield TestClient(app)


def test_message(client):
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/message",
        json={
            "message": "Hello",
            "context": {"trace_id": "tr-123", "experiment_id": "exp-456"},
        })

    assert response.status_code == 200
    data = response.json()
    session_id = data["session_id"]
    assert session_id is not None
    assert data["stream_url"] == f"/ajax-api/3.0/mlflow/assistant/stream/{data['session_id']}"

    # continue the conversation
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/message",
        json={"message": "Second message", "session_id": session_id},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == session_id


def test_stream_not_found_for_invalid_session(client):
    response = client.get("/ajax-api/3.0/mlflow/assistant/stream/invalid-session-id")
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


def test_stream_bad_request_when_no_pending_message(client):
    # Create session and consume the pending message
    r = client.post("/ajax-api/3.0/mlflow/assistant/message", json={"message": "Hi"})
    session_id = r.json()["session_id"]
    client.get(f"/ajax-api/3.0/mlflow/assistant/stream/{session_id}")

    # Try to stream again without a new message
    response = client.get(f"/ajax-api/3.0/mlflow/assistant/stream/{session_id}")

    assert response.status_code == 400
    assert "No pending message" in response.json()["detail"]


def test_stream_returns_sse_events(client):
    r = client.post("/ajax-api/3.0/mlflow/assistant/message", json={"message": "Hi"})
    session_id = r.json()["session_id"]

    response = client.get(f"/ajax-api/3.0/mlflow/assistant/stream/{session_id}")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    content = response.text
    assert "event: message" in content
    assert "event: done" in content
    assert "Hello from mock" in content


def test_status_returns_provider_info(client):
    response = client.get("/ajax-api/3.0/mlflow/assistant/status")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "mock_provider"
    assert data["available"] is True


@pytest.mark.asyncio
async def test_localhost_allows_ipv4():
    mock_request = MagicMock()
    mock_request.client.host = "127.0.0.1"
    await _require_localhost(mock_request)


@pytest.mark.asyncio
async def test_localhost_allows_ipv6():
    mock_request = MagicMock()
    mock_request.client.host = "::1"
    await _require_localhost(mock_request)


@pytest.mark.asyncio
async def test_localhost_blocks_external_ip():
    mock_request = MagicMock()
    mock_request.client.host = "192.168.1.100"

    with pytest.raises(HTTPException) as exc_info:
        await _require_localhost(mock_request)

    assert exc_info.value.status_code == 403
    assert "localhost" in exc_info.value.detail


@pytest.mark.asyncio
async def test_localhost_blocks_when_no_client():
    mock_request = MagicMock()
    mock_request.client = None

    with pytest.raises(HTTPException) as exc_info:
        await _require_localhost(mock_request)

    assert exc_info.value.status_code == 403
