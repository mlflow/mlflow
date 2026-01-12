import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mlflow.assistant.providers.base import AssistantProvider, ProviderConfig
from mlflow.assistant.types import Event, Message
from mlflow.server.assistant.api import _require_localhost, assistant_router
from mlflow.server.assistant.session import SESSION_DIR, SessionManager


class MockProvider(AssistantProvider):
    """Mock provider for testing."""

    @property
    def name(self) -> str:
        return "mock_provider"

    def is_available(self) -> bool:
        return True

    def load_config(self) -> ProviderConfig:
        return ProviderConfig()

    async def astream(self, prompt: str, session_id: str | None = None, cwd: Path | None = None):
        yield Event.from_message(message=Message(role="user", content="Hello from mock"))
        yield Event.from_result(result="complete", session_id="mock-session-123")


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear session storage before each test."""
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
        },
    )

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
    response = client.get("/ajax-api/3.0/mlflow/assistant/sessions/invalid-session-id/stream")
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


def test_stream_bad_request_when_no_pending_message(client):
    # Create session and consume the pending message
    r = client.post("/ajax-api/3.0/mlflow/assistant/message", json={"message": "Hi"})
    session_id = r.json()["session_id"]
    client.get(f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream")

    # Try to stream again without a new message
    response = client.get(f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream")

    assert response.status_code == 400
    assert "No pending message" in response.json()["detail"]


def test_stream_returns_sse_events(client):
    r = client.post("/ajax-api/3.0/mlflow/assistant/message", json={"message": "Hi"})
    session_id = r.json()["session_id"]

    response = client.get(f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream")

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

    with pytest.raises(HTTPException, match="same host"):
        await _require_localhost(mock_request)


@pytest.mark.asyncio
async def test_localhost_blocks_external_hostname():
    mock_request = MagicMock()
    mock_request.client.host = "external.example.com"

    with pytest.raises(HTTPException, match="same host"):
        await _require_localhost(mock_request)


@pytest.mark.asyncio
async def test_localhost_blocks_when_no_client():
    mock_request = MagicMock()
    mock_request.client = None

    with pytest.raises(HTTPException, match="same host"):
        await _require_localhost(mock_request)


def test_validate_session_id_accepts_valid_uuid():
    valid_uuid = "f5f28c66-5ec6-46a1-9a2e-ca55fb64bf47"
    SessionManager.validate_session_id(valid_uuid)  # Should not raise


def test_validate_session_id_rejects_invalid_format():
    with pytest.raises(ValueError, match="Invalid session ID format"):
        SessionManager.validate_session_id("invalid-session-id")


def test_validate_session_id_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid session ID format"):
        SessionManager.validate_session_id("../../../etc/passwd")
