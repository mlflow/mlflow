import shutil
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mlflow.assistant.config import AssistantConfig, ProjectConfig
from mlflow.assistant.config import ProviderConfig as AssistantProviderConfig
from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    ProviderConfig,
)
from mlflow.assistant.types import Event, Message
from mlflow.server.assistant.api import _require_localhost, assistant_router
from mlflow.server.assistant.session import SESSION_DIR, SessionManager


class MockProvider(AssistantProvider):
    """Mock provider for testing."""

    @property
    def name(self) -> str:
        return "mock_provider"

    @property
    def display_name(self) -> str:
        return "Mock Provider"

    @property
    def description(self) -> str:
        return "Mock provider for testing"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".mlflow" / "assistant" / "mock-config.json"

    def is_available(self) -> bool:
        return True

    def load_config(self) -> ProviderConfig:
        return ProviderConfig()

    def check_connection(self, echo=print) -> None:
        pass

    def resolve_skills_path(
        self,
        skills_type: Literal["global", "project", "custom"],
        custom_path: str | None = None,
        project_path: Path | None = None,
    ) -> Path:
        match skills_type:
            case "global":
                return Path.home() / ".mock" / "skills"
            case "project":
                if not project_path:
                    raise ValueError("project_path required for 'project' type")
                return project_path / ".mock" / "skills"
            case "custom":
                if not custom_path:
                    raise ValueError("custom_path required for 'custom' type")
                return Path(custom_path).expanduser()

    async def astream(
        self,
        prompt: str,
        session_id: str | None = None,
        cwd: Path | None = None,
        context: dict | None = None,
    ):
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


def test_health_check_returns_ok_when_healthy(client):
    response = client.get("/ajax-api/3.0/mlflow/assistant/providers/mock_provider/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_check_returns_404_for_unknown_provider(client):
    response = client.get("/ajax-api/3.0/mlflow/assistant/providers/unknown_provider/health")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_health_check_returns_412_when_cli_not_installed():
    app = FastAPI()
    app.include_router(assistant_router)

    async def mock_require_localhost():
        pass

    app.dependency_overrides[_require_localhost] = mock_require_localhost

    class CLINotInstalledProvider(MockProvider):
        def check_connection(self, echo=None):
            raise CLINotInstalledError("CLI not installed")

    with patch("mlflow.server.assistant.api._provider", CLINotInstalledProvider()):
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/mlflow/assistant/providers/mock_provider/health")
        assert response.status_code == 412
        assert "CLI not installed" in response.json()["detail"]


def test_health_check_returns_401_when_not_authenticated():
    app = FastAPI()
    app.include_router(assistant_router)

    async def mock_require_localhost():
        pass

    app.dependency_overrides[_require_localhost] = mock_require_localhost

    class NotAuthenticatedProvider(MockProvider):
        def check_connection(self, echo=None):
            raise NotAuthenticatedError("Not authenticated")

    with patch("mlflow.server.assistant.api._provider", NotAuthenticatedProvider()):
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/mlflow/assistant/providers/mock_provider/health")
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]


def test_get_config_returns_empty_config(client):
    with patch("mlflow.server.assistant.api.AssistantConfig.load") as mock_load:
        mock_load.return_value = AssistantConfig()
        response = client.get("/ajax-api/3.0/mlflow/assistant/config")
        assert response.status_code == 200
        data = response.json()
        assert data["providers"] == {}
        assert data["projects"] == {}


def test_get_config_returns_existing_config(client):
    with patch("mlflow.server.assistant.api.AssistantConfig.load") as mock_load:
        mock_config = AssistantConfig(
            providers={"claude_code": AssistantProviderConfig(model="default", selected=True)},
            projects={"exp-123": ProjectConfig(type="local", location="/path/to/project")},
        )
        mock_load.return_value = mock_config
        response = client.get("/ajax-api/3.0/mlflow/assistant/config")
        assert response.status_code == 200
        data = response.json()
        assert data["providers"]["claude_code"]["model"] == "default"
        assert data["providers"]["claude_code"]["selected"] is True
        assert data["projects"]["exp-123"]["location"] == "/path/to/project"


def test_update_config_sets_provider(client):
    with patch("mlflow.server.assistant.api.AssistantConfig.load") as mock_load:
        mock_config = AssistantConfig()
        mock_load.return_value = mock_config

        response = client.put(
            "/ajax-api/3.0/mlflow/assistant/config",
            json={"providers": {"claude_code": {"model": "opus", "selected": True}}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["providers"]["claude_code"]["selected"] is True


def test_update_config_sets_project(client):
    with patch("mlflow.server.assistant.api.AssistantConfig.load") as mock_load:
        mock_config = AssistantConfig()
        mock_load.return_value = mock_config

        response = client.put(
            "/ajax-api/3.0/mlflow/assistant/config",
            json={"projects": {"exp-456": {"type": "local", "location": "/my/project"}}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["projects"]["exp-456"]["location"] == "/my/project"


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


def test_install_skills_success(client):
    with (
        patch("mlflow.server.assistant.api.AssistantConfig.load") as mock_load,
        patch(
            "mlflow.server.assistant.api.install_skills", return_value=["skill1", "skill2"]
        ) as mock_install,
    ):
        mock_config = AssistantConfig()
        mock_load.return_value = mock_config

        response = client.post(
            "/ajax-api/3.0/mlflow/assistant/skills/install",
            json={"type": "custom", "custom_path": "/tmp/test-skills"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["installed_skills"] == ["skill1", "skill2"]
        assert data["skills_directory"] == "/tmp/test-skills"
        mock_install.assert_called_once_with(Path("/tmp/test-skills"))


def test_install_skills_skips_when_already_installed(client):
    with (
        patch("mlflow.server.assistant.api.AssistantConfig.load") as mock_load,
        patch("mlflow.server.assistant.api.Path.exists", return_value=True),
        patch(
            "mlflow.server.assistant.api.list_installed_skills",
            return_value=["existing_skill"],
        ) as mock_list,
        patch("mlflow.server.assistant.api.install_skills") as mock_install,
    ):
        mock_config = AssistantConfig()
        mock_load.return_value = mock_config

        response = client.post(
            "/ajax-api/3.0/mlflow/assistant/skills/install",
            json={"type": "custom", "custom_path": "/tmp/test-skills"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["installed_skills"] == ["existing_skill"]
        mock_install.assert_not_called()
        mock_list.assert_called_once()
