import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient

from mlflow.assistant.config import AssistantConfig, ProjectConfig
from mlflow.assistant.config import ProviderConfig as AssistantProviderConfig
from mlflow.assistant.providers import OllamaProvider
from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from mlflow.assistant.types import Event, Message, ToolUseBlock
from mlflow.server.assistant.api import (
    PermissionDecision,
    _INVALID_REMOTE_ACCESS_MODES_WARNED,
    _AssistantAPIRoute,
    _enforce_remote_access,
    _is_localhost,
    _provider_allows_remote_access,
    assistant_router,
)
from mlflow.server.assistant.session import SESSION_DIR, SessionManager, save_process_pid
from mlflow.utils.os import is_windows


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

    def resolve_skills_path(self, base_directory: Path) -> Path:
        return base_directory / ".mock" / "skills"

    def list_models(self, base_url: str | None = None, api_key: str | None = None) -> list[str]:
        raise NotImplementedError

    async def astream(
        self,
        prompt: str,
        tracking_uri: str,
        session_id: str | None = None,
        mlflow_session_id: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ):
        yield Event.from_message(message=Message(role="user", content="Hello from mock"))
        yield Event.from_result(result="complete", session_id="mock-session-123")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect config to tmp_path to avoid modifying real user config."""
    import mlflow.assistant.config as config_module

    config_home = tmp_path / ".mlflow" / "assistant"
    config_path = config_home / "config.json"

    monkeypatch.setattr(config_module, "MLFLOW_ASSISTANT_HOME", config_home)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    return config_home


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
    """Create test client with mock provider, treated as a localhost caller."""
    app = FastAPI()
    app.include_router(assistant_router)

    mock_provider = MockProvider()
    with (
        patch("mlflow.server.assistant.api.list_providers", return_value=[mock_provider]),
        patch("mlflow.server.assistant.api._get_selected_provider", return_value=mock_provider),
        patch("mlflow.server.assistant.api._is_localhost", return_value=True),
    ):
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

    class CLINotInstalledProvider(MockProvider):
        def check_connection(self, echo=None):
            raise CLINotInstalledError("CLI not installed")

    provider = CLINotInstalledProvider()
    with (
        patch("mlflow.server.assistant.api.list_providers", return_value=[provider]),
        patch("mlflow.server.assistant.api._is_localhost", return_value=True),
    ):
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/mlflow/assistant/providers/mock_provider/health")
        assert response.status_code == 412
        assert "CLI not installed" in response.json()["detail"]


def test_health_check_returns_401_when_not_authenticated():
    app = FastAPI()
    app.include_router(assistant_router)

    class NotAuthenticatedProvider(MockProvider):
        def check_connection(self, echo=None):
            raise NotAuthenticatedError("Not authenticated")

    provider = NotAuthenticatedProvider()
    with (
        patch("mlflow.server.assistant.api.list_providers", return_value=[provider]),
        patch("mlflow.server.assistant.api._is_localhost", return_value=True),
    ):
        client = TestClient(app)
        response = client.get("/ajax-api/3.0/mlflow/assistant/providers/mock_provider/health")
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]


def test_get_config_returns_empty_config(client):
    response = client.get("/ajax-api/3.0/mlflow/assistant/config")
    assert response.status_code == 200
    data = response.json()
    assert data["providers"] == {}
    assert data["projects"] == {}


def test_get_config_returns_existing_config(client, tmp_path):
    # Set up existing config by saving it first
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    config = AssistantConfig(
        providers={"claude_code": AssistantProviderConfig(model="default", selected=True)},
        projects={"exp-123": ProjectConfig(type="local", location=str(project_dir))},
    )
    config.save()

    response = client.get("/ajax-api/3.0/mlflow/assistant/config")
    assert response.status_code == 200
    data = response.json()
    assert data["providers"]["claude_code"]["model"] == "default"
    assert data["providers"]["claude_code"]["selected"] is True
    assert data["projects"]["exp-123"]["location"] == str(project_dir)


def test_get_config_redacts_api_key_for_remote_clients(client, tmp_path):
    config = AssistantConfig(
        providers={
            "claude_code": AssistantProviderConfig(
                model="default", selected=True, api_key="sk-secret"
            )
        },
    )
    config.save()

    with patch("mlflow.server.assistant.api._is_localhost", return_value=False):
        response = client.get("/ajax-api/3.0/mlflow/assistant/config")

    assert response.status_code == 200
    assert "api_key" not in response.json()["providers"]["claude_code"]


def test_get_config_keeps_api_key_for_localhost(client):
    config = AssistantConfig(
        providers={
            "claude_code": AssistantProviderConfig(
                model="default", selected=True, api_key="sk-secret"
            )
        },
    )
    config.save()

    response = client.get("/ajax-api/3.0/mlflow/assistant/config")

    assert response.status_code == 200
    assert response.json()["providers"]["claude_code"]["api_key"] == "sk-secret"


def test_get_config_loads_config_once(client):
    with patch.object(AssistantConfig, "load", wraps=AssistantConfig.load) as mock_load:
        response = client.get("/ajax-api/3.0/mlflow/assistant/config")

    assert response.status_code == 200
    assert mock_load.call_count == 1


@pytest.mark.parametrize(
    ("mode", "requires_local_execution", "expected"),
    [
        ("off", True, False),
        ("api-only", True, False),
        ("api-only", False, True),
        ("all", True, True),
    ],
)
def test_get_config_remote_chat_allowed(
    client, monkeypatch, mode, requires_local_execution, expected
):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", mode)
    with patch("mlflow.server.assistant.api._get_selected_provider") as mock_get_selected_provider:
        mock_get_selected_provider.return_value.requires_local_execution = requires_local_execution
        response = client.get("/ajax-api/3.0/mlflow/assistant/config")

    assert response.status_code == 200
    assert response.json()["remote_chat_allowed"] is expected


def test_message_blocked_for_remote_client_when_remote_access_off(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "off")
    app = FastAPI()
    app.include_router(assistant_router)

    mock_provider = MockProvider()
    with (
        patch("mlflow.server.assistant.api.list_providers", return_value=[mock_provider]),
        patch("mlflow.server.assistant.api._get_selected_provider", return_value=mock_provider),
        patch("mlflow.server.assistant.api._is_localhost", return_value=False),
    ):
        client = TestClient(app)
        response = client.post(
            "/ajax-api/3.0/mlflow/assistant/message",
            json={"message": "Hello"},
        )

    assert response.status_code == 403
    assert "same host" in response.json()["detail"]


def test_assistant_route_enforces_remote_access_by_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "off")

    app = FastAPI()
    router = APIRouter(prefix="/assistant", route_class=_AssistantAPIRoute)

    @router.get("/unguarded")
    async def unguarded_route():
        return {"status": "ok"}

    app.include_router(router)

    mock_provider = MockProvider()
    with (
        patch("mlflow.server.assistant.api._get_selected_provider", return_value=mock_provider),
        patch("mlflow.server.assistant.api._is_localhost", return_value=False),
    ):
        response = TestClient(app).get("/assistant/unguarded")

    assert response.status_code == 403
    assert "same host" in response.json()["detail"]


def test_message_allowed_for_remote_client_when_provider_allows_remote_access(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "api-only")
    app = FastAPI()
    app.include_router(assistant_router)

    class RemoteAllowedProvider(MockProvider):
        @property
        def requires_local_execution(self) -> bool:
            return False

    mock_provider = RemoteAllowedProvider()
    with (
        patch("mlflow.server.assistant.api.list_providers", return_value=[mock_provider]),
        patch("mlflow.server.assistant.api._get_selected_provider", return_value=mock_provider),
        patch("mlflow.server.assistant.api._is_localhost", return_value=False),
    ):
        client = TestClient(app)
        response = client.post(
            "/ajax-api/3.0/mlflow/assistant/message",
            json={"message": "Hello"},
        )

    assert response.status_code == 200


def test_update_config_sets_provider(client):
    response = client.put(
        "/ajax-api/3.0/mlflow/assistant/config",
        json={"providers": {"claude_code": {"model": "opus", "selected": True}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["providers"]["claude_code"]["selected"] is True


def test_update_config_sets_project(client, tmp_path):
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()

    response = client.put(
        "/ajax-api/3.0/mlflow/assistant/config",
        json={"projects": {"exp-456": {"type": "local", "location": str(project_dir)}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["projects"]["exp-456"]["location"] == str(project_dir)


def test_update_config_expand_user_home(client, tmp_path):
    # Create a directory under a "fake home" structure to test ~ expansion
    fake_home = tmp_path / "home" / "user"
    project_dir = fake_home / "my_project"
    project_dir.mkdir(parents=True)

    with patch("mlflow.server.assistant.api.Path.expanduser") as mock_expanduser:
        # Make expanduser return our tmp_path directory
        mock_expanduser.return_value = project_dir

        response = client.put(
            "/ajax-api/3.0/mlflow/assistant/config",
            json={"projects": {"exp-456": {"type": "local", "location": "~/my_project"}}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["projects"]["exp-456"]["location"] == str(project_dir)


def test_is_localhost_allows_ipv4():
    mock_request = MagicMock()
    mock_request.client.host = "127.0.0.1"
    assert _is_localhost(mock_request)


def test_is_localhost_allows_ipv6():
    mock_request = MagicMock()
    mock_request.client.host = "::1"
    assert _is_localhost(mock_request)


def test_is_localhost_blocks_external_ip():
    mock_request = MagicMock()
    mock_request.client.host = "192.168.1.100"
    assert not _is_localhost(mock_request)


def test_is_localhost_blocks_external_hostname():
    mock_request = MagicMock()
    mock_request.client.host = "external.example.com"
    assert not _is_localhost(mock_request)


def test_is_localhost_blocks_when_no_client():
    mock_request = MagicMock()
    mock_request.client = None
    assert not _is_localhost(mock_request)


@pytest.mark.parametrize(
    ("mode", "requires_local_execution", "expected"),
    [
        ("off", True, False),
        ("off", False, False),
        ("api-only", True, False),
        ("api-only", False, True),
        ("all", True, True),
        ("all", False, True),
    ],
)
def test_provider_allows_remote_access(mode, requires_local_execution, expected, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", mode)
    provider = MagicMock()
    provider.requires_local_execution = requires_local_execution
    assert _provider_allows_remote_access(provider) is expected


def test_provider_allows_remote_access_no_provider_selected(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "api-only")
    assert _provider_allows_remote_access(None) is False


def test_invalid_remote_access_mode_falls_back_to_off(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "bogus")
    provider = MagicMock()
    provider.requires_local_execution = False
    assert _provider_allows_remote_access(provider) is False


def test_invalid_remote_access_mode_logs_warning_once(monkeypatch, caplog):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "bogus")
    _INVALID_REMOTE_ACCESS_MODES_WARNED.clear()
    provider = MagicMock()
    provider.requires_local_execution = False
    assistant_logger = logging.getLogger("mlflow.server.assistant.api")
    assistant_logger.addHandler(caplog.handler)

    try:
        with caplog.at_level("WARNING", logger="mlflow.server.assistant.api"):
            assert _provider_allows_remote_access(provider) is False
            assert _provider_allows_remote_access(provider) is False
    finally:
        assistant_logger.removeHandler(caplog.handler)

    assert sum("Invalid value 'bogus'" in record.message for record in caplog.records) == 1


def test_enforce_remote_access_allows_localhost_regardless_of_mode(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "off")
    mock_request = MagicMock()
    mock_request.client.host = "127.0.0.1"
    _enforce_remote_access(mock_request, None)  # should not raise


def test_enforce_remote_access_blocks_remote_when_off(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "off")
    mock_request = MagicMock()
    mock_request.client.host = "192.168.1.100"
    with pytest.raises(HTTPException, match="same host"):
        _enforce_remote_access(mock_request, None)


def test_validate_session_id_accepts_valid_uuid():
    valid_uuid = "f5f28c66-5ec6-46a1-9a2e-ca55fb64bf47"
    SessionManager.validate_session_id(valid_uuid)  # Should not raise


def test_validate_session_id_rejects_invalid_format():
    with pytest.raises(ValueError, match="Invalid session ID format"):
        SessionManager.validate_session_id("invalid-session-id")


def test_validate_session_id_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid session ID format"):
        SessionManager.validate_session_id("../../../etc/passwd")


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):  # ValueError is raised on Windows
        return False


def test_patch_session_cancel_with_process(client):
    r = client.post("/ajax-api/3.0/mlflow/assistant/message", json={"message": "Hi"})
    session_id = r.json()["session_id"]

    # Start a real subprocess and register it with the session
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"]) as proc:
        save_process_pid(session_id, proc.pid)

        assert _is_process_running(proc.pid)

        response = client.patch(
            f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}",
            json={"status": "cancelled"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "terminated" in data["message"]

        # Wait for the process to actually terminate
        proc.wait(timeout=5)
        assert proc.returncode is not None
        # On non-Windows, verify the process is no longer running via PID check.
        # Skip on Windows because PIDs are reused more aggressively.
        if not is_windows():
            assert not _is_process_running(proc.pid)


class _DeferredProvider(MockProvider):
    """Pauses at the prompt on the first turn; resumes from the decision in context."""

    async def astream(
        self,
        prompt,
        tracking_uri,
        session_id=None,
        mlflow_session_id=None,
        cwd=None,
        context=None,
    ):
        decisions = (context or {}).get("tool_decisions") or {}
        if not decisions:
            # First turn: surface the tool, emit the prompt, and end the turn.
            yield Event.from_message(
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="t1", name="Bash", input={"command": "echo hi"})],
                )
            )
            yield Event.from_permission_request("t1", "Bash", {"command": "echo hi"})
            yield Event.from_result(result=None, session_id="prov-paused")
            return
        # Resume: apply the delivered decision.
        allowed = decisions.get("t1") == "allow"
        yield Event.from_message(Message(role="assistant", content="ran" if allowed else "denied"))
        yield Event.from_result(result=None, session_id="prov-done")


@pytest.mark.parametrize(("decision", "expected_text"), [("allow", "ran"), ("deny", "denied")])
@pytest.mark.asyncio
async def test_stream_pauses_then_resumes(decision, expected_text):
    """The turn ENDS at the permission prompt (no hang, no cross-request state);
    a resume request then drives a fresh stream to completion.
    """
    from mlflow.server.assistant.api import resolve_permission, stream_response

    session_id = "f5f28c66-5ec6-46a1-9a2e-ca55fb64bf47"
    session = SessionManager.create()
    session.set_pending_message(role="user", content="hi")
    SessionManager.save(session_id, session)

    mock_request = MagicMock()
    mock_request.base_url = "http://localhost:5000/"
    provider = _DeferredProvider()

    # First turn: the stream completes immediately at the prompt (no await).
    with patch("mlflow.server.assistant.api._get_selected_provider", return_value=provider):
        response = await stream_response(mock_request, session_id)
        first = "".join([c async for c in response.body_iterator])
    assert "permission_request" in first
    assert "event: done" in first

    # Deliver the decision, then a fresh stream resumes to completion.
    res = await resolve_permission(
        session_id, PermissionDecision(request_id="t1", decision=decision)
    )
    assert res.session_id == session_id

    with patch("mlflow.server.assistant.api._get_selected_provider", return_value=provider):
        response2 = await stream_response(mock_request, session_id)
        second = "".join([c async for c in response2.body_iterator])
    assert expected_text in second
    assert "event: done" in second


class _CaptureProvider(MockProvider):
    """Records the prompt and context astream is called with, then completes."""

    def __init__(self):
        self.captured: dict[str, Any] = {}

    async def astream(
        self,
        prompt,
        tracking_uri,
        session_id=None,
        mlflow_session_id=None,
        cwd=None,
        context=None,
    ):
        self.captured = {"prompt": prompt, "context": context or {}}
        yield Event.from_result(result=None, session_id="prov-done")


@pytest.mark.asyncio
async def test_stream_prefers_new_message_over_stale_tool_decision():
    """A pending message and a stale decision can coexist if a resume stream never
    consumed the decision and the user typed again. The new message must win: the
    provider sees the prompt and NOT the stale tool_decisions (which would otherwise
    resume the abandoned turn and silently drop the message).
    """
    from mlflow.server.assistant.api import stream_response

    session_id = "f5f28c66-5ec6-46a1-9a2e-ca55fb64bf47"
    session = SessionManager.create()
    session.set_pending_message(role="user", content="what is 2+2")
    session.pending_tool_decisions = {"t1": "allow"}
    SessionManager.save(session_id, session)

    mock_request = MagicMock()
    mock_request.base_url = "http://localhost:5000/"
    provider = _CaptureProvider()

    with patch("mlflow.server.assistant.api._get_selected_provider", return_value=provider):
        response = await stream_response(mock_request, session_id)
        _ = "".join([c async for c in response.body_iterator])

    assert provider.captured["prompt"] == "what is 2+2"
    assert "tool_decisions" not in provider.captured["context"]


@pytest.mark.asyncio
async def test_stream_forwards_tool_decision_when_no_pending_message():
    """A genuine resume (decision delivered, no new message) still forwards the
    tool_decisions so the provider can continue the paused turn.
    """
    from mlflow.server.assistant.api import stream_response

    session_id = "f5f28c66-5ec6-46a1-9a2e-ca55fb64bf47"
    session = SessionManager.create()
    session.pending_tool_decisions = {"t1": "allow"}
    SessionManager.save(session_id, session)

    mock_request = MagicMock()
    mock_request.base_url = "http://localhost:5000/"
    provider = _CaptureProvider()

    with patch("mlflow.server.assistant.api._get_selected_provider", return_value=provider):
        response = await stream_response(mock_request, session_id)
        _ = "".join([c async for c in response.body_iterator])

    assert provider.captured["prompt"] == ""
    assert provider.captured["context"]["tool_decisions"] == {"t1": "allow"}


def test_resolve_permission_rejects_invalid_session(client):
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/sessions/not-a-uuid/permission",
        json={"request_id": "t1", "decision": "allow"},
    )
    assert response.status_code == 400


def test_resolve_permission_returns_404_for_unknown_session(client):
    # Well-formed UUID, but no such session exists.
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/sessions/f5f28c66-5ec6-46a1-9a2e-ca55fb64bf47/permission",
        json={"request_id": "t1", "decision": "allow"},
    )
    assert response.status_code == 404


def test_install_skills_success(client):
    with patch(
        "mlflow.server.assistant.api.install_skills", return_value=["skill1", "skill2"]
    ) as mock_install:
        response = client.post(
            "/ajax-api/3.0/mlflow/assistant/skills/install",
            json={"type": "custom", "custom_path": "/tmp/test-skills"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["installed_skills"] == ["skill1", "skill2"]
        expected_path = os.path.join(os.sep, "tmp", "test-skills")
        assert data["skills_directory"] == expected_path
        mock_install.assert_called_once_with(Path(expected_path))


def test_install_skills_skips_when_already_installed(client):
    with (
        patch("mlflow.server.assistant.api.Path.exists", return_value=True),
        patch(
            "mlflow.server.assistant.api.list_installed_skills",
            return_value=["existing_skill"],
        ) as mock_list,
        patch("mlflow.server.assistant.api.install_skills") as mock_install,
    ):
        response = client.post(
            "/ajax-api/3.0/mlflow/assistant/skills/install",
            json={"type": "custom", "custom_path": "/tmp/test-skills"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["installed_skills"] == ["existing_skill"]
        mock_install.assert_not_called()
        mock_list.assert_called_once()


def test_update_config_partial_update_preserves_selected_provider(client):
    ollama = OllamaProvider.OLLAMA_PROVIDER_NAME
    # Pre-populate config: claude_code selected, ollama exists but not selected
    config = AssistantConfig(
        providers={
            "claude_code": AssistantProviderConfig(model="opus", selected=True),
            ollama: AssistantProviderConfig(
                model="llama3", selected=False, base_url="http://localhost:11434"
            ),
        }
    )
    config.save()

    # Partially update ollama base_url without a selected flag
    response = client.put(
        "/ajax-api/3.0/mlflow/assistant/config",
        json={"providers": {ollama: {"base_url": "http://localhost:12345"}}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["providers"]["claude_code"]["selected"] is True
    assert data["providers"][ollama]["selected"] is False
    assert data["providers"][ollama]["base_url"] == "http://localhost:12345"


def test_list_ollama_models_returns_model_list(client):
    mock_provider = MockProvider()
    mock_provider.list_models = MagicMock(return_value=["llama3"])

    with patch("mlflow.server.assistant.api.list_providers", return_value=[mock_provider]):
        response = client.get(
            "/ajax-api/3.0/mlflow/assistant/providers/mock_provider/models",
            params={"base_url": "http://localhost:11434"},
        )

    assert response.status_code == 200
    assert response.json() == {"models": ["llama3"]}
    mock_provider.list_models.assert_called_once_with("http://localhost:11434", None)


def test_list_models_reads_api_key_from_header_not_query(client):
    """api_key must travel as the X-API-Key header so it stays out of access
    logs, browser history, and referer headers. This test pins that
    contract and verifies the value reaches the provider unchanged.
    """
    mock_provider = MockProvider()
    mock_provider.list_models = MagicMock(return_value=["llama3"])

    with patch("mlflow.server.assistant.api.list_providers", return_value=[mock_provider]):
        response = client.get(
            "/ajax-api/3.0/mlflow/assistant/providers/mock_provider/models",
            params={"base_url": "http://localhost:11434"},
            headers={"X-API-Key": "sk-test-secret"},
        )

    assert response.status_code == 200
    assert response.json() == {"models": ["llama3"]}
    mock_provider.list_models.assert_called_once_with("http://localhost:11434", "sk-test-secret")


def test_list_models_ignores_api_key_query_param(client):
    """Defense in depth: even if a caller passes api_key as a query param,
    the endpoint must not forward it to the provider — that would re-enable
    the access-log leak the header migration was meant to prevent.
    """
    mock_provider = MockProvider()
    mock_provider.list_models = MagicMock(return_value=["llama3"])

    with patch("mlflow.server.assistant.api.list_providers", return_value=[mock_provider]):
        response = client.get(
            "/ajax-api/3.0/mlflow/assistant/providers/mock_provider/models",
            params={"base_url": "http://localhost:11434", "api_key": "sk-leaked"},
        )

    assert response.status_code == 200
    mock_provider.list_models.assert_called_once_with("http://localhost:11434", None)


def test_list_ollama_models_returns_412_when_not_installed(client):
    class MissingDependencyProvider(MockProvider):
        def list_models(self, base_url: str | None = None, api_key: str | None = None) -> list[str]:
            raise CLINotInstalledError("ollama package missing")

    with patch(
        "mlflow.server.assistant.api.list_providers",
        return_value=[MissingDependencyProvider()],
    ):
        response = client.get("/ajax-api/3.0/mlflow/assistant/providers/mock_provider/models")

    assert response.status_code == 412
    assert "ollama" in response.json()["detail"].lower()


def test_list_ollama_models_returns_503_on_connection_failure(client):
    class UnreachableProvider(MockProvider):
        def list_models(self, base_url: str | None = None, api_key: str | None = None) -> list[str]:
            raise ProviderNotConfiguredError("Cannot connect to Ollama server")

    with patch("mlflow.server.assistant.api.list_providers", return_value=[UnreachableProvider()]):
        response = client.get(
            "/ajax-api/3.0/mlflow/assistant/providers/mock_provider/models",
            params={"base_url": "http://localhost:11434"},
        )

    assert response.status_code == 503
    assert "Cannot connect" in response.json()["detail"]


def test_list_provider_models_returns_404_for_unsupported_provider(client):
    class UnsupportedProvider(MockProvider):
        @property
        def name(self) -> str:
            return "unsupported_provider"

    with patch(
        "mlflow.server.assistant.api.list_providers",
        return_value=[UnsupportedProvider()],
    ):
        response = client.get(
            "/ajax-api/3.0/mlflow/assistant/providers/unsupported_provider/models"
        )

    assert response.status_code == 404
    assert "not supported" in response.json()["detail"]
