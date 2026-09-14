from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.codex import _codex_sandbox_mode
from mlflow.assistant.providers.tool_executor import (
    execute_tool,
    is_remote_caller,
    restrict_permissions_for_remote,
    set_remote_caller,
    static_permission_error,
)
from mlflow.server.assistant import api
from mlflow.server.assistant.api import (
    _AssistantAPIRoute,
    _remote_access_policy,
    _RemoteAccessPolicy,
)


@pytest.fixture(autouse=True)
def reset_remote_caller():
    set_remote_caller(False)
    yield
    set_remote_caller(False)


def test_restrict_drops_full_access_only_for_remote():
    perms = PermissionsConfig(full_access=True, allow_edit_files=True)

    set_remote_caller(False)
    assert restrict_permissions_for_remote(perms).full_access is True  # local caller unchanged

    set_remote_caller(True)
    clamped = restrict_permissions_for_remote(perms)
    assert clamped.full_access is False
    assert clamped.allow_edit_files is True  # workspace-confined allowances are kept


def test_restrict_is_noop_without_full_access():
    perms = PermissionsConfig(full_access=False)
    set_remote_caller(True)
    assert restrict_permissions_for_remote(perms) is perms


def test_static_permission_error_denies_arbitrary_bash_under_remote_clamp(tmp_path):
    perms = PermissionsConfig(full_access=True)

    set_remote_caller(False)
    # A local full-access caller may run any Bash command.
    assert static_permission_error("Bash", {"command": "echo hi"}, perms, tmp_path) is None

    set_remote_caller(True)
    # After the remote clamp, the same command is denied (not in the restricted allowlist).
    clamped = restrict_permissions_for_remote(perms)
    assert static_permission_error("Bash", {"command": "echo hi"}, clamped, tmp_path) is not None


@pytest.mark.asyncio
async def test_execute_tool_denies_full_access_bash_for_remote_caller():
    # Even with full_access supplied, a remote caller cannot run an arbitrary command; the denial
    # happens before anything executes.
    set_remote_caller(True)
    result, is_error = await execute_tool(
        "Bash", {"command": "echo hi"}, permissions=PermissionsConfig(full_access=True)
    )
    assert is_error
    assert "Permission denied" in result


@pytest.mark.asyncio
async def test_execute_tool_allows_full_access_bash_for_local_caller(tmp_path):
    set_remote_caller(False)
    result, is_error = await execute_tool(
        "Bash",
        {"command": "echo hi"},
        cwd=tmp_path,
        permissions=PermissionsConfig(full_access=True),
    )
    assert not is_error
    assert "hi" in result


def test_codex_sandbox_mode_is_restricted_for_remote():
    set_remote_caller(False)
    assert _codex_sandbox_mode() == "danger-full-access"
    set_remote_caller(True)
    assert _codex_sandbox_mode() == "workspace-write"


def test_provider_remote_access_requires_sandbox(monkeypatch):
    provider = SimpleNamespace(allows_remote_access=True)

    # Remote-capable provider + remote mode on, but no sandbox -> remote access denied, so
    # server-side tools never run on the host for a remote caller.
    monkeypatch.setenv("MLFLOW_ENABLE_REMOTE_ASSISTANT", "true")
    monkeypatch.setattr(api, "assistant_sandbox_enabled", lambda: False)
    assert api._provider_allows_remote_access(provider) is False

    # With the sandbox on, remote access is allowed (tools run isolated in a container).
    monkeypatch.setattr(api, "assistant_sandbox_enabled", lambda: True)
    assert api._provider_allows_remote_access(provider) is True

    # Remote mode itself off -> denied regardless of the sandbox.
    monkeypatch.setenv("MLFLOW_ENABLE_REMOTE_ASSISTANT", "false")
    assert api._provider_allows_remote_access(provider) is False


def test_remote_caller_survives_into_streaming_response():
    # The clamp depends on the _remote_caller var, set in the route handler, surviving into the
    # StreamingResponse body -- which Starlette iterates AFTER the handler returns. Read it there.
    router = APIRouter(route_class=_AssistantAPIRoute)

    @router.get("/_remote_stream_probe")
    @_remote_access_policy(_RemoteAccessPolicy.NONE)
    async def _probe(request: Request):
        async def _body():
            yield f"remote={is_remote_caller()}".encode()

        return StreamingResponse(_body())

    app = FastAPI()
    app.include_router(router)
    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=False):
        response = TestClient(app).get("/_remote_stream_probe")
    assert response.content == b"remote=True"


def test_route_handler_marks_remote_caller():
    router = APIRouter(route_class=_AssistantAPIRoute)

    @router.get("/_remote_probe")
    @_remote_access_policy(_RemoteAccessPolicy.NONE)
    async def _probe(request: Request):
        return {"remote": is_remote_caller()}

    app = FastAPI()
    app.include_router(router)

    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=True):
        assert TestClient(app).get("/_remote_probe").json() == {"remote": False}
    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=False):
        assert TestClient(app).get("/_remote_probe").json() == {"remote": True}
