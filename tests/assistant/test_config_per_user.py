import asyncio
import base64
import sys
import types
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import mlflow.assistant.config as config_module
from mlflow.assistant.config import (
    AssistantConfig,
    PermissionsConfig,
    ProjectConfig,
    ProviderConfig,
    get_config_user,
    set_config_user,
)
from mlflow.assistant.providers.base import clear_config_cache, load_config
from mlflow.server.assistant.api import assistant_router


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".mlflow" / "assistant"
    monkeypatch.setattr(config_module, "MLFLOW_ASSISTANT_HOME", home)
    monkeypatch.setattr(config_module, "CONFIG_PATH", home / "config.json")
    # Start each test with no bound user and a clean provider-config cache (both are process
    # global and would otherwise leak across tests).
    set_config_user(None)
    clear_config_cache()
    yield
    set_config_user(None)
    clear_config_cache()


def _provider(model: str) -> ProviderConfig:
    return ProviderConfig(model=model, selected=True, permissions=PermissionsConfig())


def test_provider_only_save_does_not_rewrite_the_shared_file():
    # Seed the shared global file with a project (a localhost/operator write).
    set_config_user(None)
    AssistantConfig(projects={"e1": ProjectConfig(location="/srv/p")}).save()

    # A user saving only their providers must not rewrite the shared file (projects unchanged), so
    # concurrent remote provider saves never race it. Verify the shared file is not written, and
    # that the shared projects and the user's providers both persist.
    set_config_user("alice")
    cfg = AssistantConfig.load()
    cfg.providers = {"gw": _provider("m1")}

    saved_paths = []
    original_save_file = AssistantConfig._save_file

    def _spy(path, config):
        saved_paths.append(path)
        return original_save_file(path, config)

    with mock.patch.object(AssistantConfig, "_save_file", staticmethod(_spy)):
        cfg.save()
    assert config_module.CONFIG_PATH not in saved_paths

    set_config_user(None)
    assert AssistantConfig.load().projects == {"e1": ProjectConfig(location="/srv/p")}
    set_config_user("alice")
    assert AssistantConfig.load().providers["gw"].model == "m1"


def test_no_user_reads_and_writes_the_global_file():
    set_config_user(None)
    AssistantConfig(providers={"gw": _provider("m1")}).save()

    assert config_module.CONFIG_PATH.exists()
    assert AssistantConfig.load().providers["gw"].model == "m1"


def test_provider_config_is_isolated_per_user():
    set_config_user("alice")
    AssistantConfig(providers={"gw": _provider("alice-model")}).save()
    set_config_user("bob")
    AssistantConfig(providers={"gw": _provider("bob-model")}).save()

    set_config_user("alice")
    assert AssistantConfig.load().providers["gw"].model == "alice-model"
    set_config_user("bob")
    assert AssistantConfig.load().providers["gw"].model == "bob-model"


def test_one_user_does_not_see_another_users_providers():
    set_config_user("alice")
    AssistantConfig(providers={"gw": _provider("m")}).save()

    set_config_user("bob")
    assert "gw" not in AssistantConfig.load().providers


def test_projects_are_shared_server_level():
    # An operator on a no-auth path registers a project; an authenticated user sees the same one.
    set_config_user(None)
    AssistantConfig(projects={"exp1": ProjectConfig(location="/srv/proj")}).save()

    set_config_user("alice")
    assert AssistantConfig.load().projects["exp1"].location == "/srv/proj"


def test_authenticated_save_writes_projects_to_shared_file_only():
    set_config_user("alice")
    AssistantConfig(
        providers={"gw": _provider("m")},
        projects={"exp1": ProjectConfig(location="/srv/p")},
    ).save()

    # Projects land in the shared global file (server-level)...
    set_config_user(None)
    global_config = AssistantConfig.load()
    assert global_config.projects["exp1"].location == "/srv/p"
    # ...and alice's providers do not leak into it.
    assert "gw" not in global_config.providers


def test_load_config_cache_is_user_aware():
    set_config_user("alice")
    AssistantConfig(providers={"gw": _provider("alice-model")}).save()
    set_config_user("bob")
    AssistantConfig(providers={"gw": _provider("bob-model")}).save()

    set_config_user("alice")
    assert load_config("gw").model == "alice-model"
    # A different user must not get alice's cached entry.
    set_config_user("bob")
    assert load_config("gw").model == "bob-model"


def test_user_config_path_is_traversal_safe():
    escaping = config_module._user_config_path("../../../etc/passwd")
    assert config_module.MLFLOW_ASSISTANT_HOME in escaping.parents
    # The per-user directory is the hex digest, so a username with path separators or ".." can't
    # steer the path -- it is neutralized to fixed-charset hex.
    user_dir = escaping.parent.name
    assert len(user_dir) == 64
    assert all(c in "0123456789abcdef" for c in user_dir)


def test_empty_username_uses_the_global_config():
    # An empty username is falsy, so it collapses into the shared/no-auth branch rather than
    # creating a phantom per-user config.
    set_config_user("")
    AssistantConfig(providers={"gw": _provider("m")}).save()

    set_config_user(None)
    assert AssistantConfig.load().providers["gw"].model == "m"


def test_authenticated_save_preserves_existing_global_providers():
    # A provider stored in the shared global file (e.g. by a no-auth operator)...
    set_config_user(None)
    AssistantConfig(providers={"legacy": _provider("legacy-model")}).save()

    # ...must survive an authenticated user's save, which also touches the global file for projects.
    set_config_user("alice")
    AssistantConfig(
        providers={"gw": _provider("alice-model")},
        projects={"exp1": ProjectConfig(location="/srv/p")},
    ).save()

    set_config_user(None)
    global_config = AssistantConfig.load()
    assert "legacy" in global_config.providers
    assert global_config.projects["exp1"].location == "/srv/p"


def test_authenticated_save_aborts_on_unreadable_global_file():
    # A present-but-corrupt global file must abort the save rather than be rewritten empty
    # (which would destroy stored providers/projects).
    config_module.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_module.CONFIG_PATH.write_text("{ not valid json")

    set_config_user("alice")
    with pytest.raises(ValidationError, match="JSON"):
        AssistantConfig(providers={"gw": _provider("alice-model")}).save()

    # The corrupt file was left untouched, not clobbered.
    assert config_module.CONFIG_PATH.read_text() == "{ not valid json"


def test_load_config_reflects_save_only_after_cache_clear():
    set_config_user("alice")
    AssistantConfig(providers={"gw": _provider("v1")}).save()
    assert load_config("gw").model == "v1"

    AssistantConfig(providers={"gw": _provider("v2")}).save()
    assert load_config("gw").model == "v1"  # still the cached value
    clear_config_cache()
    assert load_config("gw").model == "v2"


@pytest.mark.asyncio
async def test_config_user_survives_to_thread():
    # Providers load config inside asyncio.to_thread while streaming; the bound user must be
    # visible there (to_thread copies the context).
    set_config_user("alice")
    assert await asyncio.to_thread(get_config_user) == "alice"


def test_get_config_route_resolves_per_user(monkeypatch):
    # End-to-end: the route layer binds the authenticated user, so GET /config returns that
    # user's own providers -- proving set_config_user is wired through a real request.
    set_config_user("alice")
    AssistantConfig(providers={"gw": _provider("alice-model")}).save()
    set_config_user(None)

    def _authenticate(request):
        header = request.headers.get("Authorization")
        if not header:
            return None
        username = base64.b64decode(header.split(" ", 1)[1]).decode().partition(":")[0]
        return types.SimpleNamespace(username=username, id=username)

    auth_module = types.ModuleType("mlflow.server.auth")
    auth_module.is_auth_enabled = lambda: True
    auth_module.authenticate_fastapi_request_user = _authenticate
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", auth_module)

    def _header(username: str) -> dict[str, str]:
        return {"Authorization": f"Basic {base64.b64encode(f'{username}:pw'.encode()).decode()}"}

    app = FastAPI()
    app.include_router(assistant_router)
    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=True):
        client = TestClient(app)
        alice = client.get("/ajax-api/3.0/mlflow/assistant/config", headers=_header("alice")).json()
        bob = client.get("/ajax-api/3.0/mlflow/assistant/config", headers=_header("bob")).json()

    assert alice["providers"]["gw"]["model"] == "alice-model"
    assert "gw" not in bob["providers"]
