from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.server.fastapi_app import add_mcp_exception_handlers
from mlflow.server.mcp_server_api import (
    _ensure_version_create_parent_access,
    get_mcp_server_api_route_prefixes,
    mcp_server_router,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

PREFIX = "/ajax-api/3.0/mlflow/mcp-servers"


def _server_json(name: str, version: str, **extra) -> dict[str, Any]:
    d = {"name": name, "version": version}
    d.update(extra)
    return d


def _encode_path_param(value: str) -> str:
    return quote(value, safe="")


def _create_version(client, name: str, version: str, status: str = "draft", **server_json_extra):
    initial_status = "active" if status == "deprecated" else status
    response = client.post(
        f"{PREFIX}/{_encode_path_param(name)}/versions",
        json={
            "server_json": _server_json(name, version, **server_json_extra),
            "status": initial_status,
        },
    )
    assert response.status_code == 200, response.text
    if status == "deprecated":
        response = client.patch(
            f"{PREFIX}/{_encode_path_param(name)}/versions/{version}",
            json={"status": "deprecated"},
        )
        assert response.status_code == 200, response.text
    return response


def _create_registry_fastapi_app(route_prefixes=None):
    fastapi_app = FastAPI()
    add_mcp_exception_handlers(fastapi_app)
    if route_prefixes is None:
        route_prefixes = get_mcp_server_api_route_prefixes()
    elif isinstance(route_prefixes, str):
        route_prefixes = (route_prefixes,)
    for route_prefix in route_prefixes:
        fastapi_app.include_router(mcp_server_router, prefix=route_prefix)
    return fastapi_app


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(f"sqlite:///{tmp_path / 'test.db'}", artifact_uri.as_uri())


@pytest.fixture
def client(store):
    with mock.patch(
        "mlflow.server.handlers._get_tracking_store",
        return_value=store,
    ):
        yield TestClient(_create_registry_fastapi_app())


@pytest.fixture(autouse=True)
def mock_icon_url_dns_resolution():
    def _resolve(host, port, *a, **kw):
        if host == "localhost":
            ip = "127.0.0.1"
        elif host == "example.com" or host.endswith(".example.com"):
            ip = "8.8.8.8"
        else:
            ip = host
        return [(None, None, None, None, (ip, 0))]

    with mock.patch("mlflow.utils.validation.socket.getaddrinfo", side_effect=_resolve):
        yield


def test_create_server(client):
    r = client.post(PREFIX, json={"name": "com.example/my-server", "description": "A test server"})
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "com.example/my-server"
    assert data["description"] == "A test server"
    assert data["creation_timestamp"] is not None


def test_create_server_with_icons(client):
    icons = [{"src": "https://example.com/icon.png", "sizes": ["32x32"]}]
    r = client.post(PREFIX, json={"name": "com.example/icon-server", "icons": icons})
    assert r.status_code == 200
    assert r.json()["icons"] == icons


def test_create_server_rejects_too_many_icons(client):
    icons = [{"src": f"https://example.com/icon-{i}.png"} for i in range(101)]
    r = client.post(PREFIX, json={"name": "com.example/too-many-icons", "icons": icons})
    assert r.status_code == 400
    assert "at most 100 items" in r.text


def test_create_server_with_icons_preserves_extra_fields(client):
    icons = [
        {
            "src": "https://example.com/icon.png",
            "sizes": ["32x32"],
            "purpose": "maskable",
        }
    ]
    r = client.post(
        PREFIX,
        json={"name": "com.example/icon-extra-server", "icons": icons},
    )
    assert r.status_code == 200
    assert r.json()["icons"] == icons


@pytest.mark.parametrize(
    "icons",
    [
        [{"src": "http://example.com/icon.png"}],
        [{"src": "https://127.0.0.1/icon.png"}],
        [{"src": "https://user:pass@example.com/icon.png"}],
    ],
)
def test_create_server_rejects_risky_icon_urls(client, icons):
    r = client.post(PREFIX, json={"name": "com.example/icon-server", "icons": icons})
    assert r.status_code == 400
    assert "Icon URL" in r.text


def test_create_server_rejects_icon_url_outside_allowlist(client, monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_ICON_URL_ALLOWED_DOMAINS",
        "assets.example.com,*.cdn.example.com",
    )
    r = client.post(
        PREFIX,
        json={
            "name": "com.example/icon-allowlist-server",
            "icons": [{"src": "https://other.example.com/icon.png"}],
        },
    )
    assert r.status_code == 400
    assert "allowed domain list" in r.text


def test_create_server_accepts_icon_url_inside_allowlist(client, monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_ICON_URL_ALLOWED_DOMAINS",
        "assets.example.com,*.cdn.example.com",
    )
    r = client.post(
        PREFIX,
        json={
            "name": "com.example/icon-allowlist-accepted",
            "icons": [{"src": "https://foo.cdn.example.com/icon.png"}],
        },
    )
    assert r.status_code == 200


def test_create_server_accepts_public_http_icon_url_when_scheme_enabled(client, monkeypatch):
    monkeypatch.setenv("MLFLOW_ICON_URL_ALLOWED_SCHEMES", "http,https")
    r = client.post(
        PREFIX,
        json={
            "name": "com.example/icon-http-public",
            "icons": [{"src": "http://example.com/icon.png"}],
        },
    )
    assert r.status_code == 200


def test_create_server_accepts_local_icon_url_when_private_ips_enabled(client, monkeypatch):
    monkeypatch.setenv("MLFLOW_ICON_URL_ALLOW_PRIVATE_IPS", "true")
    r = client.post(
        PREFIX,
        json={
            "name": "com.example/icon-private-localhost",
            "icons": [{"src": "https://localhost/icon.png"}],
        },
    )
    assert r.status_code == 200


def test_create_server_accepts_local_http_icon_when_scheme_and_private_flags_enabled(
    client, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ICON_URL_ALLOWED_SCHEMES", "http,https")
    monkeypatch.setenv("MLFLOW_ICON_URL_ALLOW_PRIVATE_IPS", "true")
    r = client.post(
        PREFIX,
        json={
            "name": "com.example/icon-http-localhost",
            "icons": [{"src": "http://localhost/icon.png"}],
        },
    )
    assert r.status_code == 200


def test_create_server_rejects_invalid_icon_mime_type(client):
    r = client.post(
        PREFIX,
        json={
            "name": "com.example/icon-bad-mime",
            "icons": [
                {
                    "src": "https://example.com/icon.bin",
                    "mimeType": "application/octet-stream",
                }
            ],
        },
    )
    assert r.status_code == 400
    assert "Invalid icon mimeType" in r.text


@pytest.mark.parametrize(
    "mime_type",
    [
        "image/png",
        "image/svg+xml",
        "image/heic",
        "IMAGE/PNG",
    ],
)
def test_create_server_accepts_reasonable_icon_mime_types(client, mime_type):
    r = client.post(
        PREFIX,
        json={
            "name": f"com.example/icon-mime-{mime_type.replace('/', '-').replace('+', '-')}",
            "icons": [{"src": "https://example.com/icon", "mimeType": mime_type}],
        },
    )
    assert r.status_code == 200
    assert r.json()["icons"][0]["mimeType"] == mime_type.lower()


def test_create_duplicate_server(client):
    client.post(PREFIX, json={"name": "com.example/dup"})
    r = client.post(PREFIX, json={"name": "com.example/dup"})
    assert r.status_code == 400


@pytest.mark.parametrize(
    "invalid_name",
    [
        "my-server",
        "endpoints",
        "com",
        "com/example/extra",
        "/server",
        "com.example/",
        "com.example/aliases",
        "com.example/endpoints",
        "com.example/tags",
        "com.example/versions",
        "com.example/_server",
        "com.example/server_",
        "com.example/.server",
        "com.example/server-",
    ],
)
def test_create_server_invalid_name_rejected(client, invalid_name):
    r = client.post(PREFIX, json={"name": invalid_name})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Expected '<reverse-dns namespace>/<server slug>'" in r.json()["message"]


@pytest.mark.parametrize(
    "name",
    [
        "io.github.TestOrg/server-name",
        "com/server-name",
        "io.github.test/server_name",
        "io.github.test/server.name",
    ],
)
def test_create_server_accepts_upstream_name_shapes(client, name):
    r = client.post(PREFIX, json={"name": name})
    assert r.status_code == 200
    assert r.json()["name"] == name


def test_get_server(client):
    client.post(PREFIX, json={"name": "com.example/get-me"})
    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/get-me')}")
    assert r.status_code == 200
    assert r.json()["name"] == "com.example/get-me"


def test_get_nonexistent_server(client):
    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/no-such-server')}")
    assert r.status_code == 404


def test_encoded_slashed_name_path_round_trip_for_version_routes(client):
    name = "com.example/encoded-server"
    version = "2025.6.0"

    create_r = client.post(
        f"{PREFIX}/{_encode_path_param(name)}/versions",
        json={"server_json": _server_json(name, version), "status": "active"},
    )
    assert create_r.status_code == 200, create_r.text
    assert create_r.json()["name"] == name
    assert create_r.json()["version"] == version

    get_r = client.get(
        f"{PREFIX}/{_encode_path_param(name)}/versions/{_encode_path_param(version)}"
    )
    assert get_r.status_code == 200, get_r.text
    assert get_r.json()["name"] == name
    assert get_r.json()["version"] == version


def test_search_servers(client):
    for name in ["alpha", "beta", "gamma"]:
        client.post(PREFIX, json={"name": f"com.example/{name}"})
    r = client.get(PREFIX, params={"max_results": 2})
    assert r.status_code == 200
    data = r.json()
    assert len(data["mcp_servers"]) == 2
    assert data["next_page_token"] is not None

    r2 = client.get(PREFIX, params={"max_results": 2, "page_token": data["next_page_token"]})
    assert r2.status_code == 200
    assert len(r2.json()["mcp_servers"]) == 1


def test_server_responses_include_nested_endpoint_server_name(client):
    sj = _server_json("com.example/nested-bind-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/nested-bind-srv')}/versions",
        json={"server_json": sj, "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/nested-bind-srv')}/endpoints",
        json={
            "url": "https://mcp.example.com/nested-bind-srv",
            "server_version": "1.0.0",
        },
    )

    get_r = client.get(f"{PREFIX}/{_encode_path_param('com.example/nested-bind-srv')}")
    assert get_r.status_code == 200
    assert get_r.json()["access_endpoints"][0]["server_name"] == "com.example/nested-bind-srv"

    search_r = client.get(PREFIX, params={"filter_string": "name = 'com.example/nested-bind-srv'"})
    assert search_r.status_code == 200
    assert search_r.json()["mcp_servers"][0]["access_endpoints"][0]["server_name"] == (
        "com.example/nested-bind-srv"
    )


def test_update_server(client):
    client.post(PREFIX, json={"name": "com.example/upd"})
    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/upd')}",
        json={"description": "updated", "display_name": "Upd"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["description"] == "updated"
    assert data["display_name"] == "Upd"


def test_update_server_rejects_risky_icon_urls(client):
    client.post(PREFIX, json={"name": "com.example/upd-icons"})
    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/upd-icons')}",
        json={"icons": [{"src": "https://127.0.0.1/icon.png"}]},
    )
    assert r.status_code == 400
    assert "Icon URL" in r.text


def test_update_server_rejects_latest_version_field(client):
    client.post(PREFIX, json={"name": "com.example/upd-latest"})
    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/upd-latest')}",
        json={"latest_version": "2.0.0"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "latest_version is read-only" in r.json()["message"]


def test_delete_server(client):
    client.post(PREFIX, json={"name": "com.example/del"})
    r = client.delete(f"{PREFIX}/{_encode_path_param('com.example/del')}")
    assert r.status_code == 200
    assert client.get(f"{PREFIX}/{_encode_path_param('com.example/del')}").status_code == 404


def test_delete_server_rejects_active_version(client):
    name = "com.example/del-active"
    sj = _server_json(name, "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param(name)}/versions",
        json={"server_json": sj, "status": "active"},
    )

    r = client.delete(f"{PREFIX}/{_encode_path_param(name)}")
    assert r.status_code == 400
    assert "active version" in r.text


def test_server_crud_with_slashed_name(client):
    name = "io.github.org/my-server"
    client.post(PREFIX, json={"name": name})
    r = client.get(PREFIX + "/" + _encode_path_param(name))
    assert r.status_code == 200
    assert r.json()["name"] == name


def test_version_crud_with_slashed_name(client):
    name = "io.github.org/server"
    sj = _server_json(name, "1.0.0")
    encoded_name = _encode_path_param(name)
    r = client.post(PREFIX + f"/{encoded_name}/versions", json={"server_json": sj})
    assert r.status_code == 200
    assert r.json()["name"] == name
    assert r.json()["version"] == "1.0.0"

    r = client.get(PREFIX + f"/{encoded_name}/versions/1.0.0")
    assert r.status_code == 200
    assert r.json()["version"] == "1.0.0"


def test_version_crud_with_slashed_name_and_version(client):
    name = "io.github.org/slash-server"
    version = "2025/06"
    encoded_name = _encode_path_param(name)
    sj = _server_json(name, version)
    r = client.post(
        PREFIX + f"/{encoded_name}/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 400
    assert "server_json.version" in r.json()["message"]


def test_create_version(client):
    sj = _server_json("com.example/v-server", "1.0.0", title="Test")
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/v-server')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "com.example/v-server"
    assert data["version"] == "1.0.0"
    assert data["status"] == "active"
    assert data["server_json"]["title"] == "Test"
    assert data["tools"] == []


@pytest.mark.parametrize("status", ["deprecated", "deleted"])
def test_create_version_rejects_non_initial_statuses(client, status):
    sj = _server_json(f"com.example/{status}-create", "1.0.0")
    r = client.post(
        f"{PREFIX}/{_encode_path_param(f'com.example/{status}-create')}" + "/versions",
        json={"server_json": sj, "status": status},
    )
    assert r.status_code == 400
    assert "Initial MCP server registration status must be 'draft' or 'active'" in r.text


def test_ensure_version_create_parent_access_rejects_raced_existing_parent_without_update():
    can_update_existing = mock.Mock(return_value=False)
    request = SimpleNamespace(
        state=SimpleNamespace(
            mcp_server_parent_auto_created=True,
            mcp_server_can_update_existing_recheck=can_update_existing,
        )
    )
    store = mock.Mock()
    store.create_mcp_server.side_effect = MlflowException(
        "server exists",
        error_code=RESOURCE_ALREADY_EXISTS,
    )

    with pytest.raises(MlflowException, match="Permission denied") as exc:
        _ensure_version_create_parent_access(store, "com.example/race", "alice", request)

    assert exc.value.error_code == ErrorCode.Name(PERMISSION_DENIED)
    assert request.state.mcp_server_parent_auto_created is False
    can_update_existing.assert_called_once_with()


def test_ensure_version_create_parent_access_allows_raced_existing_parent_with_update():
    can_update_existing = mock.Mock(return_value=True)
    request = SimpleNamespace(
        state=SimpleNamespace(
            mcp_server_parent_auto_created=True,
            mcp_server_can_update_existing_recheck=can_update_existing,
        )
    )
    store = mock.Mock()
    store.create_mcp_server.side_effect = MlflowException(
        "server exists",
        error_code=RESOURCE_ALREADY_EXISTS,
    )

    _ensure_version_create_parent_access(store, "com.example/race", "alice", request)

    assert request.state.mcp_server_parent_auto_created is False
    can_update_existing.assert_called_once_with()


def test_create_version_name_mismatch(client):
    sj = _server_json("com.example/wrong-name", "1.0.0")
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/v-server')}" + "/versions",
        json={"server_json": sj},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "does not match" in r.json()["message"]


def test_create_version_missing_required_field_returns_mlflow_error(client):
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/v-server')}" + "/versions",
        json={"server_json": {"name": "com.example/v-server"}},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "server_json.version" in r.json()["message"]


def test_create_version_invalid_semver_returns_mlflow_error(client):
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/v-server')}" + "/versions",
        json={"server_json": {"name": "com.example/v-server", "version": "1.0"}},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "server_json.version" in r.json()["message"]


def test_create_version_rejects_semver_component_exceeding_db_integer_limit(client):
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/v-server')}" + "/versions",
        json={"server_json": {"name": "com.example/v-server", "version": "2147483648.0.0"}},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "server_json.version" in r.json()["message"]
    assert "2147483647" in r.json()["message"]


def test_create_version_invalid_package_shape_returns_mlflow_error(client):
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/pkg-server')}" + "/versions",
        json={
            "server_json": {
                "name": "com.example/pkg-server",
                "version": "1.0.0",
                "packages": [{}],
            }
        },
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "server_json.packages.0.registryType" in r.json()["message"]


def test_create_version_rejects_non_namespaced_server_name(client):
    r = client.post(
        PREFIX + "/my-server/versions",
        json={"server_json": {"name": "my-server", "version": "1.0.0"}},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Expected '<reverse-dns namespace>/<server slug>'" in r.json()["message"]


@pytest.mark.parametrize(
    "name",
    [
        "io.github.TestOrg/server-name",
        "com/server-name",
        "io.github.test/server_name",
        "io.github.test/server.name",
    ],
)
def test_create_version_accepts_upstream_name_shapes(client, name):
    encoded_name = _encode_path_param(name)
    r = client.post(
        f"{PREFIX}/{encoded_name}/versions",
        json={"server_json": {"name": name, "version": "1.0.0"}},
    )
    assert r.status_code == 200
    assert r.json()["name"] == name


@pytest.mark.parametrize(
    "invalid_name",
    [
        "com.example/aliases",
        "com.example/endpoints",
        "com.example/tags",
        "com.example/versions",
        "com.example/_server",
        "com.example/server_",
        "com.example/.server",
        "com.example/server-",
    ],
)
def test_create_version_invalid_server_name_rejected(client, invalid_name):
    encoded_name = _encode_path_param(invalid_name)
    r = client.post(
        f"{PREFIX}/{encoded_name}/versions",
        json={"server_json": {"name": invalid_name, "version": "1.0.0"}},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Expected '<reverse-dns namespace>/<server slug>'" in r.json()["message"]


def test_create_version_with_tools(client):
    sj = _server_json("com.example/tools-server", "1.0.0")
    tools = [{"name": "web_search", "description": "Search the web"}]
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/tools-server')}" + "/versions",
        json={"server_json": sj, "tools": tools, "status": "active"},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["tools"]) == 1
    assert data["tools"][0]["name"] == "web_search"


def test_create_version_rejects_too_many_tools(client):
    sj = _server_json("com.example/too-many-tools", "1.0.0")
    tools = [{"name": f"tool-{i}"} for i in range(1001)]
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/too-many-tools')}/versions",
        json={"server_json": sj, "tools": tools, "status": "active"},
    )
    assert r.status_code == 400
    assert "at most 1000 items" in r.text


def test_create_version_with_tool_icons_preserves_extra_fields(client):
    sj = _server_json("com.example/tool-icons-server", "1.0.0")
    tools = [
        {
            "name": "web_search",
            "icons": [
                {
                    "src": "https://example.com/icon.png",
                    "sizes": ["32x32"],
                    "purpose": "maskable",
                }
            ],
        }
    ]
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/tool-icons-server')}/versions",
        json={"server_json": sj, "tools": tools, "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["tools"][0]["name"] == "web_search"
    assert r.json()["tools"][0]["icons"] == tools[0]["icons"]


def test_create_version_rejects_risky_tool_icon_urls(client):
    sj = _server_json("com.example/tool-icons-invalid", "1.0.0")
    tools = [{"name": "web_search", "icons": [{"src": "https://127.0.0.1/icon.png"}]}]
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/tool-icons-invalid')}/versions",
        json={"server_json": sj, "tools": tools, "status": "active"},
    )
    assert r.status_code == 400
    assert "Icon URL" in r.text


def test_create_version_rejects_risky_server_json_icon_urls(client):
    sj = _server_json(
        "com.example/server-json-icons-invalid",
        "1.0.0",
        icons=[{"src": "http://example.com/icon.png"}],
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/server-json-icons-invalid')}/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 400
    assert "Icon URL" in r.text


def test_create_version_accepts_remote_with_null_type(client):
    sj = _server_json(
        "com.example/null-remote-type",
        "1.0.0",
        remotes=[{"type": None, "url": "https://example.com/mcp"}],
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/null-remote-type')}/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["server_json"]["remotes"][0]["type"] is None


def test_create_version_accepts_remote_without_url(client):
    sj = _server_json(
        "com.example/missing-remote-url",
        "1.0.0",
        remotes=[{"type": "streamable-http"}],
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/missing-remote-url')}/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["server_json"]["remotes"][0]["type"] == "streamable-http"
    assert "url" not in r.json()["server_json"]["remotes"][0]


def test_create_version_preserves_meta_icons_metadata(client):
    sj = _server_json(
        "com.example/meta-icons",
        "1.0.0",
        _meta={"icons": {"not": "an-icon-list"}, "other": "preserved"},
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/meta-icons')}/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["server_json"]["_meta"] == sj["_meta"]


def test_create_version_preserves_empty_tools_list(client):
    sj = _server_json("com.example/empty-tools-server", "1.0.0")
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/empty-tools-server')}" + "/versions",
        json={"server_json": sj, "tools": [], "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["tools"] == []


def test_create_version_accepts_repository_object(client):
    sj = _server_json(
        "com.example/repo-srv",
        "1.0.0",
        repository={
            "url": "https://github.com/modelcontextprotocol/servers",
            "source": "github",
            "id": "repo-id-123",
            "subfolder": "src/repo-srv",
        },
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/repo-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["server_json"]["repository"] == sj["repository"]


def test_get_version(client):
    sj = _server_json("com.example/gv", "2.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/gv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/gv')}" + "/versions/2.0.0")
    assert r.status_code == 200
    assert r.json()["version"] == "2.0.0"
    assert r.json()["tools"] == []


def test_latest_alias_does_not_override_literal_version_route(client):
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/lat')}" + "/versions",
        json={"server_json": _server_json("com.example/lat", "1.0.0"), "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/lat')}" + "/versions",
        json={"server_json": _server_json("com.example/lat", "2.0.0"), "status": "active"},
    )

    latest_version_r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/lat')}" + "/versions/latest"
    )
    assert latest_version_r.status_code == 404

    latest_alias_r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/lat')}" + "/aliases/latest"
    )
    assert latest_alias_r.status_code == 200
    assert latest_alias_r.json()["version"] == "2.0.0"


def test_latest_alias_returns_highest_active_semver(client):
    for version in ("1.2.0", "1.10.0"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/semver-lat')}" + "/versions",
            json={
                "server_json": _server_json("com.example/semver-lat", version),
                "status": "active",
            },
        )

    alias_r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/semver-lat')}" + "/aliases/latest"
    )
    assert alias_r.status_code == 200
    assert alias_r.json()["version"] == "1.10.0"

    literal_r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/semver-lat')}" + "/versions/1.10.0"
    )
    assert literal_r.status_code == 200
    assert literal_r.json()["version"] == "1.10.0"


def test_latest_alias_uses_created_at_before_raw_version_as_tiebreaker(client, monkeypatch):
    times = iter((1000, 2000))
    monkeypatch.setattr(
        "mlflow.store.tracking.mcp_server_registry.sqlalchemy_mixin.get_current_time_millis",
        lambda: next(times),
    )
    for version in ("1.0.0+xyz", "1.0.0+abc"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/build-meta-lat')}" + "/versions",
            json={
                "server_json": _server_json("com.example/build-meta-lat", version),
                "status": "active",
            },
        )

    alias_r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/build-meta-lat')}" + "/aliases/latest"
    )
    assert alias_r.status_code == 200
    assert alias_r.json()["version"] == "1.0.0+abc"

    server_r = client.get(f"{PREFIX}/{_encode_path_param('com.example/build-meta-lat')}")
    assert server_r.status_code == 200
    assert server_r.json()["latest_version"] == "1.0.0+abc"


def test_latest_alias_handles_alphanumeric_prefix_prerelease_ordering(client):
    for version in ("1.0.0-alpha", "1.0.0-alpha1"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/prefix-prerelease-lat')}" + "/versions",
            json={
                "server_json": _server_json("com.example/prefix-prerelease-lat", version),
                "status": "active",
            },
        )

    alias_r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/prefix-prerelease-lat')}" + "/aliases/latest"
    )
    assert alias_r.status_code == 200
    assert alias_r.json()["version"] == "1.0.0-alpha1"


def test_latest_alias_handles_prerelease_tuple_length_and_numeric_rules(client):
    for version in ("1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-1"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/prerelease-tuple-lat')}" + "/versions",
            json={
                "server_json": _server_json("com.example/prerelease-tuple-lat", version),
                "status": "active",
            },
        )

    alias_r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/prerelease-tuple-lat')}" + "/aliases/latest"
    )
    assert alias_r.status_code == 200
    assert alias_r.json()["version"] == "1.0.0-alpha.1"


def test_search_versions(client):
    for v in ["1.0.0", "2.0.0", "3.0.0"]:
        sj = _server_json("com.example/sv", v)
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/sv')}" + "/versions",
            json={"server_json": sj, "status": "active"},
        )
    r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/sv')}" + "/versions", params={"max_results": 2}
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["mcp_server_versions"]) == 2
    assert all(version["tools"] == [] for version in data["mcp_server_versions"])


def test_search_versions_order_by_version_uses_semver_desc(client):
    for version in ("1.2.0-alpha", "1.2.0", "1.10.0"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/semver-order-desc')}/versions",
            json={
                "server_json": _server_json("com.example/semver-order-desc", version),
                "status": "active",
            },
        )

    r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/semver-order-desc')}/versions",
        params=[("order_by", "`version` DESC")],
    )
    assert r.status_code == 200
    assert [v["version"] for v in r.json()["mcp_server_versions"]] == [
        "1.10.0",
        "1.2.0",
        "1.2.0-alpha",
    ]


def test_search_versions_order_by_version_uses_semver_asc(client):
    for version in ("1.2.0-alpha", "1.2.0", "1.10.0"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/semver-order-asc')}/versions",
            json={
                "server_json": _server_json("com.example/semver-order-asc", version),
                "status": "active",
            },
        )

    r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/semver-order-asc')}/versions",
        params=[("order_by", "`version` ASC")],
    )
    assert r.status_code == 200
    assert [v["version"] for v in r.json()["mcp_server_versions"]] == [
        "1.2.0-alpha",
        "1.2.0",
        "1.10.0",
    ]


def test_search_versions_order_by_version_ignores_build_metadata_precedence(client):
    for version in ("1.0.1", "1.0.0+aaa", "1.0.0+zzz"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/semver-order-build')}/versions",
            json={
                "server_json": _server_json("com.example/semver-order-build", version),
                "status": "active",
            },
        )

    r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/semver-order-build')}/versions",
        params=[("order_by", "`version` DESC")],
    )
    assert r.status_code == 200
    versions = [v["version"] for v in r.json()["mcp_server_versions"]]
    assert versions[0] == "1.0.1"
    assert set(versions[1:]) == {"1.0.0+aaa", "1.0.0+zzz"}


def test_search_versions_filter_by_version_equality_uses_exact_string_match(client):
    for version in ("1.0.0-alpha+aaa", "1.0.0-alpha+zzz", "1.0.0"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/semver-filter-eq')}/versions",
            json={
                "server_json": _server_json("com.example/semver-filter-eq", version),
                "status": "active",
            },
        )

    r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/semver-filter-eq')}/versions",
        params={"filter_string": "version = '1.0.0-alpha+aaa'"},
    )
    assert r.status_code == 200
    versions = {v["version"] for v in r.json()["mcp_server_versions"]}
    assert versions == {"1.0.0-alpha+aaa"}


def test_search_versions_filter_by_version_inequality_uses_semver_precedence(client):
    for version in ("1.2.0-alpha", "1.2.0", "1.10.0"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/semver-filter-gt')}/versions",
            json={
                "server_json": _server_json("com.example/semver-filter-gt", version),
                "status": "active",
            },
        )

    r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/semver-filter-gt')}/versions",
        params={"filter_string": "version > '1.2.0-alpha'"},
    )
    assert r.status_code == 200
    versions = {v["version"] for v in r.json()["mcp_server_versions"]}
    assert versions == {"1.2.0", "1.10.0"}


def test_search_versions_filter_by_version_rejects_like(client):
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/semver-filter-like')}/versions",
        json={
            "server_json": _server_json("com.example/semver-filter-like", "1.2.3"),
            "status": "active",
        },
    )

    r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/semver-filter-like')}/versions",
        params={"filter_string": "version LIKE '1.%'"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "version only supports semantic comparators" in r.json()["message"]


def test_update_version_status(client):
    sj = _server_json("com.example/uv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/uv')}" + "/versions", json={"server_json": sj}
    )
    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/uv')}" + "/versions/1.0.0",
        json={"status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "active"


def test_server_response_recomputes_status_and_latest_after_transitions(client):
    for version in ("1.0.0", "2.0.0"):
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/status-lat')}" + "/versions",
            json={
                "server_json": _server_json("com.example/status-lat", version),
                "status": "active",
            },
        )

    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/status-lat')}").json()
    assert server["status"] == "active"
    assert server["latest_version"] == "2.0.0"

    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/status-lat')}" + "/versions/2.0.0",
        json={"status": "deprecated"},
    )
    assert r.status_code == 200

    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/status-lat')}").json()
    assert server["status"] == "active"
    assert server["latest_version"] == "1.0.0"

    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/status-lat')}" + "/versions/1.0.0",
        json={"status": "draft"},
    )
    assert r.status_code == 200

    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/status-lat')}").json()
    assert server["status"] == "deprecated"
    assert server["latest_version"] == "2.0.0"


def test_server_response_description_falls_back_to_parent_resolved_version(client):
    _create_version(
        client,
        "com.example/description-fallback",
        "1.0.0",
        status="active",
        description="active description",
    )
    _create_version(
        client,
        "com.example/description-fallback",
        "2.0.0",
        status="deprecated",
        description="deprecated description",
    )

    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/description-fallback')}").json()
    assert server["description"] == "active description"
    assert server["latest_version"] == "1.0.0"
    assert server["status"] == "active"

    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/description-fallback')}" + "/versions/1.0.0",
        json={"status": "draft"},
    )
    assert r.status_code == 200

    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/description-fallback')}").json()
    assert server["description"] == "deprecated description"
    assert server["latest_version"] == "2.0.0"
    assert server["status"] == "deprecated"


def test_latest_alias_falls_back_to_non_active_version(client):
    _create_version(client, "com.example/latest-fallback", "1.2.0", status="deprecated")
    _create_version(client, "com.example/latest-fallback", "1.3.0", status="draft")

    alias_r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/latest-fallback')}" + "/aliases/latest"
    )
    assert alias_r.status_code == 200
    assert alias_r.json()["version"] == "1.3.0"

    server_r = client.get(f"{PREFIX}/{_encode_path_param('com.example/latest-fallback')}")
    assert server_r.status_code == 200
    assert server_r.json()["latest_version"] == "1.3.0"
    assert server_r.json()["status"] == "draft"


def test_update_version_rejects_null_status(client):
    sj = _server_json("com.example/null-status", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/null-status')}" + "/versions",
        json={"server_json": sj},
    )
    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/null-status')}" + "/versions/1.0.0",
        json={"status": None},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "status cannot be null" in r.json()["message"]


def test_delete_version(client):
    sj = _server_json("com.example/dv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/dv')}" + "/versions", json={"server_json": sj}
    )
    r = client.delete(f"{PREFIX}/{_encode_path_param('com.example/dv')}" + "/versions/1.0.0")
    assert r.status_code == 200


def test_create_endpoint_with_version(client):
    sj = _server_json("com.example/bind-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bind-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bind-srv')}" + "/endpoints",
        json={
            "url": "https://mcp.example.com/bind-srv",
            "server_version": "1.0.0",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["server_name"] == "com.example/bind-srv"
    assert data["url"] == "https://mcp.example.com/bind-srv"
    assert data["server_version"] == "1.0.0"


def test_create_endpoint_with_alias(client):
    sj = _server_json("com.example/alias-bind-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/alias-bind-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/alias-bind-srv')}" + "/aliases",
        json={"alias": "prod", "version": "1.0.0"},
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/alias-bind-srv')}" + "/endpoints",
        json={
            "url": "https://mcp.example.com/alias-bind-srv",
            "server_alias": "prod",
        },
    )
    assert r.status_code == 200
    assert r.json()["server_alias"] == "prod"


def test_get_endpoint_with_tools(client):
    sj = _server_json("com.example/bt-srv", "1.0.0")
    tools = [{"name": "tool1", "description": "A tool"}]
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bt-srv')}" + "/versions",
        json={"server_json": sj, "status": "active", "tools": tools},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bt-srv')}" + "/endpoints",
        json={"url": "https://mcp.example.com/bt-srv", "server_version": "1.0.0"},
    )
    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/bt-srv')}" + "/endpoints")
    assert r.status_code == 200
    endpoints = r.json()["mcp_access_endpoints"]
    assert len(endpoints) == 1
    assert endpoints[0]["tools"][0]["name"] == "tool1"


def test_get_endpoint_includes_resolved_version(client):
    sj = _server_json("com.example/brv-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/brv-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    create_r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/brv-srv')}" + "/endpoints",
        json={"url": "https://mcp.example.com/brv-srv", "server_version": "1.0.0"},
    )
    bid = create_r.json()["id"]
    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/brv-srv')}" + f"/endpoints/{bid}")
    assert r.status_code == 200
    assert r.json()["resolved_version"]["name"] == "com.example/brv-srv"
    assert r.json()["resolved_version"]["version"] == "1.0.0"
    assert r.json()["resolved_version"]["tools"] == []


def test_get_endpoint_preserves_empty_tools_list(client):
    sj = _server_json("com.example/bt-empty-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bt-empty-srv')}" + "/versions",
        json={"server_json": sj, "status": "active", "tools": []},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bt-empty-srv')}" + "/endpoints",
        json={"url": "https://mcp.example.com/bt-empty-srv", "server_version": "1.0.0"},
    )
    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/bt-empty-srv')}" + "/endpoints")
    assert r.status_code == 200
    endpoints = r.json()["mcp_access_endpoints"]
    assert len(endpoints) == 1
    assert endpoints[0]["tools"] == []


def test_search_endpoints_workspace_wide(client):
    for name in ["ws-a", "ws-b"]:
        sj = _server_json(f"com.example/{name}", "1.0.0")
        client.post(
            f"{PREFIX}/{_encode_path_param(f'com.example/{name}')}/versions",
            json={"server_json": sj, "status": "active"},
        )
        client.post(
            f"{PREFIX}/{_encode_path_param(f'com.example/{name}')}/endpoints",
            json={"url": f"https://mcp.example.com/{name}", "server_version": "1.0.0"},
        )
    r = client.get(PREFIX + "/endpoints")
    assert r.status_code == 200
    assert len(r.json()["mcp_access_endpoints"]) == 2


def test_search_endpoints_server_scoped(client):
    for name in ["sc-a", "sc-b"]:
        sj = _server_json(f"com.example/{name}", "1.0.0")
        client.post(
            f"{PREFIX}/{_encode_path_param(f'com.example/{name}')}/versions",
            json={"server_json": sj, "status": "active"},
        )
        client.post(
            f"{PREFIX}/{_encode_path_param(f'com.example/{name}')}/endpoints",
            json={"url": f"https://mcp.example.com/{name}", "server_version": "1.0.0"},
        )
    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/sc-a')}/endpoints")
    assert r.status_code == 200
    assert len(r.json()["mcp_access_endpoints"]) == 1
    assert r.json()["mcp_access_endpoints"][0]["server_name"] == "com.example/sc-a"


def test_update_endpoint(client):
    sj = _server_json("com.example/ub-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/ub-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    create_r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/ub-srv')}" + "/endpoints",
        json={"url": "https://old.example.com", "server_version": "1.0.0"},
    )
    bid = create_r.json()["id"]
    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/ub-srv')}" + f"/endpoints/{bid}",
        json={"url": "https://new.example.com"},
    )
    assert r.status_code == 200
    assert r.json()["url"] == "https://new.example.com"


def test_delete_endpoint(client):
    sj = _server_json("com.example/db-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/db-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    create_r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/db-srv')}" + "/endpoints",
        json={"url": "https://mcp.example.com/db", "server_version": "1.0.0"},
    )
    bid = create_r.json()["id"]
    r = client.delete(f"{PREFIX}/{_encode_path_param('com.example/db-srv')}" + f"/endpoints/{bid}")
    assert r.status_code == 200
    assert (
        client.get(
            f"{PREFIX}/{_encode_path_param('com.example/db-srv')}" + f"/endpoints/{bid}"
        ).status_code
        == 404
    )


def test_set_and_delete_server_tag(client):
    client.post(PREFIX, json={"name": "com.example/tag-srv"})
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/tag-srv')}" + "/tags",
        json={"key": "env", "value": "prod"},
    )
    assert r.status_code == 200
    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/tag-srv')}").json()
    assert server["tags"]["env"] == "prod"

    r = client.delete(f"{PREFIX}/{_encode_path_param('com.example/tag-srv')}" + "/tags/env")
    assert r.status_code == 200
    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/tag-srv')}").json()
    assert "env" not in server["tags"]


def test_delete_server_tag_with_slash_key(client):
    client.post(PREFIX, json={"name": "com.example/slash-tag-srv"})
    encoded_key = _encode_path_param("team/platform")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/slash-tag-srv')}" + "/tags",
        json={"key": "team/platform", "value": "prod"},
    )
    r = client.delete(
        f"{PREFIX}/{_encode_path_param('com.example/slash-tag-srv')}" + f"/tags/{encoded_key}"
    )
    assert r.status_code == 200
    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/slash-tag-srv')}").json()
    assert "team/platform" not in server["tags"]


def test_set_and_delete_version_tag(client):
    sj = _server_json("com.example/vt-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/vt-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/vt-srv')}" + "/versions/1.0.0/tags",
        json={"key": "stage", "value": "beta"},
    )
    assert r.status_code == 200
    ver = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/vt-srv')}" + "/versions/1.0.0"
    ).json()
    assert ver["tags"]["stage"] == "beta"

    r = client.delete(
        f"{PREFIX}/{_encode_path_param('com.example/vt-srv')}" + "/versions/1.0.0/tags/stage"
    )
    assert r.status_code == 200
    ver = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/vt-srv')}" + "/versions/1.0.0"
    ).json()
    assert "stage" not in ver["tags"]


def test_delete_version_tag_with_slash_key(client):
    sj = _server_json("com.example/slash-vt-srv", "1.0.0")
    encoded_key = _encode_path_param("team/platform")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/slash-vt-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/slash-vt-srv')}" + "/versions/1.0.0/tags",
        json={"key": "team/platform", "value": "beta"},
    )
    r = client.delete(
        f"{PREFIX}/{_encode_path_param('com.example/slash-vt-srv')}"
        + f"/versions/1.0.0/tags/{encoded_key}"
    )
    assert r.status_code == 200
    ver = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/slash-vt-srv')}" + "/versions/1.0.0"
    ).json()
    assert "team/platform" not in ver["tags"]


def test_set_and_resolve_alias(client):
    sj = _server_json("com.example/alias-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/alias-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/alias-srv')}" + "/aliases",
        json={"alias": "prod", "version": "1.0.0"},
    )
    assert r.status_code == 200

    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/alias-srv')}" + "/aliases/prod")
    assert r.status_code == 200
    assert r.json()["version"] == "1.0.0"


def test_alias_with_slash_round_trips(client):
    sj = _server_json("com.example/slash-alias-srv", "1.0.0")
    encoded_alias = _encode_path_param("team/prod")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/slash-alias-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/slash-alias-srv')}" + "/aliases",
        json={"alias": "team/prod", "version": "1.0.0"},
    )
    r = client.get(
        f"{PREFIX}/{_encode_path_param('com.example/slash-alias-srv')}"
        + f"/aliases/{encoded_alias}"
    )
    assert r.status_code == 200
    assert r.json()["version"] == "1.0.0"

    r = client.delete(
        f"{PREFIX}/{_encode_path_param('com.example/slash-alias-srv')}"
        + f"/aliases/{encoded_alias}"
    )
    assert r.status_code == 200


def test_resolve_latest_alias(client):
    for v in ["1.0.0", "2.0.0"]:
        sj = _server_json("com.example/latest-srv", v)
        client.post(
            f"{PREFIX}/{_encode_path_param('com.example/latest-srv')}" + "/versions",
            json={"server_json": sj, "status": "active"},
        )
    r = client.get(f"{PREFIX}/{_encode_path_param('com.example/latest-srv')}" + "/aliases/latest")
    assert r.status_code == 200
    assert r.json()["version"] == "2.0.0"


def test_delete_alias(client):
    sj = _server_json("com.example/da-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/da-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/da-srv')}" + "/aliases",
        json={"alias": "staging", "version": "1.0.0"},
    )
    r = client.delete(f"{PREFIX}/{_encode_path_param('com.example/da-srv')}" + "/aliases/staging")
    assert r.status_code == 200


def test_missing_required_field(client):
    r = client.post(PREFIX, json={})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"


def test_invalid_server_json(client):
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/x')}" + "/versions",
        json={"server_json": {"name": "com.example/x"}},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"


def test_invalid_status_transition(client):
    sj = _server_json("com.example/is-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/is-srv')}" + "/versions",
        json={"server_json": sj},
    )
    r = client.patch(
        f"{PREFIX}/{_encode_path_param('com.example/is-srv')}" + "/versions/1.0.0",
        json={"status": "deprecated"},
    )
    assert r.status_code == 400


def test_invalid_status_value(client):
    sj = _server_json("com.example/bad-status", "1.0.0")
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bad-status')}" + "/versions",
        json={"server_json": sj, "status": "bogus"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Invalid status" in r.json()["message"]


def test_invalid_transport_type(client):
    sj = _server_json("com.example/bad-transport", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bad-transport')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/bad-transport')}" + "/endpoints",
        json={
            "url": "https://example.com",
            "transport_type": "ftp",
            "server_version": "1.0.0",
        },
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Invalid transport_type" in r.json()["message"]


def test_server_response_has_aliases_as_list(client):
    sj = _server_json("com.example/shape-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/shape-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/shape-srv')}" + "/aliases",
        json={"alias": "prod", "version": "1.0.0"},
    )
    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/shape-srv')}").json()
    assert isinstance(server["aliases"], list)
    assert server["aliases"][0]["alias"] == "prod"
    assert server["aliases"][0]["version"] == "1.0.0"


def test_server_response_includes_endpoints(client):
    sj = _server_json("com.example/sb-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/sb-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/sb-srv')}" + "/endpoints",
        json={"url": "https://example.com/sb", "server_version": "1.0.0"},
    )
    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/sb-srv')}").json()
    assert len(server["access_endpoints"]) == 1
    assert server["access_endpoints"][0]["url"] == "https://example.com/sb"


def test_server_response_includes_endpoint_resolved_version(client):
    sj = _server_json("com.example/sbrv-srv", "1.0.0")
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/sbrv-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    client.post(
        f"{PREFIX}/{_encode_path_param('com.example/sbrv-srv')}" + "/endpoints",
        json={"url": "https://example.com/sbrv", "server_version": "1.0.0"},
    )
    server = client.get(f"{PREFIX}/{_encode_path_param('com.example/sbrv-srv')}").json()
    assert len(server["access_endpoints"]) == 1
    assert server["access_endpoints"][0]["resolved_version"]["version"] == "1.0.0"


def test_server_json_extra_fields_preserved(client):
    sj = _server_json("com.example/extra-srv", "1.0.0", custom_field="preserved")
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/extra-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["server_json"]["custom_field"] == "preserved"


def test_server_json_explicit_nulls_preserved(client):
    sj = _server_json("com.example/null-srv", "1.0.0", description=None, custom_field=None)
    r = client.post(
        f"{PREFIX}/{_encode_path_param('com.example/null-srv')}" + "/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    assert "description" in r.json()["server_json"]
    assert r.json()["server_json"]["description"] is None
    assert "custom_field" in r.json()["server_json"]
    assert r.json()["server_json"]["custom_field"] is None
    assert "repository" not in r.json()["server_json"]
