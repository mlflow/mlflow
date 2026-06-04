from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock
from urllib.parse import quote

import pytest
from starlette.testclient import TestClient

from mlflow.server.registry_fastapi_app import create_registry_fastapi_app
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

PREFIX = "/ajax-api/3.0/mlflow/mcp-servers"


def _server_json(name: str, version: str, **extra) -> dict[str, Any]:
    d = {"name": name, "version": version}
    d.update(extra)
    return d


def _encode_path_param(value: str) -> str:
    return quote(value, safe="")


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
        yield TestClient(create_registry_fastapi_app())


def test_create_server(client):
    r = client.post(PREFIX, json={"name": "my-server", "description": "A test server"})
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "my-server"
    assert data["description"] == "A test server"
    assert data["creation_timestamp"] is not None


def test_create_server_with_icons(client):
    icons = [{"src": "https://example.com/icon.png", "sizes": ["32x32"]}]
    r = client.post(PREFIX, json={"name": "icon-server", "icons": icons})
    assert r.status_code == 200
    assert r.json()["icons"] == icons


def test_create_duplicate_server(client):
    client.post(PREFIX, json={"name": "dup"})
    r = client.post(PREFIX, json={"name": "dup"})
    assert r.status_code == 400


def test_get_server(client):
    client.post(PREFIX, json={"name": "get-me"})
    r = client.get(PREFIX + "/get-me")
    assert r.status_code == 200
    assert r.json()["name"] == "get-me"


def test_get_nonexistent_server(client):
    r = client.get(PREFIX + "/no-such-server")
    assert r.status_code == 404


def test_search_servers(client):
    for name in ["alpha", "beta", "gamma"]:
        client.post(PREFIX, json={"name": name})
    r = client.get(PREFIX, params={"max_results": 2})
    assert r.status_code == 200
    data = r.json()
    assert len(data["mcp_servers"]) == 2
    assert data["next_page_token"] is not None

    r2 = client.get(PREFIX, params={"max_results": 2, "page_token": data["next_page_token"]})
    assert r2.status_code == 200
    assert len(r2.json()["mcp_servers"]) == 1


def test_server_responses_include_nested_binding_server_name(client):
    sj = _server_json("nested-bind-srv", "1.0")
    client.post(PREFIX + "/nested-bind-srv/versions", json={"server_json": sj, "status": "active"})
    client.post(
        PREFIX + "/nested-bind-srv/bindings",
        json={
            "endpoint_url": "https://mcp.example.com/nested-bind-srv",
            "server_version": "1.0",
        },
    )

    get_r = client.get(PREFIX + "/nested-bind-srv")
    assert get_r.status_code == 200
    assert get_r.json()["access_bindings"][0]["server_name"] == "nested-bind-srv"

    search_r = client.get(PREFIX, params={"filter_string": "name = 'nested-bind-srv'"})
    assert search_r.status_code == 200
    assert search_r.json()["mcp_servers"][0]["access_bindings"][0]["server_name"] == (
        "nested-bind-srv"
    )


def test_update_server(client):
    client.post(PREFIX, json={"name": "upd"})
    r = client.patch(PREFIX + "/upd", json={"description": "updated", "display_name": "Upd"})
    assert r.status_code == 200
    data = r.json()
    assert data["description"] == "updated"
    assert data["display_name"] == "Upd"


def test_delete_server(client):
    client.post(PREFIX, json={"name": "del"})
    r = client.delete(PREFIX + "/del")
    assert r.status_code == 200
    assert client.get(PREFIX + "/del").status_code == 404


def test_server_crud_with_slashed_name(client):
    name = "io.github.org/my-server"
    client.post(PREFIX, json={"name": name})
    r = client.get(PREFIX + "/" + _encode_path_param(name))
    assert r.status_code == 200
    assert r.json()["name"] == name


def test_version_crud_with_slashed_name(client):
    name = "io.github.org/server"
    sj = _server_json(name, "1.0")
    encoded_name = _encode_path_param(name)
    r = client.post(PREFIX + f"/{encoded_name}/versions", json={"server_json": sj})
    assert r.status_code == 200
    assert r.json()["name"] == name
    assert r.json()["version"] == "1.0"

    r = client.get(PREFIX + f"/{encoded_name}/versions/1.0")
    assert r.status_code == 200
    assert r.json()["version"] == "1.0"


def test_version_crud_with_slashed_name_and_version(client):
    name = "io.github.org/slash-server"
    version = "2025/06"
    encoded_name = _encode_path_param(name)
    encoded_version = _encode_path_param(version)
    sj = _server_json(name, version)
    r = client.post(
        PREFIX + f"/{encoded_name}/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["version"] == version

    r = client.get(PREFIX + f"/{encoded_name}/versions/{encoded_version}")
    assert r.status_code == 200
    assert r.json()["version"] == version

    r = client.patch(
        PREFIX + f"/{encoded_name}/versions/{encoded_version}",
        json={"display_name": "June 2025"},
    )
    assert r.status_code == 200
    assert r.json()["display_name"] == "June 2025"


def test_create_version(client):
    sj = _server_json("v-server", "1.0", title="Test")
    r = client.post(
        PREFIX + "/v-server/versions",
        json={"server_json": sj, "status": "active"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "v-server"
    assert data["version"] == "1.0"
    assert data["status"] == "active"
    assert data["server_json"]["title"] == "Test"


def test_create_version_name_mismatch(client):
    sj = _server_json("wrong-name", "1.0")
    r = client.post(PREFIX + "/v-server/versions", json={"server_json": sj})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "does not match" in r.json()["message"]


def test_create_version_missing_required_field_returns_mlflow_error(client):
    r = client.post(PREFIX + "/v-server/versions", json={"server_json": {"name": "v-server"}})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "server_json.version" in r.json()["message"]


def test_create_version_invalid_package_shape_returns_mlflow_error(client):
    r = client.post(
        PREFIX + "/pkg-server/versions",
        json={
            "server_json": {
                "name": "pkg-server",
                "version": "1.0",
                "packages": [{}],
            }
        },
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "server_json.packages.0.registryType" in r.json()["message"]


def test_create_version_with_tools(client):
    sj = _server_json("tools-server", "1.0")
    tools = [{"name": "web_search", "description": "Search the web"}]
    r = client.post(
        PREFIX + "/tools-server/versions",
        json={"server_json": sj, "tools": tools, "status": "active"},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["tools"]) == 1
    assert data["tools"][0]["name"] == "web_search"


def test_create_version_preserves_empty_tools_list(client):
    sj = _server_json("empty-tools-server", "1.0")
    r = client.post(
        PREFIX + "/empty-tools-server/versions",
        json={"server_json": sj, "tools": [], "status": "active"},
    )
    assert r.status_code == 200
    assert r.json()["tools"] == []


def test_get_version(client):
    sj = _server_json("gv", "2.0")
    client.post(PREFIX + "/gv/versions", json={"server_json": sj, "status": "active"})
    r = client.get(PREFIX + "/gv/versions/2.0")
    assert r.status_code == 200
    assert r.json()["version"] == "2.0"


def test_version_named_latest_round_trips_via_version_route(client):
    client.post(
        PREFIX + "/lat/versions",
        json={"server_json": _server_json("lat", "latest"), "status": "active"},
    )
    client.post(
        PREFIX + "/lat/versions",
        json={"server_json": _server_json("lat", "2.0"), "status": "active"},
    )
    client.patch(PREFIX + "/lat", json={"latest_version": "2.0"})

    latest_version_r = client.get(PREFIX + "/lat/versions/latest")
    assert latest_version_r.status_code == 200
    assert latest_version_r.json()["version"] == "latest"

    latest_alias_r = client.get(PREFIX + "/lat/aliases/latest")
    assert latest_alias_r.status_code == 200
    assert latest_alias_r.json()["version"] == "2.0"


def test_search_versions(client):
    for v in ["1.0", "2.0", "3.0"]:
        sj = _server_json("sv", v)
        client.post(PREFIX + "/sv/versions", json={"server_json": sj, "status": "active"})
    r = client.get(PREFIX + "/sv/versions", params={"max_results": 2})
    assert r.status_code == 200
    data = r.json()
    assert len(data["mcp_server_versions"]) == 2


def test_update_version_status(client):
    sj = _server_json("uv", "1.0")
    client.post(PREFIX + "/uv/versions", json={"server_json": sj})
    r = client.patch(PREFIX + "/uv/versions/1.0", json={"status": "active"})
    assert r.status_code == 200
    assert r.json()["status"] == "active"


def test_delete_version(client):
    sj = _server_json("dv", "1.0")
    client.post(PREFIX + "/dv/versions", json={"server_json": sj})
    r = client.delete(PREFIX + "/dv/versions/1.0")
    assert r.status_code == 200


def test_create_binding_with_version(client):
    sj = _server_json("bind-srv", "1.0")
    client.post(PREFIX + "/bind-srv/versions", json={"server_json": sj, "status": "active"})
    r = client.post(
        PREFIX + "/bind-srv/bindings",
        json={
            "endpoint_url": "https://mcp.example.com/bind-srv",
            "server_version": "1.0",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["server_name"] == "bind-srv"
    assert data["endpoint_url"] == "https://mcp.example.com/bind-srv"
    assert data["server_version"] == "1.0"


def test_create_binding_with_alias(client):
    sj = _server_json("alias-bind-srv", "1.0")
    client.post(PREFIX + "/alias-bind-srv/versions", json={"server_json": sj, "status": "active"})
    client.post(PREFIX + "/alias-bind-srv/aliases", json={"alias": "prod", "version": "1.0"})
    r = client.post(
        PREFIX + "/alias-bind-srv/bindings",
        json={
            "endpoint_url": "https://mcp.example.com/alias-bind-srv",
            "server_alias": "prod",
        },
    )
    assert r.status_code == 200
    assert r.json()["server_alias"] == "prod"


def test_get_binding_with_tools(client):
    sj = _server_json("bt-srv", "1.0")
    tools = [{"name": "tool1", "description": "A tool"}]
    client.post(
        PREFIX + "/bt-srv/versions",
        json={"server_json": sj, "status": "active", "tools": tools},
    )
    client.post(
        PREFIX + "/bt-srv/bindings",
        json={"endpoint_url": "https://mcp.example.com/bt-srv", "server_version": "1.0"},
    )
    r = client.get(PREFIX + "/bt-srv/bindings")
    assert r.status_code == 200
    bindings = r.json()["mcp_access_bindings"]
    assert len(bindings) == 1
    assert bindings[0]["tools"][0]["name"] == "tool1"


def test_get_binding_includes_resolved_version(client):
    sj = _server_json("brv-srv", "1.0")
    client.post(PREFIX + "/brv-srv/versions", json={"server_json": sj, "status": "active"})
    create_r = client.post(
        PREFIX + "/brv-srv/bindings",
        json={"endpoint_url": "https://mcp.example.com/brv-srv", "server_version": "1.0"},
    )
    bid = create_r.json()["binding_id"]
    r = client.get(PREFIX + f"/brv-srv/bindings/{bid}")
    assert r.status_code == 200
    assert r.json()["resolved_version"]["name"] == "brv-srv"
    assert r.json()["resolved_version"]["version"] == "1.0"


def test_get_binding_preserves_empty_tools_list(client):
    sj = _server_json("bt-empty-srv", "1.0")
    client.post(
        PREFIX + "/bt-empty-srv/versions",
        json={"server_json": sj, "status": "active", "tools": []},
    )
    client.post(
        PREFIX + "/bt-empty-srv/bindings",
        json={"endpoint_url": "https://mcp.example.com/bt-empty-srv", "server_version": "1.0"},
    )
    r = client.get(PREFIX + "/bt-empty-srv/bindings")
    assert r.status_code == 200
    bindings = r.json()["mcp_access_bindings"]
    assert len(bindings) == 1
    assert bindings[0]["tools"] == []


def test_search_bindings_workspace_wide(client):
    for name in ["ws-a", "ws-b"]:
        sj = _server_json(name, "1.0")
        client.post(PREFIX + f"/{name}/versions", json={"server_json": sj, "status": "active"})
        client.post(
            PREFIX + f"/{name}/bindings",
            json={"endpoint_url": f"https://mcp.example.com/{name}", "server_version": "1.0"},
        )
    r = client.get(PREFIX + "/bindings")
    assert r.status_code == 200
    assert len(r.json()["mcp_access_bindings"]) == 2


def test_search_bindings_server_scoped(client):
    for name in ["sc-a", "sc-b"]:
        sj = _server_json(name, "1.0")
        client.post(PREFIX + f"/{name}/versions", json={"server_json": sj, "status": "active"})
        client.post(
            PREFIX + f"/{name}/bindings",
            json={"endpoint_url": f"https://mcp.example.com/{name}", "server_version": "1.0"},
        )
    r = client.get(PREFIX + "/sc-a/bindings")
    assert r.status_code == 200
    assert len(r.json()["mcp_access_bindings"]) == 1
    assert r.json()["mcp_access_bindings"][0]["server_name"] == "sc-a"


def test_update_binding(client):
    sj = _server_json("ub-srv", "1.0")
    client.post(PREFIX + "/ub-srv/versions", json={"server_json": sj, "status": "active"})
    create_r = client.post(
        PREFIX + "/ub-srv/bindings",
        json={"endpoint_url": "https://old.example.com", "server_version": "1.0"},
    )
    bid = create_r.json()["binding_id"]
    r = client.patch(
        PREFIX + f"/ub-srv/bindings/{bid}",
        json={"endpoint_url": "https://new.example.com"},
    )
    assert r.status_code == 200
    assert r.json()["endpoint_url"] == "https://new.example.com"


def test_delete_binding(client):
    sj = _server_json("db-srv", "1.0")
    client.post(PREFIX + "/db-srv/versions", json={"server_json": sj, "status": "active"})
    create_r = client.post(
        PREFIX + "/db-srv/bindings",
        json={"endpoint_url": "https://mcp.example.com/db", "server_version": "1.0"},
    )
    bid = create_r.json()["binding_id"]
    r = client.delete(PREFIX + f"/db-srv/bindings/{bid}")
    assert r.status_code == 200
    assert client.get(PREFIX + f"/db-srv/bindings/{bid}").status_code == 404


def test_set_and_delete_server_tag(client):
    client.post(PREFIX, json={"name": "tag-srv"})
    r = client.post(PREFIX + "/tag-srv/tags", json={"key": "env", "value": "prod"})
    assert r.status_code == 200
    server = client.get(PREFIX + "/tag-srv").json()
    assert server["tags"]["env"] == "prod"

    r = client.delete(PREFIX + "/tag-srv/tags/env")
    assert r.status_code == 200
    server = client.get(PREFIX + "/tag-srv").json()
    assert "env" not in server["tags"]


def test_delete_server_tag_with_slash_key(client):
    client.post(PREFIX, json={"name": "slash-tag-srv"})
    encoded_key = _encode_path_param("team/platform")
    client.post(
        PREFIX + "/slash-tag-srv/tags",
        json={"key": "team/platform", "value": "prod"},
    )
    r = client.delete(PREFIX + f"/slash-tag-srv/tags/{encoded_key}")
    assert r.status_code == 200
    server = client.get(PREFIX + "/slash-tag-srv").json()
    assert "team/platform" not in server["tags"]


def test_set_and_delete_version_tag(client):
    sj = _server_json("vt-srv", "1.0")
    client.post(PREFIX + "/vt-srv/versions", json={"server_json": sj, "status": "active"})
    r = client.post(PREFIX + "/vt-srv/versions/1.0/tags", json={"key": "stage", "value": "beta"})
    assert r.status_code == 200
    ver = client.get(PREFIX + "/vt-srv/versions/1.0").json()
    assert ver["tags"]["stage"] == "beta"

    r = client.delete(PREFIX + "/vt-srv/versions/1.0/tags/stage")
    assert r.status_code == 200
    ver = client.get(PREFIX + "/vt-srv/versions/1.0").json()
    assert "stage" not in ver["tags"]


def test_delete_version_tag_with_slash_key(client):
    sj = _server_json("slash-vt-srv", "1.0")
    encoded_key = _encode_path_param("team/platform")
    client.post(PREFIX + "/slash-vt-srv/versions", json={"server_json": sj, "status": "active"})
    client.post(
        PREFIX + "/slash-vt-srv/versions/1.0/tags",
        json={"key": "team/platform", "value": "beta"},
    )
    r = client.delete(PREFIX + f"/slash-vt-srv/versions/1.0/tags/{encoded_key}")
    assert r.status_code == 200
    ver = client.get(PREFIX + "/slash-vt-srv/versions/1.0").json()
    assert "team/platform" not in ver["tags"]


def test_set_and_resolve_alias(client):
    sj = _server_json("alias-srv", "1.0")
    client.post(PREFIX + "/alias-srv/versions", json={"server_json": sj, "status": "active"})
    r = client.post(PREFIX + "/alias-srv/aliases", json={"alias": "prod", "version": "1.0"})
    assert r.status_code == 200

    r = client.get(PREFIX + "/alias-srv/aliases/prod")
    assert r.status_code == 200
    assert r.json()["version"] == "1.0"


def test_alias_with_slash_round_trips(client):
    sj = _server_json("slash-alias-srv", "1.0")
    encoded_alias = _encode_path_param("team/prod")
    client.post(PREFIX + "/slash-alias-srv/versions", json={"server_json": sj, "status": "active"})
    client.post(
        PREFIX + "/slash-alias-srv/aliases",
        json={"alias": "team/prod", "version": "1.0"},
    )
    r = client.get(PREFIX + f"/slash-alias-srv/aliases/{encoded_alias}")
    assert r.status_code == 200
    assert r.json()["version"] == "1.0"

    r = client.delete(PREFIX + f"/slash-alias-srv/aliases/{encoded_alias}")
    assert r.status_code == 200


def test_resolve_latest_alias(client):
    for v in ["1.0", "2.0"]:
        sj = _server_json("latest-srv", v)
        client.post(PREFIX + "/latest-srv/versions", json={"server_json": sj, "status": "active"})
    r = client.get(PREFIX + "/latest-srv/aliases/latest")
    assert r.status_code == 200
    assert r.json()["version"] == "2.0"


def test_delete_alias(client):
    sj = _server_json("da-srv", "1.0")
    client.post(PREFIX + "/da-srv/versions", json={"server_json": sj, "status": "active"})
    client.post(PREFIX + "/da-srv/aliases", json={"alias": "staging", "version": "1.0"})
    r = client.delete(PREFIX + "/da-srv/aliases/staging")
    assert r.status_code == 200


def test_missing_required_field(client):
    r = client.post(PREFIX, json={})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"


def test_invalid_server_json(client):
    r = client.post(PREFIX + "/x/versions", json={"server_json": {"name": "x"}})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"


def test_invalid_status_transition(client):
    sj = _server_json("is-srv", "1.0")
    client.post(PREFIX + "/is-srv/versions", json={"server_json": sj})
    r = client.patch(PREFIX + "/is-srv/versions/1.0", json={"status": "deprecated"})
    assert r.status_code == 400


def test_invalid_status_value(client):
    sj = _server_json("bad-status", "1.0")
    r = client.post(PREFIX + "/bad-status/versions", json={"server_json": sj, "status": "bogus"})
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Invalid status" in r.json()["message"]


def test_invalid_transport_type(client):
    sj = _server_json("bad-transport", "1.0")
    client.post(PREFIX + "/bad-transport/versions", json={"server_json": sj, "status": "active"})
    r = client.post(
        PREFIX + "/bad-transport/bindings",
        json={
            "endpoint_url": "https://example.com",
            "transport_type": "ftp",
            "server_version": "1.0",
        },
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Invalid transport_type" in r.json()["message"]


def test_server_response_has_aliases_as_list(client):
    sj = _server_json("shape-srv", "1.0")
    client.post(PREFIX + "/shape-srv/versions", json={"server_json": sj, "status": "active"})
    client.post(PREFIX + "/shape-srv/aliases", json={"alias": "prod", "version": "1.0"})
    server = client.get(PREFIX + "/shape-srv").json()
    assert isinstance(server["aliases"], list)
    assert server["aliases"][0]["alias"] == "prod"
    assert server["aliases"][0]["version"] == "1.0"


def test_server_response_includes_bindings(client):
    sj = _server_json("sb-srv", "1.0")
    client.post(PREFIX + "/sb-srv/versions", json={"server_json": sj, "status": "active"})
    client.post(
        PREFIX + "/sb-srv/bindings",
        json={"endpoint_url": "https://example.com/sb", "server_version": "1.0"},
    )
    server = client.get(PREFIX + "/sb-srv").json()
    assert len(server["access_bindings"]) == 1
    assert server["access_bindings"][0]["endpoint_url"] == "https://example.com/sb"


def test_server_response_includes_binding_resolved_version(client):
    sj = _server_json("sbrv-srv", "1.0")
    client.post(PREFIX + "/sbrv-srv/versions", json={"server_json": sj, "status": "active"})
    client.post(
        PREFIX + "/sbrv-srv/bindings",
        json={"endpoint_url": "https://example.com/sbrv", "server_version": "1.0"},
    )
    server = client.get(PREFIX + "/sbrv-srv").json()
    assert len(server["access_bindings"]) == 1
    assert server["access_bindings"][0]["resolved_version"]["version"] == "1.0"


def test_server_json_extra_fields_preserved(client):
    sj = _server_json("extra-srv", "1.0", custom_field="preserved")
    r = client.post(PREFIX + "/extra-srv/versions", json={"server_json": sj, "status": "active"})
    assert r.status_code == 200
    assert r.json()["server_json"]["custom_field"] == "preserved"


def test_server_json_explicit_nulls_preserved(client):
    sj = _server_json("null-srv", "1.0", description=None, custom_field=None)
    r = client.post(PREFIX + "/null-srv/versions", json={"server_json": sj, "status": "active"})
    assert r.status_code == 200
    assert "description" in r.json()["server_json"]
    assert r.json()["server_json"]["description"] is None
    assert "custom_field" in r.json()["server_json"]
    assert r.json()["server_json"]["custom_field"] is None
    assert "repository" not in r.json()["server_json"]
