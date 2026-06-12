"""Tests for mlflow.genai MCP server SDK functions.

Integration tests run against a real SQLAlchemy store (SQLite) so the full
round-trip (SDK -> MlflowClient -> store) is covered without needing an HTTP
server.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from unittest import mock

import pytest

from mlflow import genai
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus, MCPTool
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking.client import MlflowClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(f"sqlite:///{tmp_path / 'test.db'}", artifact_uri.as_uri())


@pytest.fixture(autouse=True)
def patch_store(store):
    with mock.patch(
        "mlflow.tracking._tracking_service.utils._get_store",
        return_value=store,
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server_json(name: str, version: str, **extra) -> dict:
    d = {"name": name, "version": version}
    d.update(extra)
    return d


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, n=-1):
        return self._payload[:n] if n >= 0 else self._payload


# ---------------------------------------------------------------------------
# register_mcp_server
# ---------------------------------------------------------------------------


def test_register_mcp_server_returns_version():
    sj = _server_json("io.github.test/brave-search", "1.0.0", description="Test")
    version = genai.register_mcp_server(server_json=sj)

    assert version.name == "io.github.test/brave-search"
    assert version.version == "1.0.0"
    assert version.status == MCPStatus.DRAFT


def test_register_mcp_server_auto_creates_parent_server():
    sj = _server_json("io.github.test/auto-server", "1.0.0")
    genai.register_mcp_server(server_json=sj)

    server = genai.get_mcp_server(name="io.github.test/auto-server")
    assert server.name == "io.github.test/auto-server"


def test_register_mcp_server_with_status_active():
    sj = _server_json("io.github.test/active-server", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    assert version.status == MCPStatus.ACTIVE


def test_register_mcp_server_creates_bindings_from_remotes():
    sj = _server_json(
        "io.github.test/remote-server",
        "1.0.0",
        remotes=[
            {"type": "streamable-http", "url": "https://mcp.example.com/server"},
        ],
    )
    version = genai.register_mcp_server(
        server_json=sj, create_access_bindings_from_remotes=True
    )

    bindings = genai.search_mcp_access_bindings(server_name=version.name)
    assert len(bindings) == 1
    assert bindings[0].endpoint_url == "https://mcp.example.com/server"
    assert bindings[0].server_version == version.version


def test_register_mcp_server_no_bindings_when_flag_false():
    sj = _server_json(
        "io.github.test/no-binding-server",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/nb"}],
    )
    genai.register_mcp_server(server_json=sj, create_access_bindings_from_remotes=False)

    bindings = genai.search_mcp_access_bindings(server_name="io.github.test/no-binding-server")
    assert len(bindings) == 0


def test_register_mcp_server_skips_remotes_without_url():
    sj = _server_json(
        "io.github.test/no-url-remote",
        "1.0.0",
        remotes=[{"type": "streamable-http"}],
    )
    version = genai.register_mcp_server(
        server_json=sj, create_access_bindings_from_remotes=True
    )
    bindings = genai.search_mcp_access_bindings(server_name=version.name)
    assert len(bindings) == 0


def test_register_mcp_server_falls_back_on_unknown_transport():
    sj = _server_json(
        "io.github.test/unknown-transport",
        "1.0.0",
        remotes=[{"type": "grpc-bidirectional", "url": "https://mcp.example.com/grpc"}],
    )
    version = genai.register_mcp_server(
        server_json=sj, create_access_bindings_from_remotes=True
    )
    bindings = genai.search_mcp_access_bindings(server_name=version.name)
    assert len(bindings) == 1
    assert bindings[0].transport_type == MCPRemoteTransportType.STREAMABLE_HTTP


# ---------------------------------------------------------------------------
# register_mcp_server_from_url
# ---------------------------------------------------------------------------


def test_register_mcp_server_from_url():
    payload = b'{"name": "io.github.test/url-server", "version": "2.0.0"}'
    with mock.patch.object(
        urllib.request, "urlopen", return_value=_FakeResponse(payload)
    ) as mock_urlopen:
        version = genai.register_mcp_server_from_url(url="https://example.com/server.json")

    mock_urlopen.assert_called_once_with("https://example.com/server.json", timeout=30)
    assert version.name == "io.github.test/url-server"
    assert version.version == "2.0.0"


def test_register_mcp_server_from_url_sets_source_from_url():
    payload = b'{"name": "io.github.test/src-server", "version": "1.0.0"}'
    url = "https://example.com/src.json"
    with mock.patch.object(
        urllib.request, "urlopen", return_value=_FakeResponse(payload)
    ) as mock_urlopen:
        version = genai.register_mcp_server_from_url(url=url)

    mock_urlopen.assert_called_once_with(url, timeout=30)
    assert version.source == url


def test_register_mcp_server_from_url_invalid_scheme():
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException, match="http or https"):
        genai.register_mcp_server_from_url(url="ftp://example.com/server.json")


# ---------------------------------------------------------------------------
# create / get / search / update / delete MCPServer
# ---------------------------------------------------------------------------


def test_create_mcp_server():
    server = genai.create_mcp_server(name="io.github.test/my-server", description="Test")
    assert server.name == "io.github.test/my-server"
    assert server.description == "Test"


def test_get_mcp_server():
    genai.create_mcp_server(name="io.github.test/get-server")
    server = genai.get_mcp_server(name="io.github.test/get-server")
    assert server.name == "io.github.test/get-server"


def test_search_mcp_servers_returns_all():
    genai.create_mcp_server(name="io.github.test/search-a")
    genai.create_mcp_server(name="io.github.test/search-b")
    results = genai.search_mcp_servers()
    names = [s.name for s in results]
    assert "io.github.test/search-a" in names
    assert "io.github.test/search-b" in names


def test_search_mcp_servers_filter_by_status():
    sj = _server_json("io.github.test/status-active", "1.0.0")
    genai.register_mcp_server(server_json=sj, status="active")
    genai.create_mcp_server(name="io.github.test/no-version-server")

    results = genai.search_mcp_servers(filter_string="status = 'active'")
    assert any(s.name == "io.github.test/status-active" for s in results)
    assert all(s.status == MCPStatus.ACTIVE for s in results if s.status is not None)


def test_update_mcp_server():
    genai.create_mcp_server(name="io.github.test/upd-server", description="old")
    server = genai.update_mcp_server(name="io.github.test/upd-server", description="new")
    assert server.description == "new"


def test_delete_mcp_server():
    from mlflow.exceptions import MlflowException

    genai.create_mcp_server(name="io.github.test/del-server")
    genai.delete_mcp_server(name="io.github.test/del-server")
    with pytest.raises(MlflowException, match="MCP server .* not found"):
        genai.get_mcp_server(name="io.github.test/del-server")


# ---------------------------------------------------------------------------
# MCPServerVersion CRUD
# ---------------------------------------------------------------------------


def test_create_mcp_server_version():
    sj = _server_json("io.github.test/ver-server", "1.0.0")
    version = genai.create_mcp_server_version(server_json=sj)
    assert version.version == "1.0.0"
    assert version.status == MCPStatus.DRAFT


def test_create_mcp_server_version_does_not_create_bindings():
    sj = _server_json(
        "io.github.test/ver-no-bind",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/x"}],
    )
    version = genai.create_mcp_server_version(server_json=sj)
    bindings = genai.search_mcp_access_bindings(server_name=version.name)
    assert len(bindings) == 0


def test_get_mcp_server_version():
    sj = _server_json("io.github.test/get-ver", "1.0.0")
    genai.register_mcp_server(server_json=sj)
    v = genai.get_mcp_server_version(name="io.github.test/get-ver", version="1.0.0")
    assert v.version == "1.0.0"


def test_get_latest_mcp_server_version():
    sj1 = _server_json("io.github.test/latest-ver", "1.0.0")
    sj2 = _server_json("io.github.test/latest-ver", "2.0.0")
    genai.register_mcp_server(server_json=sj1, status="active")
    genai.register_mcp_server(server_json=sj2, status="active")
    latest = genai.get_latest_mcp_server_version(name="io.github.test/latest-ver")
    assert latest.version == "2.0.0"


def test_get_mcp_server_version_by_alias():
    sj = _server_json("io.github.test/alias-ver", "1.0.0")
    genai.register_mcp_server(server_json=sj, status="active")
    genai.set_mcp_server_alias(
        name="io.github.test/alias-ver", alias="production", version="1.0.0"
    )
    v = genai.get_mcp_server_version_by_alias(name="io.github.test/alias-ver", alias="production")
    assert v.version == "1.0.0"


def test_get_mcp_server_version_by_alias_latest():
    sj = _server_json("io.github.test/latest-alias", "1.0.0")
    genai.register_mcp_server(server_json=sj, status="active")
    v = genai.get_mcp_server_version_by_alias(name="io.github.test/latest-alias", alias="latest")
    assert v.version == "1.0.0"


def test_search_mcp_server_versions():
    sj1 = _server_json("io.github.test/search-ver", "1.0.0")
    sj2 = _server_json("io.github.test/search-ver", "2.0.0")
    genai.register_mcp_server(server_json=sj1)
    genai.register_mcp_server(server_json=sj2)
    versions = genai.search_mcp_server_versions(name="io.github.test/search-ver")
    assert len(versions) == 2


def test_update_mcp_server_version_status():
    sj = _server_json("io.github.test/upd-ver", "1.0.0")
    genai.register_mcp_server(server_json=sj)
    v = genai.update_mcp_server_version(
        name="io.github.test/upd-ver", version="1.0.0", status="active"
    )
    assert v.status == MCPStatus.ACTIVE


def test_update_mcp_server_version_tools():
    sj = _server_json("io.github.test/tools-ver", "1.0.0")
    genai.register_mcp_server(server_json=sj)
    tools = [MCPTool(name="search", description="search the web")]
    v = genai.update_mcp_server_version(
        name="io.github.test/tools-ver", version="1.0.0", tools=tools
    )
    assert v.tools is not None
    assert v.tools[0].name == "search"


def test_delete_mcp_server_version():
    from mlflow.exceptions import MlflowException

    sj = _server_json("io.github.test/del-ver", "1.0.0")
    genai.register_mcp_server(server_json=sj)
    genai.delete_mcp_server_version(name="io.github.test/del-ver", version="1.0.0")
    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_server_version(name="io.github.test/del-ver", version="1.0.0")


# ---------------------------------------------------------------------------
# MCPAccessBinding CRUD
# ---------------------------------------------------------------------------


def test_create_and_get_mcp_access_binding():
    sj = _server_json("io.github.test/bind-server", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    binding = genai.create_mcp_access_binding(
        server_name=version.name,
        endpoint_url="https://mcp.example.com/server",
        transport_type="streamable-http",
        server_version=version.version,
    )
    assert binding.endpoint_url == "https://mcp.example.com/server"
    assert binding.transport_type == MCPRemoteTransportType.STREAMABLE_HTTP

    fetched = genai.get_mcp_access_binding(
        server_name=version.name, binding_id=binding.binding_id
    )
    assert fetched.binding_id == binding.binding_id


def test_create_mcp_access_binding_via_alias():
    sj = _server_json("io.github.test/alias-bind", "1.0.0")
    genai.register_mcp_server(server_json=sj, status="active")
    genai.set_mcp_server_alias(
        name="io.github.test/alias-bind", alias="prod", version="1.0.0"
    )
    binding = genai.create_mcp_access_binding(
        server_name="io.github.test/alias-bind",
        endpoint_url="https://mcp.example.com/ab",
        server_alias="prod",
    )
    assert binding.server_alias == "prod"
    assert binding.server_version is None


def test_search_mcp_access_bindings():
    sj = _server_json("io.github.test/search-bind", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    genai.create_mcp_access_binding(
        server_name=version.name,
        endpoint_url="https://a.example.com",
        server_version=version.version,
    )
    genai.create_mcp_access_binding(
        server_name=version.name,
        endpoint_url="https://b.example.com",
        server_version=version.version,
    )
    bindings = genai.search_mcp_access_bindings(server_name=version.name)
    assert len(bindings) == 2


def test_update_mcp_access_binding():
    sj = _server_json("io.github.test/upd-bind", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    binding = genai.create_mcp_access_binding(
        server_name=version.name,
        endpoint_url="https://old.example.com",
        server_version=version.version,
    )
    updated = genai.update_mcp_access_binding(
        server_name=version.name,
        binding_id=binding.binding_id,
        endpoint_url="https://new.example.com",
    )
    assert updated.endpoint_url == "https://new.example.com"


def test_delete_mcp_access_binding():
    from mlflow.exceptions import MlflowException

    sj = _server_json("io.github.test/del-bind", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    binding = genai.create_mcp_access_binding(
        server_name=version.name,
        endpoint_url="https://del.example.com",
        server_version=version.version,
    )
    genai.delete_mcp_access_binding(
        server_name=version.name, binding_id=binding.binding_id
    )
    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_access_binding(
            server_name=version.name, binding_id=binding.binding_id
        )


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_set_and_delete_mcp_server_tag():
    genai.create_mcp_server(name="io.github.test/tag-server")
    genai.set_mcp_server_tag(name="io.github.test/tag-server", key="env", value="prod")
    server = genai.get_mcp_server(name="io.github.test/tag-server")
    assert server.tags.get("env") == "prod"

    genai.delete_mcp_server_tag(name="io.github.test/tag-server", key="env")
    server = genai.get_mcp_server(name="io.github.test/tag-server")
    assert "env" not in server.tags


def test_set_and_delete_mcp_server_version_tag():
    sj = _server_json("io.github.test/tag-ver", "1.0.0")
    genai.register_mcp_server(server_json=sj)
    genai.set_mcp_server_version_tag(
        name="io.github.test/tag-ver", version="1.0.0", key="stage", value="beta"
    )
    v = genai.get_mcp_server_version(name="io.github.test/tag-ver", version="1.0.0")
    assert v.tags.get("stage") == "beta"

    genai.delete_mcp_server_version_tag(
        name="io.github.test/tag-ver", version="1.0.0", key="stage"
    )
    v = genai.get_mcp_server_version(name="io.github.test/tag-ver", version="1.0.0")
    assert "stage" not in v.tags


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_set_and_delete_mcp_server_alias():
    sj = _server_json("io.github.test/alias-server", "1.0.0")
    genai.register_mcp_server(server_json=sj, status="active")
    genai.set_mcp_server_alias(
        name="io.github.test/alias-server", alias="production", version="1.0.0"
    )
    server = genai.get_mcp_server(name="io.github.test/alias-server")
    assert server.aliases.get("production") == "1.0.0"

    genai.delete_mcp_server_alias(name="io.github.test/alias-server", alias="production")
    server = genai.get_mcp_server(name="io.github.test/alias-server")
    assert "production" not in server.aliases


def test_set_latest_alias_raises():
    from mlflow.exceptions import MlflowException

    sj = _server_json("io.github.test/reserved-alias", "1.0.0")
    genai.register_mcp_server(server_json=sj, status="active")
    with pytest.raises(MlflowException, match="reserved"):
        genai.set_mcp_server_alias(
            name="io.github.test/reserved-alias", alias="latest", version="1.0.0"
        )


# ---------------------------------------------------------------------------
# MlflowClient direct usage
# ---------------------------------------------------------------------------


def test_mlflow_client_create_and_get_server():
    client = MlflowClient()
    server = client.create_mcp_server(name="io.github.test/client-server", description="via client")
    assert server.name == "io.github.test/client-server"

    fetched = client.get_mcp_server("io.github.test/client-server")
    assert fetched.description == "via client"


def test_mlflow_client_version_lifecycle():
    client = MlflowClient()
    sj = _server_json("io.github.test/lifecycle-ver", "1.0.0")
    v = client.create_mcp_server_version(server_json=sj)
    assert v.status == MCPStatus.DRAFT

    v2 = client.update_mcp_server_version(
        name="io.github.test/lifecycle-ver",
        version="1.0.0",
        status=MCPStatus.ACTIVE,
    )
    assert v2.status == MCPStatus.ACTIVE

    client.update_mcp_server_version(
        name="io.github.test/lifecycle-ver",
        version="1.0.0",
        status=MCPStatus.DEPRECATED,
    )
    client.delete_mcp_server_version(name="io.github.test/lifecycle-ver", version="1.0.0")

    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException, match="not found"):
        client.get_mcp_server_version(name="io.github.test/lifecycle-ver", version="1.0.0")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_all_functions_exported():
    import mlflow.genai

    expected = [
        "register_mcp_server",
        "register_mcp_server_from_url",
        "create_mcp_server",
        "get_mcp_server",
        "search_mcp_servers",
        "update_mcp_server",
        "delete_mcp_server",
        "create_mcp_server_version",
        "get_mcp_server_version",
        "get_mcp_server_version_by_alias",
        "get_latest_mcp_server_version",
        "search_mcp_server_versions",
        "update_mcp_server_version",
        "delete_mcp_server_version",
        "create_mcp_access_binding",
        "get_mcp_access_binding",
        "search_mcp_access_bindings",
        "update_mcp_access_binding",
        "delete_mcp_access_binding",
        "set_mcp_server_tag",
        "delete_mcp_server_tag",
        "set_mcp_server_version_tag",
        "delete_mcp_server_version_tag",
        "set_mcp_server_alias",
        "delete_mcp_server_alias",
    ]
    for name in expected:
        assert name in mlflow.genai.__all__
        assert hasattr(mlflow.genai, name)
