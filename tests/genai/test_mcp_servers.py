from __future__ import annotations

import urllib.request
from pathlib import Path
from unittest import mock

import pytest

from mlflow import genai
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus, MCPTool
from mlflow.exceptions import MlflowException
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server_json(name: str, version: str, **extra) -> dict[str, str]:
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


def test_register_mcp_server_creates_endpoints_from_remotes():
    sj = _server_json(
        "io.github.test/remote-server",
        "1.0.0",
        remotes=[
            {"type": "streamable-http", "url": "https://mcp.example.com/server"},
        ],
    )
    version = genai.register_mcp_server(
        server_json=sj, status="active", create_access_endpoints_from_remotes=True
    )

    endpoints = genai.search_mcp_access_endpoints(server_name=version.name)
    assert len(endpoints) == 1
    assert endpoints[0].url == "https://mcp.example.com/server"
    assert endpoints[0].server_version == version.version


def test_register_mcp_server_no_endpoints_when_flag_false():
    sj = _server_json(
        "io.github.test/no-endpoint-server",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/nb"}],
    )
    genai.register_mcp_server(
        server_json=sj, status="active", create_access_endpoints_from_remotes=False
    )

    endpoints = genai.search_mcp_access_endpoints(server_name="io.github.test/no-endpoint-server")
    assert len(endpoints) == 0


@pytest.mark.parametrize("status", ["draft", None])
def test_register_mcp_server_rejects_auto_endpoints_for_inactive_initial_status(status):
    kwargs = {
        "server_json": _server_json(
            f"io.github.test/draft-no-bind-{status}",
            "1.0.0",
            remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/draft"}],
        ),
        "create_access_endpoints_from_remotes": True,
    }
    if status is not None:
        kwargs["status"] = status
    with pytest.raises(
        MlflowException, match="create_access_endpoints_from_remotes=True requires status='active'"
    ):
        genai.register_mcp_server(**kwargs)

    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_server(name=kwargs["server_json"]["name"])


def test_register_mcp_server_creates_endpoints_for_active_status():
    sj = _server_json(
        "io.github.test/published-bind-active",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/pub"}],
    )
    version = genai.register_mcp_server(
        server_json=sj, status="active", create_access_endpoints_from_remotes=True
    )

    endpoints = genai.search_mcp_access_endpoints(server_name=version.name)
    assert len(endpoints) == 1


@pytest.mark.parametrize("status", ["deprecated", "deleted"])
def test_register_mcp_server_rejects_non_initial_statuses(status):
    sj = _server_json(f"io.github.test/reject-status-{status}", "1.0.0")

    with pytest.raises(
        MlflowException, match="Initial MCP server registration status must be 'draft' or 'active'"
    ):
        genai.register_mcp_server(server_json=sj, status=status)


def test_register_mcp_server_skips_remotes_without_url():
    sj = _server_json(
        "io.github.test/no-url-remote",
        "1.0.0",
        remotes=[{"type": "streamable-http"}],
    )
    version = genai.register_mcp_server(
        server_json=sj, status="active", create_access_endpoints_from_remotes=True
    )
    endpoints = genai.search_mcp_access_endpoints(server_name=version.name)
    assert len(endpoints) == 0


def test_register_mcp_server_accepts_null_remotes():
    sj = _server_json("io.github.test/null-remotes", "1.0.0", remotes=None)
    version = genai.register_mcp_server(
        server_json=sj, status="active", create_access_endpoints_from_remotes=True
    )

    endpoints = genai.search_mcp_access_endpoints(server_name=version.name)
    assert len(endpoints) == 0


@pytest.mark.parametrize("url", [123, True])
def test_register_mcp_server_rejects_remote_with_non_string_url(url):
    sj = _server_json(
        "io.github.test/bad-remote-url-type",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": url}],
    )
    with pytest.raises(MlflowException, match="remote.url"):
        genai.register_mcp_server(
            server_json=sj, status="active", create_access_endpoints_from_remotes=True
        )


def test_register_mcp_server_rejects_remote_with_blank_url():
    sj = _server_json(
        "io.github.test/bad-remote-url-blank",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "   "}],
    )
    with pytest.raises(MlflowException, match="remote.url"):
        genai.register_mcp_server(
            server_json=sj, status="active", create_access_endpoints_from_remotes=True
        )


def test_register_mcp_server_defaults_null_remote_type_to_streamable_http():
    sj = _server_json(
        "io.github.test/null-remote-type",
        "1.0.0",
        remotes=[{"type": None, "url": "https://mcp.example.com/default-type"}],
    )
    version = genai.register_mcp_server(
        server_json=sj, status="active", create_access_endpoints_from_remotes=True
    )

    endpoints = genai.search_mcp_access_endpoints(server_name=version.name)
    assert len(endpoints) == 1
    assert endpoints[0].transport_type == MCPRemoteTransportType.STREAMABLE_HTTP


def test_register_mcp_server_rejects_non_list_remotes():
    sj = _server_json("io.github.test/bad-remotes-shape", "1.0.0", remotes="not-a-list")
    with pytest.raises(MlflowException, match="server_json.remotes"):
        genai.register_mcp_server(
            server_json=sj, status="active", create_access_endpoints_from_remotes=True
        )

    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_server(name="io.github.test/bad-remotes-shape")


def test_register_mcp_server_rejects_non_object_remote_entries():
    sj = _server_json("io.github.test/bad-remote-entry", "1.0.0", remotes=[None])
    with pytest.raises(MlflowException, match="server_json.remotes entry"):
        genai.register_mcp_server(
            server_json=sj, status="active", create_access_endpoints_from_remotes=True
        )

    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_server(name="io.github.test/bad-remote-entry")


def test_register_mcp_server_rejects_unknown_transport():
    sj = _server_json(
        "io.github.test/unknown-transport",
        "1.0.0",
        remotes=[{"type": "grpc-bidirectional", "url": "https://mcp.example.com/grpc"}],
    )
    with pytest.raises(MlflowException, match="Invalid transport_type"):
        genai.register_mcp_server(
            server_json=sj, status="active", create_access_endpoints_from_remotes=True
        )

    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_server(name="io.github.test/unknown-transport")


def test_register_mcp_server_validates_all_remotes_before_creating():
    sj = _server_json(
        "io.github.test/partial-bad-remotes",
        "1.0.0",
        remotes=[
            {"type": "streamable-http", "url": "https://mcp.example.com/good"},
            {"type": "bad-transport", "url": "https://mcp.example.com/bad"},
        ],
    )
    with pytest.raises(MlflowException, match="Invalid transport_type"):
        genai.register_mcp_server(
            server_json=sj, status="active", create_access_endpoints_from_remotes=True
        )

    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_server(name="io.github.test/partial-bad-remotes")


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


def test_register_mcp_server_from_url_local_file(tmp_path: Path):
    server_file = tmp_path / "server.json"
    server_file.write_text('{"name": "io.github.test/local-server", "version": "3.0.0"}')

    version = genai.register_mcp_server_from_url(url=str(server_file))

    assert version.name == "io.github.test/local-server"
    assert version.version == "3.0.0"
    assert version.source == str(server_file)


def test_register_mcp_server_from_url_file_uri(tmp_path: Path):
    server_file = tmp_path / "server.json"
    server_file.write_text('{"name": "io.github.test/file-uri-server", "version": "4.0.0"}')

    file_uri = server_file.as_uri()
    version = genai.register_mcp_server_from_url(url=file_uri)

    assert version.name == "io.github.test/file-uri-server"
    assert version.version == "4.0.0"
    assert version.source == file_uri


def test_register_mcp_server_from_url_sanitizes_source():
    payload = b'{"name": "io.github.test/sanitize-server", "version": "1.0.0"}'
    url = "https://user:token@example.com:8080/server.json?sig=secret&expires=123#frag"
    with mock.patch.object(
        urllib.request, "urlopen", return_value=_FakeResponse(payload)
    ) as mock_urlopen:
        version = genai.register_mcp_server_from_url(url=url)

    mock_urlopen.assert_called_once_with(url, timeout=30)
    assert version.source == "https://example.com:8080/server.json"


def test_register_mcp_server_from_url_sanitizes_ipv6_source():
    payload = b'{"name": "io.github.test/ipv6-server", "version": "1.0.0"}'
    url = "https://user:token@[::1]:8080/server.json?sig=secret&expires=123#frag"
    with mock.patch.object(
        urllib.request, "urlopen", return_value=_FakeResponse(payload)
    ) as mock_urlopen:
        version = genai.register_mcp_server_from_url(url=url)

    mock_urlopen.assert_called_once_with(url, timeout=30)
    assert version.source == "https://[::1]:8080/server.json"


def test_register_mcp_server_from_url_explicit_source_preserved():
    payload = b'{"name": "io.github.test/explicit-src", "version": "1.0.0"}'
    url = "https://user:pass@example.com/server.json?token=secret"
    with mock.patch.object(
        urllib.request, "urlopen", return_value=_FakeResponse(payload)
    ) as mock_urlopen:
        version = genai.register_mcp_server_from_url(url=url, source="https://github.com/org/repo")

    mock_urlopen.assert_called_once_with(url, timeout=30)
    assert version.source == "https://github.com/org/repo"


def test_register_mcp_server_from_url_local_file_not_found(tmp_path: Path):
    missing = str(tmp_path / "does_not_exist.json")
    with pytest.raises(MlflowException, match="not found"):
        genai.register_mcp_server_from_url(url=missing)


def test_register_mcp_server_from_url_local_file_invalid_json(tmp_path: Path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json at all")
    with pytest.raises(MlflowException, match="valid JSON"):
        genai.register_mcp_server_from_url(url=str(bad_file))


def test_register_mcp_server_from_url_invalid_scheme():
    with pytest.raises(MlflowException, match="http, https, or file"):
        genai.register_mcp_server_from_url(url="ftp://example.com/server.json")


# ---------------------------------------------------------------------------
# create / get / search / update / delete MCPServer
# ---------------------------------------------------------------------------


def test_get_mcp_server():
    sj = _server_json("io.github.test/get-server", "1.0.0")
    genai.register_mcp_server(server_json=sj)
    server = genai.get_mcp_server(name="io.github.test/get-server")
    assert server.name == "io.github.test/get-server"


def test_search_mcp_servers_returns_all():
    genai.register_mcp_server(server_json=_server_json("io.github.test/search-a", "1.0.0"))
    genai.register_mcp_server(server_json=_server_json("io.github.test/search-b", "1.0.0"))
    results = genai.search_mcp_servers()
    names = [s.name for s in results]
    assert "io.github.test/search-a" in names
    assert "io.github.test/search-b" in names


def test_search_mcp_servers_filter_by_status():
    sj = _server_json("io.github.test/status-active", "1.0.0")
    genai.register_mcp_server(server_json=sj, status="active")
    genai.register_mcp_server(server_json=_server_json("io.github.test/no-version-server", "1.0.0"))

    results = genai.search_mcp_servers(filter_string="status = 'active'")
    assert any(s.name == "io.github.test/status-active" for s in results)
    assert all(s.status == MCPStatus.ACTIVE for s in results if s.status is not None)


def test_update_mcp_server():
    genai.register_mcp_server(
        server_json=_server_json("io.github.test/upd-server", "1.0.0", description="old")
    )
    server = genai.update_mcp_server(name="io.github.test/upd-server", description="new")
    assert server.description == "new"


def test_delete_mcp_server():
    genai.register_mcp_server(server_json=_server_json("io.github.test/del-server", "1.0.0"))
    genai.delete_mcp_server(name="io.github.test/del-server")
    with pytest.raises(MlflowException, match="MCP server .* not found"):
        genai.get_mcp_server(name="io.github.test/del-server")


def test_delete_mcp_server_rejects_active_version():
    name = "io.github.test/del-active-server"
    genai.register_mcp_server(server_json=_server_json(name, "1.0.0"), status="active")

    with pytest.raises(MlflowException, match="active version"):
        genai.delete_mcp_server(name=name)

    assert genai.get_mcp_server(name=name).name == name


# ---------------------------------------------------------------------------
# MCPServerVersion CRUD
# ---------------------------------------------------------------------------


def test_client_create_mcp_server_version():
    client = MlflowClient()
    sj = _server_json("io.github.test/ver-server", "1.0.0")
    version = client.create_mcp_server_version(server_json=sj)
    assert version.version == "1.0.0"
    assert version.status == MCPStatus.DRAFT


def test_client_create_mcp_server_version_does_not_create_endpoints():
    client = MlflowClient()
    sj = _server_json(
        "io.github.test/ver-no-bind",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/x"}],
    )
    version = client.create_mcp_server_version(server_json=sj)
    endpoints = genai.search_mcp_access_endpoints(server_name=version.name)
    assert len(endpoints) == 0


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
    genai.set_mcp_server_alias(name="io.github.test/alias-ver", alias="production", version="1.0.0")
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
    sj = _server_json("io.github.test/del-ver", "1.0.0")
    genai.register_mcp_server(server_json=sj)
    genai.delete_mcp_server_version(name="io.github.test/del-ver", version="1.0.0")
    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_server_version(name="io.github.test/del-ver", version="1.0.0")


def test_delete_mcp_server_version_cascades_to_endpoints():
    sj = _server_json(
        "io.github.test/cascade-del",
        "1.0.0",
        remotes=[
            {"type": "streamable-http", "url": "https://mcp.example.com/a"},
            {"type": "streamable-http", "url": "https://mcp.example.com/b"},
        ],
    )
    genai.register_mcp_server(
        server_json=sj, status="active", create_access_endpoints_from_remotes=True
    )
    endpoints = genai.search_mcp_access_endpoints(server_name="io.github.test/cascade-del")
    assert len(endpoints) == 2

    genai.update_mcp_server_version(
        name="io.github.test/cascade-del", version="1.0.0", status="deprecated"
    )
    genai.delete_mcp_server_version(name="io.github.test/cascade-del", version="1.0.0")

    endpoints = genai.search_mcp_access_endpoints(server_name="io.github.test/cascade-del")
    assert len(endpoints) == 0


# ---------------------------------------------------------------------------
# MCPAccessEndpoint CRUD
# ---------------------------------------------------------------------------


def test_create_and_get_mcp_access_endpoint():
    sj = _server_json("io.github.test/bind-server", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    endpoint = genai.create_mcp_access_endpoint(
        server_name=version.name,
        url="https://mcp.example.com/server",
        transport_type="streamable-http",
        server_version=version.version,
    )
    assert endpoint.url == "https://mcp.example.com/server"
    assert endpoint.transport_type == MCPRemoteTransportType.STREAMABLE_HTTP

    fetched = genai.get_mcp_access_endpoint(server_name=version.name, endpoint_id=endpoint.id)
    assert fetched.id == endpoint.id


def test_create_mcp_access_endpoint_via_alias():
    sj = _server_json("io.github.test/alias-bind", "1.0.0")
    genai.register_mcp_server(server_json=sj, status="active")
    genai.set_mcp_server_alias(name="io.github.test/alias-bind", alias="prod", version="1.0.0")
    endpoint = genai.create_mcp_access_endpoint(
        server_name="io.github.test/alias-bind",
        url="https://mcp.example.com/ab",
        server_alias="prod",
    )
    assert endpoint.server_alias == "prod"
    assert endpoint.server_version is None


def test_search_mcp_access_endpoints():
    sj = _server_json("io.github.test/search-bind", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    genai.create_mcp_access_endpoint(
        server_name=version.name,
        url="https://a.example.com",
        server_version=version.version,
    )
    genai.create_mcp_access_endpoint(
        server_name=version.name,
        url="https://b.example.com",
        server_version=version.version,
    )
    endpoints = genai.search_mcp_access_endpoints(server_name=version.name)
    assert len(endpoints) == 2


def test_update_mcp_access_endpoint():
    sj = _server_json("io.github.test/upd-bind", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    endpoint = genai.create_mcp_access_endpoint(
        server_name=version.name,
        url="https://old.example.com",
        server_version=version.version,
    )
    updated = genai.update_mcp_access_endpoint(
        server_name=version.name,
        endpoint_id=endpoint.id,
        url="https://new.example.com",
    )
    assert updated.url == "https://new.example.com"


def test_delete_mcp_access_endpoint():
    sj = _server_json("io.github.test/del-bind", "1.0.0")
    version = genai.register_mcp_server(server_json=sj, status="active")
    endpoint = genai.create_mcp_access_endpoint(
        server_name=version.name,
        url="https://del.example.com",
        server_version=version.version,
    )
    genai.delete_mcp_access_endpoint(server_name=version.name, endpoint_id=endpoint.id)
    with pytest.raises(MlflowException, match="not found"):
        genai.get_mcp_access_endpoint(server_name=version.name, endpoint_id=endpoint.id)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_set_and_delete_mcp_server_tag():
    genai.register_mcp_server(server_json=_server_json("io.github.test/tag-server", "1.0.0"))
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

    genai.delete_mcp_server_version_tag(name="io.github.test/tag-ver", version="1.0.0", key="stage")
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


def test_mlflow_client_create_server_rejects_risky_icons():
    client = MlflowClient()
    with pytest.raises(MlflowException, match="Icon URL"):
        client.create_mcp_server(
            name="io.github.test/client-icon-server",
            icons=[{"src": "https://127.0.0.1/icon.png"}],
        )


def test_mlflow_client_create_server_rejects_too_many_icons():
    client = MlflowClient()
    with pytest.raises(MlflowException, match="at most 100 items"):
        client.create_mcp_server(
            name="io.github.test/client-too-many-icons",
            icons=[{"src": f"https://example.com/icon-{i}.png"} for i in range(101)],
        )


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

    with pytest.raises(MlflowException, match="not found"):
        client.get_mcp_server_version(name="io.github.test/lifecycle-ver", version="1.0.0")


def test_mlflow_client_create_version_rejects_risky_server_json_icons():
    client = MlflowClient()
    with pytest.raises(MlflowException, match="Icon URL"):
        client.create_mcp_server_version(
            server_json=_server_json(
                "io.github.test/client-server-json-icons",
                "1.0.0",
                icons=[{"src": "https://127.0.0.1/icon.png"}],
            )
        )


@pytest.mark.parametrize("status", [MCPStatus.DEPRECATED, MCPStatus.DELETED])
def test_mlflow_client_create_version_rejects_non_initial_statuses(status):
    client = MlflowClient()
    with pytest.raises(
        MlflowException,
        match="Initial MCP server registration status must be 'draft' or 'active'",
    ):
        client.create_mcp_server_version(
            server_json=_server_json(f"io.github.test/client-status-{status.value}", "1.0.0"),
            status=status,
        )


def test_mlflow_client_create_version_rejects_too_many_tools():
    client = MlflowClient()
    with pytest.raises(MlflowException, match="at most 1000 items"):
        client.create_mcp_server_version(
            server_json=_server_json("io.github.test/client-too-many-tools", "1.0.0"),
            tools=[MCPTool(name=f"tool-{i}") for i in range(1001)],
        )


def test_mlflow_client_update_version_rejects_risky_tool_icons():
    client = MlflowClient()
    client.create_mcp_server_version(_server_json("io.github.test/client-tool-icons", "1.0.0"))

    with pytest.raises(MlflowException, match="Icon URL"):
        client.update_mcp_server_version(
            name="io.github.test/client-tool-icons",
            version="1.0.0",
            tools=[MCPTool(name="search", icons=[{"src": "https://127.0.0.1/icon.png"}])],
        )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "register_mcp_server",
        "register_mcp_server_from_url",
        "get_mcp_server",
        "search_mcp_servers",
        "update_mcp_server",
        "delete_mcp_server",
        "get_mcp_server_version",
        "get_mcp_server_version_by_alias",
        "get_latest_mcp_server_version",
        "search_mcp_server_versions",
        "update_mcp_server_version",
        "delete_mcp_server_version",
        "create_mcp_access_endpoint",
        "get_mcp_access_endpoint",
        "search_mcp_access_endpoints",
        "update_mcp_access_endpoint",
        "delete_mcp_access_endpoint",
        "set_mcp_server_tag",
        "delete_mcp_server_tag",
        "set_mcp_server_version_tag",
        "delete_mcp_server_version_tag",
        "set_mcp_server_alias",
        "delete_mcp_server_alias",
    ],
)
def test_function_exported(name):
    import mlflow.genai

    assert name in mlflow.genai.__all__
    assert hasattr(mlflow.genai, name)
