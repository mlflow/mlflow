from unittest import mock

import pytest

from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus, MCPTool
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import NOT_SET

pytestmark = pytest.mark.notrackingurimock


def _server_json(name="io.github.test/servererver", version="1.0.0", description=None, **extra):
    server_json = {"name": name, "version": version, "title": f"Test {name}"}
    if description is not None:
        server_json["description"] = description
    server_json.update(extra)
    return server_json


@pytest.fixture(autouse=True)
def mock_icon_url_dns_resolution(monkeypatch):
    def _resolve(host, port, *a, **kw):
        if host == "localhost":
            ip = "127.0.0.1"
        elif host == "example.com" or host.endswith(".example.com"):
            ip = "8.8.8.8"
        else:
            ip = host
        return [(None, None, None, None, (ip, 0))]

    monkeypatch.setattr("mlflow.utils.validation.socket.getaddrinfo", _resolve)


def _setup_server(store, name, versions=("1.0.0",), aliases=None):
    """Create a server with versions and optional aliases for endpoint tests."""
    for v in versions:
        store.create_mcp_server_version(_server_json(name, v))
    for alias, ver in (aliases or {}).items():
        store.set_mcp_server_alias(name, alias, ver)


def _create_version(
    store,
    name="io.github.test/servererver",
    version="1.0.0",
    description=None,
    status=MCPStatus.DRAFT,
    **extra,
):
    initial_status = MCPStatus.ACTIVE if status == MCPStatus.DEPRECATED else status
    created = store.create_mcp_server_version(
        _server_json(name, version, description=description, **extra),
        status=initial_status,
    )
    if status == MCPStatus.DEPRECATED:
        return store.update_mcp_server_version(name, version, status=MCPStatus.DEPRECATED)
    return created


# --- MCPServer CRUD ---


def test_create_mcp_server(store):
    server = store.create_mcp_server(
        "io.github.test/server", description="A test server", created_by="alice"
    )
    assert server.name == "io.github.test/server"
    assert server.description == "A test server"
    assert server.creation_timestamp is not None
    assert server.created_by == "alice"
    assert server.last_updated_by == "alice"


def test_create_mcp_server_duplicate_raises(store):
    store.create_mcp_server("io.github.test/servererver")
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_mcp_server("io.github.test/servererver")
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_mcp_server_empty_name_raises(store):
    with pytest.raises(MlflowException, match="must not be empty") as exc:
        store.create_mcp_server("")
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


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
def test_create_mcp_server_invalid_name_raises(store, invalid_name):
    with pytest.raises(
        MlflowException, match="Expected '<reverse-dns namespace>/<server slug>'"
    ) as exc:
        store.create_mcp_server(invalid_name)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


@pytest.mark.parametrize(
    "name",
    [
        "io.github.TestOrg/server-name",
        "com/server-name",
        "io.github.test/servererver_name",
        "io.github.test/servererver.name",
    ],
)
def test_create_mcp_server_accepts_upstream_name_shapes(store, name):
    server = store.create_mcp_server(name)
    assert server.name == name


def test_create_mcp_server_with_icons(store):
    icons = [{"src": "https://example.com/icon.png", "sizes": "32x32"}]
    server = store.create_mcp_server("io.github.test/servererver", icons=icons)
    assert server.icons == [{**icons[0], "source": "server"}]


def test_create_mcp_server_rejects_risky_icons(store):
    with pytest.raises(MlflowException, match="Icon URL"):
        store.create_mcp_server(
            "io.github.test/risky-icons",
            icons=[{"src": "https://127.0.0.1/icon.png"}],
        )


def test_create_mcp_server_rejects_too_many_icons(store):
    with pytest.raises(MlflowException, match="at most 100 items"):
        store.create_mcp_server(
            "io.github.test/too-many-icons",
            icons=[{"src": f"https://example.com/icon-{i}.png"} for i in range(101)],
        )


def test_update_mcp_server_sets_last_updated_by(store):
    store.create_mcp_server("io.github.test/server")
    server = store.update_mcp_server(
        "io.github.test/server", description="updated", last_updated_by="bob"
    )
    assert server.last_updated_by == "bob"
    assert server.created_by is None


def test_get_mcp_server(store):
    store.create_mcp_server("io.github.test/servererver", description="desc")
    server = store.get_mcp_server("io.github.test/servererver")
    assert server.name == "io.github.test/servererver"
    assert server.description == "desc"


def test_get_mcp_server_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_mcp_server("io.github.test/nonexistent")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_get_mcp_server_with_tags(store):
    store.create_mcp_server("io.github.test/servererver")
    store.set_mcp_server_tag("io.github.test/servererver", "team", "platform")
    server = store.get_mcp_server("io.github.test/servererver")
    assert server.tags == {"team": "platform"}


def test_get_mcp_server_access_endpoints_include_resolved_versions(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "prod", "1.0.0")
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://direct.example.com", server_version="1.0.0"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://alias.example.com", server_alias="prod"
    )
    server = store.get_mcp_server("io.github.test/server")
    endpoints = {e.url: e for e in server.access_endpoints}
    assert endpoints["https://direct.example.com"].resolved_version is not None
    assert endpoints["https://direct.example.com"].resolved_version.version == "1.0.0"
    assert endpoints["https://alias.example.com"].resolved_version is not None
    assert endpoints["https://alias.example.com"].resolved_version.version == "1.0.0"


def test_search_mcp_servers_access_endpoints_include_resolved_versions(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "prod", "1.0.0")
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://alias.example.com", server_alias="prod"
    )
    server = store.search_mcp_servers()[0]
    assert len(server.access_endpoints) == 1
    assert server.access_endpoints[0].resolved_version is not None
    assert server.access_endpoints[0].resolved_version.version == "1.0.0"


def test_search_mcp_servers_empty(store):
    result = store.search_mcp_servers()
    assert len(result) == 0
    assert result.token is None


def test_search_mcp_servers_returns_all(store):
    store.create_mcp_server("io.github.test/servererver-a")
    store.create_mcp_server("io.github.test/servererver-b")
    result = store.search_mcp_servers()
    assert len(result) == 2


def test_search_mcp_servers_pagination(store):
    for i in range(5):
        store.create_mcp_server(f"io.github.test/servererver-{i:02d}")
    page1 = store.search_mcp_servers(max_results=2)
    assert len(page1) == 2
    assert page1.token is not None
    page2 = store.search_mcp_servers(max_results=2, page_token=page1.token)
    assert len(page2) == 2
    assert page2.token is not None
    page3 = store.search_mcp_servers(max_results=2, page_token=page2.token)
    assert len(page3) == 1
    assert page3.token is None


def test_update_mcp_server_description(store):
    store.create_mcp_server("io.github.test/servererver", description="old")
    updated = store.update_mcp_server("io.github.test/servererver", description="new")
    assert updated.description == "new"


def test_update_mcp_server_rejects_risky_icons(store):
    store.create_mcp_server("io.github.test/update-icons")
    with pytest.raises(MlflowException, match="Icon URL"):
        store.update_mcp_server(
            "io.github.test/update-icons",
            icons=[{"src": "https://127.0.0.1/icon.png"}],
        )


def test_update_mcp_server_rejects_too_many_icons(store):
    store.create_mcp_server("io.github.test/update-too-many-icons")
    with pytest.raises(MlflowException, match="at most 100 items"):
        store.update_mcp_server(
            "io.github.test/update-too-many-icons",
            icons=[{"src": f"https://example.com/icon-{i}.png"} for i in range(101)],
        )


def test_update_mcp_server_display_name(store):
    store.create_mcp_server("io.github.test/servererver")
    updated = store.update_mcp_server("io.github.test/servererver", display_name="My Server")
    assert updated.display_name == "My Server"


def test_update_mcp_server_returns_complete_entity(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_tag("io.github.test/server", "env", "prod")
    store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")
    updated = store.update_mcp_server("io.github.test/server", description="new desc")
    assert updated.description == "new desc"
    assert updated.status == MCPStatus.ACTIVE
    assert updated.tags == {"env": "prod"}
    assert updated.aliases == {"stable": "1.0.0"}


def test_update_mcp_server_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_server("io.github.test/nonexistent", description="x")


def test_update_mcp_server_no_latest_version_param(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/servererver", "2.0.0"), status=MCPStatus.ACTIVE
    )
    with pytest.raises(TypeError, match="latest_version"):
        store.update_mcp_server("io.github.test/servererver", latest_version="2.0.0")


def test_create_mcp_server_version_rejects_semver_component_exceeding_db_integer_limit(store):
    with pytest.raises(MlflowException, match="server_json.version.*2147483647"):
        store.create_mcp_server_version(
            _server_json("io.github.test/servererver", "2147483648.0.0"),
            status=MCPStatus.ACTIVE,
        )


@pytest.mark.parametrize("status", [MCPStatus.DEPRECATED, MCPStatus.DELETED])
def test_create_mcp_server_version_rejects_non_initial_statuses(store, status):
    with pytest.raises(
        MlflowException,
        match="Initial MCP server registration status must be 'draft' or 'active'",
    ):
        store.create_mcp_server_version(
            _server_json(f"io.github.test/non-initial-{status.value}", "1.0.0"),
            status=status,
        )


def test_get_mcp_server_latest_version_uses_highest_active_semver(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/servererver", "1.2.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/servererver", "1.10.0"), status=MCPStatus.ACTIVE
    )
    updated = store.get_mcp_server("io.github.test/servererver")
    assert updated.latest_version == "1.10.0"


def test_resolved_parent_query_prefers_active_over_higher_deprecated_version(store):
    from mlflow.store.tracking.dbmodels.models import SqlMCPServer

    store.create_mcp_server_version(
        _server_json("io.github.test/servererver", "1.0.0"), status=MCPStatus.ACTIVE
    )
    _create_version(store, "io.github.test/servererver", "2.0.0", status=MCPStatus.DEPRECATED)

    with store.ManagedSessionMaker() as session:
        query = store._get_query(session, SqlMCPServer).filter(
            SqlMCPServer.name == "io.github.test/servererver"
        )
        server = SqlMCPServer.with_resolved_latest(query).one()
        assert server.resolved_status == MCPStatus.ACTIVE.value
        assert server.resolved_latest_version == "1.0.0"


def test_resolved_parent_query_fallback_uses_highest_semver_without_active(store):
    from mlflow.store.tracking.dbmodels.models import SqlMCPServer

    _create_version(store, "io.github.test/servererver", "1.0.0", status=MCPStatus.DEPRECATED)
    store.create_mcp_server_version(
        _server_json("io.github.test/servererver", "2.0.0"), status=MCPStatus.DRAFT
    )

    with store.ManagedSessionMaker() as session:
        query = store._get_query(session, SqlMCPServer).filter(
            SqlMCPServer.name == "io.github.test/servererver"
        )
        server = SqlMCPServer.with_resolved_latest(query).one()
        assert server.resolved_status == MCPStatus.DRAFT.value
        assert server.resolved_latest_version == "2.0.0"


def test_get_mcp_server_resolved_status(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "2.0.0"), status=MCPStatus.DRAFT
    )
    server = store.get_mcp_server("io.github.test/server")
    assert server.status == MCPStatus.ACTIVE


def test_get_mcp_server_resolved_status_fallback(store):
    _create_version(store, "io.github.test/server", "2.0.0", status=MCPStatus.DEPRECATED)
    server = store.get_mcp_server("io.github.test/server")
    assert server.status == MCPStatus.DEPRECATED
    assert server.latest_version == "2.0.0"


def test_get_mcp_server_description_falls_back_to_highest_active_semver(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0", description="active description"),
        status=MCPStatus.ACTIVE,
    )
    _create_version(
        store,
        "io.github.test/server",
        "2.0.0",
        description="deprecated description",
        status=MCPStatus.DEPRECATED,
    )
    server = store.get_mcp_server("io.github.test/server")
    assert server.description == "active description"


def test_get_mcp_server_description_falls_back_to_highest_non_deleted_version(store):
    _create_version(
        store,
        "io.github.test/server",
        "2.0.0",
        description="deprecated description",
        status=MCPStatus.DEPRECATED,
    )
    server = store.get_mcp_server("io.github.test/server")
    assert server.description == "deprecated description"
    assert server.latest_version == "2.0.0"


def test_get_mcp_server_icons_fall_back_to_highest_active_semver(store):
    store.create_mcp_server_version(
        _server_json(
            "io.github.test/server",
            "1.0.0",
            icons=[{"src": "https://example.com/active.png", "theme": "dark"}],
        ),
        status=MCPStatus.ACTIVE,
    )
    _create_version(
        store,
        "io.github.test/server",
        "2.0.0",
        icons=[{"src": "https://example.com/deprecated.png", "theme": "dark"}],
        status=MCPStatus.DEPRECATED,
    )

    server = store.get_mcp_server("io.github.test/server")
    assert server.icons == [
        {
            "src": "https://example.com/active.png",
            "theme": "dark",
            "source": "version",
        }
    ]


def test_get_mcp_server_icons_fall_back_to_highest_non_deleted_version(store):
    _create_version(
        store,
        "io.github.test/server",
        "2.0.0",
        icons=[{"src": "https://example.com/deprecated.png", "theme": "dark"}],
        status=MCPStatus.DEPRECATED,
    )

    server = store.get_mcp_server("io.github.test/server")
    assert server.icons == [
        {
            "src": "https://example.com/deprecated.png",
            "theme": "dark",
            "source": "version",
        }
    ]


def test_get_mcp_server_icons_merge_server_and_version_by_theme(store):
    store.create_mcp_server(
        "io.github.test/server",
        icons=[{"src": "https://example.com/server-dark.png", "theme": "dark"}],
    )
    store.create_mcp_server_version(
        _server_json(
            "io.github.test/server",
            "1.0.0",
            icons=[
                {"src": "https://example.com/version-dark.png", "theme": "dark"},
                {"src": "https://example.com/version-light.png", "theme": "light"},
            ],
        ),
        status=MCPStatus.ACTIVE,
    )

    server = store.get_mcp_server("io.github.test/server")
    assert server.icons == [
        {
            "src": "https://example.com/server-dark.png",
            "theme": "dark",
            "source": "server",
        },
        {
            "src": "https://example.com/version-light.png",
            "theme": "light",
            "source": "version",
        },
    ]


def test_get_mcp_server_empty_icons_do_not_fall_back_to_version(store):
    store.create_mcp_server(
        "io.github.test/server",
        icons=[],
    )
    store.create_mcp_server_version(
        _server_json(
            "io.github.test/server",
            "1.0.0",
            icons=[{"src": "https://example.com/version.png", "theme": "dark"}],
        ),
        status=MCPStatus.ACTIVE,
    )

    server = store.get_mcp_server("io.github.test/server")
    assert server.icons == []


def test_get_mcp_server_empty_icons_remain_empty_without_version(store):
    store.create_mcp_server("io.github.test/server", icons=[])
    server = store.get_mcp_server("io.github.test/server")
    assert server.icons == []


def test_create_mcp_server_version_ignores_server_icon_source(store):
    version = store.create_mcp_server_version(
        _server_json(
            "io.github.test/server",
            "1.0.0",
            icons=[{"src": "https://example.com/icon.png", "source": "server"}],
        ),
        tools=[
            MCPTool(
                name="search",
                icons=[{"src": "https://example.com/tool.png", "source": "server"}],
            )
        ],
        status=MCPStatus.ACTIVE,
    )
    assert version.server_json["icons"] == [{"src": "https://example.com/icon.png"}]
    assert version.tools[0].icons == [{"src": "https://example.com/tool.png"}]


def test_create_mcp_server_version_rejects_version_icon_source(store):
    with pytest.raises(MlflowException, match="source.*version"):
        store.create_mcp_server_version(
            _server_json(
                "io.github.test/server",
                "1.0.0",
                icons=[{"src": "https://example.com/icon.png", "source": "version"}],
            ),
            status=MCPStatus.ACTIVE,
        )


def test_create_mcp_server_rejects_version_icon_source(store):
    with pytest.raises(MlflowException, match="source.*version"):
        store.create_mcp_server(
            "io.github.test/server",
            icons=[{"src": "https://example.com/icon.png", "source": "version"}],
        )


def test_get_mcp_server_resolved_status_no_versions(store):
    store.create_mcp_server("io.github.test/server")
    server = store.get_mcp_server("io.github.test/server")
    assert server.status is None


def test_search_mcp_servers_resolves_icons_from_latest_version(store):
    store.create_mcp_server_version(
        _server_json(
            "io.github.test/server",
            "1.0.0",
            icons=[{"src": "https://example.com/icon.png"}],
        ),
        status=MCPStatus.ACTIVE,
    )

    servers = store.search_mcp_servers()
    assert servers[0].icons == [{"src": "https://example.com/icon.png", "source": "version"}]


def test_delete_mcp_server(store):
    store.create_mcp_server("io.github.test/servererver")
    store.delete_mcp_server("io.github.test/servererver")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server("io.github.test/servererver")


def test_delete_mcp_server_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server("io.github.test/nonexistent")


def test_delete_mcp_server_cascades_to_versions(store):
    store.create_mcp_server_version(_server_json("io.github.test/server1", "1.0.0"))
    store.delete_mcp_server("io.github.test/server1")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version("io.github.test/server1", "1.0.0")


def test_delete_mcp_server_cascades_to_tags(store):
    store.create_mcp_server("io.github.test/server1")
    store.set_mcp_server_tag("io.github.test/server1", "k", "v")
    store.delete_mcp_server("io.github.test/server1")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server("io.github.test/server1")


def test_delete_mcp_server_cascades_to_aliases(store):
    store.create_mcp_server_version(_server_json("io.github.test/server1", "1.0.0"))
    store.set_mcp_server_alias("io.github.test/server1", "stable", "1.0.0")
    store.delete_mcp_server("io.github.test/server1")
    # Recreate the server and verify the old alias is gone (was cascade-deleted).
    store.create_mcp_server_version(_server_json("io.github.test/server1", "1.0.0"))
    server = store.get_mcp_server("io.github.test/server1")
    assert server.aliases == {}


def test_delete_mcp_server_cascades_to_endpoints(store):
    _setup_server(store, "io.github.test/server1")
    store.create_mcp_access_endpoint(
        "io.github.test/server1", "https://a.com", server_version="1.0.0"
    )
    store.delete_mcp_server("io.github.test/server1")
    result = store.search_mcp_access_endpoints(server_name="io.github.test/server1")
    assert len(result) == 0


def test_delete_mcp_server_with_active_version_raises(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/delete-active-server", "1.0.0"),
        status=MCPStatus.ACTIVE,
    )
    with pytest.raises(MlflowException, match="active version"):
        store.delete_mcp_server("io.github.test/delete-active-server")


# --- MCPServerVersion CRUD ---


def test_create_mcp_server_version(store):
    sv = store.create_mcp_server_version(
        _server_json(),
        source="https://github.com/org/repo",
        status=MCPStatus.ACTIVE,
        created_by="alice",
    )
    assert sv.name == "io.github.test/servererver"
    assert sv.version == "1.0.0"
    assert sv.status == MCPStatus.ACTIVE
    assert sv.source == "https://github.com/org/repo"
    assert sv.server_json == _server_json()
    assert sv.created_by == "alice"
    assert sv.last_updated_by == "alice"


def test_create_mcp_server_version_rejects_risky_server_json_icons(store):
    with pytest.raises(MlflowException, match="Icon URL"):
        store.create_mcp_server_version(
            _server_json(
                "io.github.test/server-json-icons",
                "1.0.0",
                icons=[{"src": "https://127.0.0.1/icon.png"}],
            )
        )


def test_create_mcp_server_version_preserves_meta_icons_metadata(store):
    sv = store.create_mcp_server_version(
        _server_json(
            "io.github.test/meta-icons",
            "1.0.0",
            _meta={"icons": {"not": "an-icon-list"}, "other": "preserved"},
        )
    )
    assert sv.server_json["_meta"] == {"icons": {"not": "an-icon-list"}, "other": "preserved"}


def test_create_mcp_server_version_auto_creates_parent(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/new-server", "1.0.0"), created_by="alice"
    )
    server = store.get_mcp_server("io.github.test/new-server")
    assert server.name == "io.github.test/new-server"
    assert server.created_by == "alice"
    assert server.last_updated_by == "alice"


def test_create_mcp_server_version_with_existing_parent(store):
    store.create_mcp_server("io.github.test/servererver", description="existing")
    store.create_mcp_server_version(_server_json())
    server = store.get_mcp_server("io.github.test/servererver")
    assert server.description == "existing"


def test_create_mcp_server_version_duplicate_raises(store):
    store.create_mcp_server_version(_server_json())
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_mcp_server_version(_server_json())
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_mcp_server_version_deleted_version_not_reusable(store):
    original = _server_json("io.github.test/server", "1.0.0")
    replacement = {
        "name": "io.github.test/server",
        "version": "1.0.0",
        "title": "Replacement Title",
        "description": "Replacement payload with same identity",
    }
    store.create_mcp_server_version(original)
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_mcp_server_version(replacement)
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_mcp_server_version_missing_name_raises(store):
    with pytest.raises(MlflowException, match="name.*version") as exc:
        store.create_mcp_server_version({"version": "1.0.0"})
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_mcp_server_version_missing_version_raises(store):
    with pytest.raises(MlflowException, match="name.*version") as exc:
        store.create_mcp_server_version({"name": "test"})
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


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
def test_create_mcp_server_version_invalid_name_raises(store, invalid_name):
    with pytest.raises(
        MlflowException, match="Expected '<reverse-dns namespace>/<server slug>'"
    ) as exc:
        store.create_mcp_server_version({"name": invalid_name, "version": "1.0.0"})
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


@pytest.mark.parametrize(
    "name",
    [
        "io.github.TestOrg/server-name",
        "com/server-name",
        "io.github.test/servererver_name",
        "io.github.test/servererver.name",
    ],
)
def test_create_mcp_server_version_accepts_upstream_name_shapes(store, name):
    sv = store.create_mcp_server_version({"name": name, "version": "1.0.0"})
    assert sv.name == name


def test_create_mcp_server_version_with_tools(store):
    tools = [MCPTool(name="web_search", description="Search the web")]
    sv = store.create_mcp_server_version(_server_json(), tools=tools)
    assert sv.tools is not None
    assert len(sv.tools) == 1
    assert sv.tools[0].name == "web_search"


def test_create_mcp_server_version_omitted_tools_store_null_without_discovery(store):
    # No remotes: omit/NOT_SET stores null without network.
    sv = store.create_mcp_server_version(_server_json("io.github.test/omit-tools", "1.0.0"))
    assert sv.tools is None

    remotes_sj = _server_json(
        "io.github.test/discover-tools",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/sql"}],
    )
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
        sv_discovered = store.create_mcp_server_version(remotes_sj, tools=NOT_SET)
    mock_discover.assert_not_called()
    assert sv_discovered.tools is None

    # Explicit None disables discovery even when remotes exist.
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
        sv_none = store.create_mcp_server_version(
            _server_json(
                "io.github.test/none-tools",
                "1.0.0",
                remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/skip"}],
            ),
            tools=None,
        )
    mock_discover.assert_not_called()
    assert sv_none.tools is None


def test_create_mcp_server_version_duplicate_raises_without_discovery(store):
    remotes_sj = _server_json(
        "io.github.test/dup-discover",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/dup"}],
    )
    store.create_mcp_server_version(remotes_sj, tools=None)
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
        with pytest.raises(MlflowException, match="already exists") as exc:
            store.create_mcp_server_version(remotes_sj)
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"
    mock_discover.assert_not_called()


def test_create_mcp_server_version_deleted_duplicate_raises_without_discovery(store):
    remotes_sj = _server_json(
        "io.github.test/dup-deleted-discover",
        "1.0.0",
        remotes=[{"type": "streamable-http", "url": "https://mcp.example.com/dup-del"}],
    )
    store.create_mcp_server_version(remotes_sj, tools=None)
    store.delete_mcp_server_version("io.github.test/dup-deleted-discover", "1.0.0")
    with mock.patch("mlflow.genai.mcp_tool_discovery.discover_mcp_tools") as mock_discover:
        with pytest.raises(MlflowException, match="already exists") as exc:
            store.create_mcp_server_version(remotes_sj)
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"
    mock_discover.assert_not_called()


def test_create_mcp_server_version_rejects_risky_tool_icons(store):
    with pytest.raises(MlflowException, match="Icon URL"):
        store.create_mcp_server_version(
            _server_json(),
            tools=[MCPTool(name="search", icons=[{"src": "https://127.0.0.1/icon.png"}])],
        )


def test_create_mcp_server_version_rejects_too_many_tools(store):
    with pytest.raises(MlflowException, match="at most 1000 items"):
        store.create_mcp_server_version(
            _server_json("io.github.test/too-many-tools", "1.0.0"),
            tools=[MCPTool(name=f"tool-{i}") for i in range(1001)],
        )


def test_create_mcp_server_version_with_empty_tools_preserves_empty_list(store):
    sv = store.create_mcp_server_version(_server_json(), tools=[])
    assert sv.tools == []


def test_create_mcp_server_version_with_status(store):
    sv = store.create_mcp_server_version(_server_json(), status=MCPStatus.ACTIVE)
    assert sv.status == MCPStatus.ACTIVE


def test_create_mcp_server_version_with_source(store):
    sv = store.create_mcp_server_version(_server_json(), source="https://github.com/org/repo")
    assert sv.source == "https://github.com/org/repo"


def test_get_mcp_server_version(store):
    store.create_mcp_server_version(_server_json())
    sv = store.get_mcp_server_version("io.github.test/servererver", "1.0.0")
    assert sv.version == "1.0.0"


def test_get_mcp_server_version_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version("io.github.test/nonexistent", "1.0.0")


def test_get_latest_mcp_server_version_highest_semver_active(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.2.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.10.0"), status=MCPStatus.ACTIVE
    )
    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.10.0"


def test_prerelease_semver_resolution_end_to_end(store):
    for version in ("1.0.0-alpha.2", "1.0.0-alpha.10", "1.0.0-beta.1"):
        store.create_mcp_server_version(
            _server_json("io.github.test/server", version), status=MCPStatus.ACTIVE
        )

    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0-beta.1"

    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0-beta.1"
    assert server.status == MCPStatus.ACTIVE

    aliased = store.get_mcp_server_version_by_alias("io.github.test/server", "latest")
    assert aliased.version == "1.0.0-beta.1"

    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )
    assert endpoint.resolved_version is not None
    assert endpoint.resolved_version.version == "1.0.0-beta.1"


def test_prerelease_numeric_tiebreak_applies_to_parent_metadata(store):
    for version in ("1.0.0-alpha.2", "1.0.0-alpha.10"):
        store.create_mcp_server_version(
            _server_json("io.github.test/server", version), status=MCPStatus.ACTIVE
        )

    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0-alpha.10"

    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0-alpha.10"
    assert server.status == MCPStatus.ACTIVE

    searched = store.search_mcp_servers(filter_string="name = 'io.github.test/server'")
    assert len(searched) == 1
    assert searched[0].latest_version == "1.0.0-alpha.10"
    assert searched[0].status == MCPStatus.ACTIVE


def test_prerelease_prefix_identifier_resolution_end_to_end(store):
    for version in ("1.0.0-alpha", "1.0.0-alpha1"):
        store.create_mcp_server_version(
            _server_json("io.github.test/server", version), status=MCPStatus.ACTIVE
        )

    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0-alpha1"

    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0-alpha1"

    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )
    assert endpoint.resolved_version is not None
    assert endpoint.resolved_version.version == "1.0.0-alpha1"


def test_prerelease_prefix_identifier_with_hyphen_resolution_end_to_end(store):
    for version in ("1.0.0-alpha", "1.0.0-alpha-1"):
        store.create_mcp_server_version(
            _server_json("io.github.test/server", version), status=MCPStatus.ACTIVE
        )

    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0-alpha-1"

    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0-alpha-1"


def test_prerelease_shorter_identifier_list_sorts_lower_end_to_end(store):
    for version in ("1.0.0-alpha", "1.0.0-alpha.1"):
        store.create_mcp_server_version(
            _server_json("io.github.test/server", version), status=MCPStatus.ACTIVE
        )

    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0-alpha.1"

    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0-alpha.1"


def test_prerelease_numeric_identifier_sorts_lower_than_nonnumeric_end_to_end(store):
    for version in ("1.0.0-1", "1.0.0-alpha"):
        store.create_mcp_server_version(
            _server_json("io.github.test/server", version), status=MCPStatus.ACTIVE
        )

    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0-alpha"

    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0-alpha"


def test_build_metadata_uses_created_at_before_raw_version_as_latest_tiebreaker(store, monkeypatch):
    times = iter((1000, 2000, 3000))
    monkeypatch.setattr(
        "mlflow.store.tracking.mcp_server_registry.sqlalchemy_mixin.get_current_time_millis",
        lambda: next(times),
    )
    for version in ("1.0.0+xyz", "1.0.0+abc"):
        store.create_mcp_server_version(
            _server_json("io.github.test/server", version), status=MCPStatus.ACTIVE
        )

    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0+abc"

    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0+abc"
    assert server.status == MCPStatus.ACTIVE

    aliased = store.get_mcp_server_version_by_alias("io.github.test/server", "latest")
    assert aliased.version == "1.0.0+abc"

    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )
    assert endpoint.resolved_version is not None
    assert endpoint.resolved_version.version == "1.0.0+abc"


def test_get_latest_mcp_server_version_ignores_non_active_versions(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.update_mcp_server_version("io.github.test/server", "1.0.0", status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("io.github.test/server", "2.0.0"))
    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0"


def test_get_latest_mcp_server_version_same_timestamp_tiebreaker(store, monkeypatch):
    """When two versions share the same created_at, raw version breaks the tie.

    This must match the ordering in _latest_candidates_query to keep
    get_latest_mcp_server_version and resolved_status consistent.
    """
    monkeypatch.setattr(
        "mlflow.store.tracking.mcp_server_registry.sqlalchemy_mixin.get_current_time_millis",
        lambda: 1000,
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "2.0.0"), status=MCPStatus.ACTIVE
    )
    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "2.0.0"

    server = store.get_mcp_server("io.github.test/server")
    assert server.status == MCPStatus.ACTIVE


def test_get_latest_mcp_server_version_falls_back_to_draft(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0"


def test_get_latest_mcp_server_version_server_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_latest_mcp_server_version("io.github.test/nonexistent")


def test_get_mcp_server_version_by_alias(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")
    sv = store.get_mcp_server_version_by_alias("io.github.test/server", "stable")
    assert sv.version == "1.0.0"


def test_get_mcp_server_version_by_latest_alias(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    sv = store.get_mcp_server_version_by_alias("io.github.test/server", "latest")
    assert sv.version == "1.0.0"


def test_get_mcp_server_version_by_alias_not_found(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version_by_alias("io.github.test/server", "io.github.test/nonexistent")


def test_search_mcp_server_versions(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.create_mcp_server_version(_server_json("io.github.test/server", "2.0.0"))
    result = store.search_mcp_server_versions("io.github.test/server")
    assert len(result) == 2


def test_search_mcp_server_versions_scoped(store):
    store.create_mcp_server_version(_server_json("io.github.test/server1", "1.0.0"))
    store.create_mcp_server_version(_server_json("io.github.test/server2", "1.0.0"))
    result = store.search_mcp_server_versions("io.github.test/server1")
    assert len(result) == 1
    assert result[0].name == "io.github.test/server1"


def test_update_mcp_server_version_status(store):
    store.create_mcp_server_version(_server_json())
    updated = store.update_mcp_server_version(
        "io.github.test/servererver", "1.0.0", status=MCPStatus.ACTIVE, last_updated_by="bob"
    )
    assert updated.status == MCPStatus.ACTIVE
    assert updated.last_updated_by == "bob"


def test_update_mcp_server_version_null_status_raises(store):
    store.create_mcp_server_version(_server_json())
    with pytest.raises(MlflowException, match="status cannot be null") as exc:
        store.update_mcp_server_version("io.github.test/servererver", "1.0.0", status=None)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_mcp_server_version_to_draft_recomputes_latest_version(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "2.0.0"), status=MCPStatus.ACTIVE
    )
    store.update_mcp_server_version("io.github.test/server", "2.0.0", status=MCPStatus.DRAFT)
    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0"
    assert server.status == MCPStatus.ACTIVE


def test_latest_resolution_falls_back_to_highest_non_active_semver(store):
    _create_version(store, "io.github.test/server", "1.2.0", status=MCPStatus.DEPRECATED)
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.3.0"), status=MCPStatus.DRAFT
    )

    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.3.0"

    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.3.0"
    assert server.status == MCPStatus.DRAFT


def test_update_mcp_server_version_invalid_transition(store):
    store.create_mcp_server_version(_server_json(), status=MCPStatus.ACTIVE)
    with pytest.raises(MlflowException, match="Invalid status transition") as exc:
        store.update_mcp_server_version(
            "io.github.test/servererver", "1.0.0", status=MCPStatus.DELETED
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_mcp_server_version_display_name(store):
    store.create_mcp_server_version(_server_json())
    updated = store.update_mcp_server_version(
        "io.github.test/servererver", "1.0.0", display_name="v1"
    )
    assert updated.display_name == "v1"


def test_update_mcp_server_version_tools(store):
    store.create_mcp_server_version(_server_json())
    tools = [MCPTool(name="calculator")]
    updated = store.update_mcp_server_version("io.github.test/servererver", "1.0.0", tools=tools)
    assert len(updated.tools) == 1
    assert updated.tools[0].name == "calculator"


def test_update_mcp_server_version_rejects_risky_tool_icons(store):
    store.create_mcp_server_version(_server_json())
    with pytest.raises(MlflowException, match="Icon URL"):
        store.update_mcp_server_version(
            "io.github.test/servererver",
            "1.0.0",
            tools=[MCPTool(name="search", icons=[{"src": "https://127.0.0.1/icon.png"}])],
        )


def test_update_mcp_server_version_rejects_too_many_tools(store):
    store.create_mcp_server_version(_server_json())
    with pytest.raises(MlflowException, match="at most 1000 items"):
        store.update_mcp_server_version(
            "io.github.test/servererver",
            "1.0.0",
            tools=[MCPTool(name=f"tool-{i}") for i in range(1001)],
        )


def test_update_mcp_server_version_tools_empty_list_preserved(store):
    store.create_mcp_server_version(
        _server_json(),
        tools=[MCPTool(name="calculator")],
    )
    updated = store.update_mcp_server_version("io.github.test/servererver", "1.0.0", tools=[])
    assert updated.tools == []


def test_update_mcp_server_version_tools_none_clears_tools(store):
    store.create_mcp_server_version(
        _server_json(),
        tools=[MCPTool(name="calculator")],
    )
    updated = store.update_mcp_server_version("io.github.test/servererver", "1.0.0", tools=None)
    assert updated.tools is None


def test_update_mcp_server_version_omitted_tools_leaves_unchanged(store):
    store.create_mcp_server_version(
        _server_json(),
        tools=[MCPTool(name="calculator")],
    )
    updated = store.update_mcp_server_version(
        "io.github.test/servererver", "1.0.0", display_name="v1"
    )
    assert updated.display_name == "v1"
    assert updated.tools is not None
    assert updated.tools[0].name == "calculator"


def test_update_mcp_server_version_returns_complete_entity(store):
    store.create_mcp_server_version(_server_json(), status=MCPStatus.ACTIVE)
    store.set_mcp_server_version_tag("io.github.test/servererver", "1.0.0", "env", "prod")
    updated = store.update_mcp_server_version(
        "io.github.test/servererver", "1.0.0", display_name="Updated"
    )
    assert updated.display_name == "Updated"
    assert updated.status == MCPStatus.ACTIVE
    assert updated.server_json == _server_json()
    assert updated.tags == {"env": "prod"}


def test_update_mcp_server_version_deleted_raises(store):
    store.create_mcp_server_version(_server_json())
    store.delete_mcp_server_version("io.github.test/servererver", "1.0.0")
    with pytest.raises(MlflowException, match="not found") as exc:
        store.update_mcp_server_version(
            "io.github.test/servererver", "1.0.0", display_name="Updated"
        )
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_delete_mcp_server_version_soft_delete(store):
    store.create_mcp_server_version(_server_json())
    store.delete_mcp_server_version("io.github.test/servererver", "1.0.0")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version("io.github.test/servererver", "1.0.0")


def test_delete_mcp_server_version_clears_latest_pin(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    _create_version(store, "io.github.test/server", "2.0.0", status=MCPStatus.DEPRECATED)
    store.delete_mcp_server_version("io.github.test/server", "2.0.0")
    server = store.get_mcp_server("io.github.test/server")
    assert server.latest_version == "1.0.0"
    assert server.status == MCPStatus.ACTIVE


def test_delete_mcp_server_version_active_raises(store):
    store.create_mcp_server_version(_server_json(), status=MCPStatus.ACTIVE)
    with pytest.raises(MlflowException, match="Invalid status transition"):
        store.delete_mcp_server_version("io.github.test/servererver", "1.0.0")


def test_delete_mcp_server_version_cleans_up_aliases_and_endpoints(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://direct.example.com", server_version="1.0.0"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://alias.example.com", server_alias="stable"
    )
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    server = store.get_mcp_server("io.github.test/server")
    assert server.aliases == {}
    assert len(store.search_mcp_access_endpoints(server_name="io.github.test/server")) == 0


def test_delete_mcp_server_version_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server_version("io.github.test/nonexistent", "1.0.0")


# --- MCPAccessEndpoint CRUD ---


def test_create_mcp_access_endpoint_with_version(store):
    _setup_server(store, "io.github.test/server")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://mcp.example.com",
        server_version="1.0.0",
        created_by="alice",
    )
    assert endpoint.server_name == "io.github.test/server"
    assert endpoint.url == "https://mcp.example.com"
    assert endpoint.server_version == "1.0.0"
    assert endpoint.transport_type == MCPRemoteTransportType.STREAMABLE_HTTP
    assert endpoint.created_by == "alice"
    assert endpoint.last_updated_by == "alice"


def test_create_mcp_access_endpoint_with_alias(store):
    _setup_server(store, "io.github.test/server", aliases={"stable": "1.0.0"})
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://mcp.example.com", server_alias="stable"
    )
    assert endpoint.server_alias == "stable"


def test_create_mcp_access_endpoint_nonexistent_version_raises(store):
    _setup_server(store, "io.github.test/server")
    with pytest.raises(MlflowException, match="not found"):
        store.create_mcp_access_endpoint(
            "io.github.test/server", "https://mcp.example.com", server_version="9.9.0"
        )


def test_create_mcp_access_endpoint_nonexistent_alias_raises(store):
    _setup_server(store, "io.github.test/server")
    with pytest.raises(MlflowException, match="not found"):
        store.create_mcp_access_endpoint(
            "io.github.test/server", "https://mcp.example.com", server_alias="fake"
        )


def test_create_mcp_access_endpoint_deleted_version_raises(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    with pytest.raises(MlflowException, match="deleted MCP server version"):
        store.create_mcp_access_endpoint(
            "io.github.test/server", "https://mcp.example.com", server_version="1.0.0"
        )


def test_create_mcp_access_endpoint_neither_raises(store):
    _setup_server(store, "io.github.test/server")
    with pytest.raises(MlflowException, match="Exactly one"):
        store.create_mcp_access_endpoint("io.github.test/server", "https://mcp.example.com")


def test_create_mcp_access_endpoint_both_raises(store):
    _setup_server(store, "io.github.test/server", aliases={"stable": "1.0.0"})
    with pytest.raises(MlflowException, match="Exactly one"):
        store.create_mcp_access_endpoint(
            "io.github.test/server",
            "https://mcp.example.com",
            server_version="1.0.0",
            server_alias="stable",
        )


def test_create_mcp_access_endpoint_server_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store.create_mcp_access_endpoint(
            "io.github.test/nonexistent", "https://mcp.example.com", server_version="1.0.0"
        )


@pytest.mark.parametrize("blank_url", ["", "   ", "\t"])
def test_create_mcp_access_endpoint_rejects_blank_url(store, blank_url):
    _setup_server(store, "io.github.test/blank-url")
    with pytest.raises(MlflowException, match="cannot be empty or just whitespace"):
        store.create_mcp_access_endpoint(
            "io.github.test/blank-url", blank_url, server_version="1.0.0"
        )


@pytest.mark.parametrize("blank_url", ["", "   "])
def test_update_mcp_access_endpoint_rejects_blank_url(store, blank_url):
    _setup_server(store, "io.github.test/blank-url-upd")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/blank-url-upd", "https://mcp.example.com", server_version="1.0.0"
    )
    with pytest.raises(MlflowException, match="cannot be empty or just whitespace"):
        store.update_mcp_access_endpoint("io.github.test/blank-url-upd", endpoint.id, url=blank_url)


def test_get_mcp_access_endpoint_not_found_raises(store):
    _setup_server(store, "io.github.test/server")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_access_endpoint("io.github.test/server", 99999)


def test_search_mcp_access_endpoints_all(store):
    _setup_server(store, "io.github.test/server1")
    _setup_server(store, "io.github.test/server2")
    store.create_mcp_access_endpoint(
        "io.github.test/server1", "https://a.com", server_version="1.0.0"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server2", "https://b.com", server_version="1.0.0"
    )
    result = store.search_mcp_access_endpoints()
    assert len(result) == 2


def test_search_mcp_access_endpoints_by_server(store):
    _setup_server(store, "io.github.test/server1")
    _setup_server(store, "io.github.test/server2")
    store.create_mcp_access_endpoint(
        "io.github.test/server1", "https://a.com", server_version="1.0.0"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server2", "https://b.com", server_version="1.0.0"
    )
    result = store.search_mcp_access_endpoints(server_name="io.github.test/server1")
    assert len(result) == 1
    assert result[0].server_name == "io.github.test/server1"


def test_search_mcp_access_endpoints_by_version(store):
    _setup_server(store, "io.github.test/server", versions=("1.0.0", "2.0.0"))
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://b.com", server_version="2.0.0"
    )
    result = store.search_mcp_access_endpoints(server_version="1.0.0")
    assert len(result) == 1


def test_delete_mcp_access_endpoint(store):
    _setup_server(store, "io.github.test/server")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    store.delete_mcp_access_endpoint("io.github.test/server", endpoint.id)
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_access_endpoint("io.github.test/server", endpoint.id)


# --- Tags ---


def test_set_and_get_mcp_server_tag(store):
    store.create_mcp_server("io.github.test/server")
    store.set_mcp_server_tag("io.github.test/server", "env", "prod")
    server = store.get_mcp_server("io.github.test/server")
    assert server.tags == {"env": "prod"}


def test_upsert_mcp_server_tag(store):
    store.create_mcp_server("io.github.test/server")
    store.set_mcp_server_tag("io.github.test/server", "env", "dev")
    store.set_mcp_server_tag("io.github.test/server", "env", "prod")
    server = store.get_mcp_server("io.github.test/server")
    assert server.tags == {"env": "prod"}


def test_delete_mcp_server_tag(store):
    store.create_mcp_server("io.github.test/server")
    store.set_mcp_server_tag("io.github.test/server", "env", "prod")
    store.delete_mcp_server_tag("io.github.test/server", "env")
    server = store.get_mcp_server("io.github.test/server")
    assert server.tags == {}


def test_delete_mcp_server_tag_not_found_raises(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server_tag("io.github.test/server", "io.github.test/nonexistent")


def test_set_and_get_mcp_server_version_tag(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.set_mcp_server_version_tag("io.github.test/server", "1.0.0", "env", "prod")
    sv = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert sv.tags == {"env": "prod"}


def test_upsert_mcp_server_version_tag(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.set_mcp_server_version_tag("io.github.test/server", "1.0.0", "env", "dev")
    store.set_mcp_server_version_tag("io.github.test/server", "1.0.0", "env", "prod")
    sv = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert sv.tags == {"env": "prod"}


def test_delete_mcp_server_version_tag(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.set_mcp_server_version_tag("io.github.test/server", "1.0.0", "env", "prod")
    store.delete_mcp_server_version_tag("io.github.test/server", "1.0.0", "env")
    sv = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert sv.tags == {}


def test_set_mcp_server_version_tag_deleted_version_raises(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    with pytest.raises(MlflowException, match="not found") as exc:
        store.set_mcp_server_version_tag("io.github.test/server", "1.0.0", "env", "prod")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_delete_mcp_server_version_tag_deleted_version_raises(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.set_mcp_server_version_tag("io.github.test/server", "1.0.0", "env", "prod")
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    with pytest.raises(MlflowException, match="not found") as exc:
        store.delete_mcp_server_version_tag("io.github.test/server", "1.0.0", "env")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


# --- Aliases ---


def test_set_and_resolve_mcp_server_alias(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")
    sv = store.get_mcp_server_version_by_alias("io.github.test/server", "stable")
    assert sv.version == "1.0.0"


def test_upsert_mcp_server_alias_retargets(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.create_mcp_server_version(_server_json("io.github.test/server", "2.0.0"))
    store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")
    store.set_mcp_server_alias("io.github.test/server", "stable", "2.0.0")
    sv = store.get_mcp_server_version_by_alias("io.github.test/server", "stable")
    assert sv.version == "2.0.0"


def test_set_mcp_server_alias_latest_reserved(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="reserved") as exc:
        store.set_mcp_server_alias("io.github.test/server", "latest", "1.0.0")
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_set_mcp_server_alias_to_deleted_version_raises(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    with pytest.raises(MlflowException, match="Cannot set alias"):
        store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")


def test_delete_mcp_server_alias(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")
    store.delete_mcp_server_alias("io.github.test/server", "stable")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version_by_alias("io.github.test/server", "stable")


def test_delete_mcp_server_alias_cleans_up_alias_endpoints(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://alias.example.com",
        server_alias="stable",
    )
    store.delete_mcp_server_alias("io.github.test/server", "stable")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_access_endpoint("io.github.test/server", endpoint.id)
    assert len(store.search_mcp_access_endpoints(server_name="io.github.test/server")) == 0
    server = store.get_mcp_server("io.github.test/server")
    assert server.aliases == {}
    assert server.access_endpoints == []


def test_delete_mcp_server_alias_not_found_raises(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server_alias("io.github.test/server", "io.github.test/nonexistent")


def test_set_mcp_server_alias_nonexistent_version_raises(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="not found"):
        store.set_mcp_server_alias("io.github.test/server", "stable", "io.github.test/nonexistent")


def test_aliases_appear_on_server(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.set_mcp_server_alias("io.github.test/server", "stable", "1.0.0")
    server = store.get_mcp_server("io.github.test/server")
    assert "stable" in server.aliases
    assert server.aliases["stable"] == "1.0.0"


# --- Status transitions ---


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (MCPStatus.DRAFT, MCPStatus.ACTIVE),
        (MCPStatus.DRAFT, MCPStatus.DELETED),
        (MCPStatus.ACTIVE, MCPStatus.DRAFT),
        (MCPStatus.ACTIVE, MCPStatus.DEPRECATED),
        (MCPStatus.DEPRECATED, MCPStatus.ACTIVE),
        (MCPStatus.DEPRECATED, MCPStatus.DELETED),
    ],
)
def test_valid_status_transitions(store, from_status, to_status):
    _create_version(store, status=from_status)
    updated = store.update_mcp_server_version(
        "io.github.test/servererver", "1.0.0", status=to_status
    )
    assert updated.status == to_status


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (MCPStatus.ACTIVE, MCPStatus.DELETED),
        (MCPStatus.DRAFT, MCPStatus.DEPRECATED),
    ],
)
def test_invalid_status_transitions(store, from_status, to_status):
    store.create_mcp_server_version(_server_json(), status=from_status)
    with pytest.raises(MlflowException, match="Invalid status transition"):
        store.update_mcp_server_version("io.github.test/servererver", "1.0.0", status=to_status)


# --- filter_string ---


def test_search_mcp_servers_filter_by_name(store):
    store.create_mcp_server("io.github.test/alpha")
    store.create_mcp_server("io.github.test/beta")
    store.create_mcp_server("io.github.other/gamma")
    result = store.search_mcp_servers(filter_string="name = 'io.github.test/alpha'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/alpha"


def test_search_mcp_servers_filter_by_name_like(store):
    store.create_mcp_server("io.github.test/alpha")
    store.create_mcp_server("io.github.test/beta")
    store.create_mcp_server("io.github.other/gamma")
    result = store.search_mcp_servers(filter_string="name LIKE '%test%'")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"io.github.test/alpha", "io.github.test/beta"}


def test_search_mcp_servers_filter_by_display_name(store):
    store.create_mcp_server("io.github.test/alpha")
    store.update_mcp_server("io.github.test/alpha", display_name="Pretty Alpha")
    result = store.search_mcp_servers(filter_string="display_name ILIKE '%pretty%'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/alpha"


def test_search_mcp_servers_filter_by_tag(store):
    store.create_mcp_server("io.github.test/server1")
    store.create_mcp_server("io.github.test/server2")
    store.create_mcp_server("io.github.test/server3")
    store.set_mcp_server_tag("io.github.test/server1", "env", "prod")
    store.set_mcp_server_tag("io.github.test/server2", "env", "staging")
    store.set_mcp_server_tag("io.github.test/server3", "env", "prod")
    result = store.search_mcp_servers(filter_string="tags.env = 'prod'")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"io.github.test/server1", "io.github.test/server3"}


def test_search_mcp_servers_filter_by_multiple_tags(store):
    store.create_mcp_server("io.github.test/server1")
    store.create_mcp_server("io.github.test/server2")
    store.set_mcp_server_tag("io.github.test/server1", "env", "prod")
    store.set_mcp_server_tag("io.github.test/server1", "team", "ai-hub")
    store.set_mcp_server_tag("io.github.test/server2", "env", "prod")
    store.set_mcp_server_tag("io.github.test/server2", "team", "other")
    result = store.search_mcp_servers(filter_string="tags.env = 'prod' AND tags.team = 'ai-hub'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/server1"


def test_search_mcp_servers_filter_attribute_and_tag(store):
    store.create_mcp_server("io.github.test/alpha")
    store.create_mcp_server("io.github.test/beta")
    store.set_mcp_server_tag("io.github.test/alpha", "env", "prod")
    store.set_mcp_server_tag("io.github.test/beta", "env", "prod")
    result = store.search_mcp_servers(filter_string="name LIKE '%alpha%' AND tags.env = 'prod'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/alpha"


def test_search_mcp_servers_filter_has_access_endpoints_true(store):
    _setup_server(store, "io.github.test/server1")
    store.create_mcp_server("io.github.test/server2")
    _setup_server(store, "io.github.test/server3")
    store.create_mcp_access_endpoint(
        "io.github.test/server1", "https://a.com", server_version="1.0.0"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server3", "https://b.com", server_version="1.0.0"
    )
    result = store.search_mcp_servers(filter_string="has_access_endpoints = 'true'")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"io.github.test/server1", "io.github.test/server3"}


def test_search_mcp_servers_filter_by_status_uses_resolved_latest(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server1", "1.0.0"), status=MCPStatus.ACTIVE
    )
    _create_version(store, "io.github.test/server1", "2.0.0", status=MCPStatus.DEPRECATED)
    store.create_mcp_server_version(
        _server_json("io.github.test/server2", "1.0.0"), status=MCPStatus.ACTIVE
    )
    result = store.search_mcp_servers(filter_string="status = 'active'")
    names = {s.name for s in result}
    assert names == {"io.github.test/server1", "io.github.test/server2"}


def test_search_mcp_servers_filter_by_status_with_active_version(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    _create_version(store, "io.github.test/server", "2.0.0", status=MCPStatus.DEPRECATED)
    result = store.search_mcp_servers(filter_string="status = 'active'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/server"


def test_search_mcp_servers_filter_by_status_no_versions(store):
    store.create_mcp_server("io.github.test/server1")
    store.create_mcp_server_version(
        _server_json("io.github.test/server2", "1.0.0"), status=MCPStatus.ACTIVE
    )
    result = store.search_mcp_servers(filter_string="status = 'active'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/server2"


def test_search_mcp_servers_empty_filter_string(store):
    store.create_mcp_server("io.github.test/server1")
    store.create_mcp_server("io.github.test/server2")
    result = store.search_mcp_servers(filter_string="")
    assert len(result) == 2


def test_search_mcp_servers_filter_has_access_endpoints_false(store):
    _setup_server(store, "io.github.test/server1")
    store.create_mcp_server("io.github.test/server2")
    _setup_server(store, "io.github.test/server3")
    store.create_mcp_access_endpoint(
        "io.github.test/server1", "https://a.com", server_version="1.0.0"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server3", "https://b.com", server_version="1.0.0"
    )
    result = store.search_mcp_servers(filter_string="has_access_endpoints = 'false'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/server2"


def test_search_mcp_access_endpoints_filter_by_transport_type(store):
    _setup_server(store, "io.github.test/server")
    store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://a.com",
        server_version="1.0.0",
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://b.com",
        server_version="1.0.0",
        transport_type=MCPRemoteTransportType.SSE,
    )
    result = store.search_mcp_access_endpoints(filter_string="transport_type = 'streamable-http'")
    assert len(result) == 1
    assert result[0].url == "https://a.com"


def test_search_mcp_servers_filter_by_status_in(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server1", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server2", "1.0.0"), status=MCPStatus.DRAFT
    )
    _create_version(store, "io.github.test/server3", "1.0.0", status=MCPStatus.DEPRECATED)
    result = store.search_mcp_servers(filter_string="status IN ('active', 'deprecated')")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"io.github.test/server1", "io.github.test/server3"}


def test_search_mcp_server_versions_filter_by_status_in(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "2.0.0"), status=MCPStatus.DRAFT
    )
    _create_version(store, "io.github.test/server", "3.0.0", status=MCPStatus.DEPRECATED)
    result = store.search_mcp_server_versions(
        "io.github.test/server", filter_string="status IN ('active', 'deprecated')"
    )
    assert len(result) == 2
    versions = {v.version for v in result}
    assert versions == {"1.0.0", "3.0.0"}


def test_search_mcp_servers_filter_by_status(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server1", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server2", "1.0.0"), status=MCPStatus.DRAFT
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server3", "1.0.0"), status=MCPStatus.ACTIVE
    )
    result = store.search_mcp_servers(filter_string="status = 'active'")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"io.github.test/server1", "io.github.test/server3"}


def test_search_mcp_server_versions_filter_by_status(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "2.0.0"), status=MCPStatus.DRAFT
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "3.0.0"), status=MCPStatus.ACTIVE
    )
    result = store.search_mcp_server_versions(
        "io.github.test/server", filter_string="status = 'active'"
    )
    assert len(result) == 2
    versions = {v.version for v in result}
    assert versions == {"1.0.0", "3.0.0"}


def test_search_mcp_servers_filter_invalid_attribute(store):
    with pytest.raises(MlflowException, match="Invalid attribute key"):
        store.search_mcp_servers(filter_string="bogus = 'x'")


# --- get_mcp_access_endpoint happy path ---


def test_get_mcp_access_endpoint(store):
    _setup_server(store, "io.github.test/server")
    created = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    fetched = store.get_mcp_access_endpoint("io.github.test/server", created.id)
    assert fetched.id == created.id
    assert fetched.server_name == "io.github.test/server"
    assert fetched.url == "https://a.com"
    assert fetched.server_version == "1.0.0"
    assert fetched.transport_type == MCPRemoteTransportType.STREAMABLE_HTTP


# --- update_mcp_access_endpoint ---


def test_update_mcp_access_endpoint_version_clears_alias(store):
    _setup_server(
        store, "io.github.test/server", versions=("1.0.0", "2.0.0"), aliases={"stable": "1.0.0"}
    )
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_alias="stable"
    )
    assert endpoint.server_alias == "stable"
    updated = store.update_mcp_access_endpoint(
        "io.github.test/server",
        endpoint.id,
        server_version="2.0.0",
        last_updated_by="bob",
    )
    assert updated.server_version == "2.0.0"
    assert updated.server_alias is None
    assert updated.last_updated_by == "bob"


def test_update_mcp_access_endpoint_alias_clears_version(store):
    _setup_server(store, "io.github.test/server", aliases={"prod": "1.0.0"})
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    assert endpoint.server_version == "1.0.0"
    updated = store.update_mcp_access_endpoint(
        "io.github.test/server", endpoint.id, server_alias="prod"
    )
    assert updated.server_alias == "prod"
    assert updated.server_version is None


def test_update_mcp_access_endpoint_endpoint_and_transport(store):
    _setup_server(store, "io.github.test/server")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    updated = store.update_mcp_access_endpoint(
        "io.github.test/server",
        endpoint.id,
        url="https://b.com",
        transport_type=MCPRemoteTransportType.SSE,
    )
    assert updated.url == "https://b.com"
    assert updated.transport_type == MCPRemoteTransportType.SSE


def test_update_mcp_access_endpoint_url_none_raises(store):
    _setup_server(store, "io.github.test/server")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    with pytest.raises(MlflowException, match="url cannot be None") as exc:
        store.update_mcp_access_endpoint("io.github.test/server", endpoint.id, url=None)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_mcp_access_endpoint_both_version_and_alias_raises(store):
    _setup_server(store, "io.github.test/server", aliases={"stable": "1.0.0"})
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    with pytest.raises(MlflowException, match="Cannot set both"):
        store.update_mcp_access_endpoint(
            "io.github.test/server",
            endpoint.id,
            server_version="1.0.0",
            server_alias="stable",
        )


def test_update_mcp_access_endpoint_nonexistent_version_raises(store):
    _setup_server(store, "io.github.test/server")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_access_endpoint(
            "io.github.test/server", endpoint.id, server_version="9.9.0"
        )


def test_update_mcp_access_endpoint_deleted_version_raises(store):
    _setup_server(store, "io.github.test/server", versions=("1.0.0", "2.0.0"))
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    store.delete_mcp_server_version("io.github.test/server", "2.0.0")
    with pytest.raises(MlflowException, match="deleted MCP server version"):
        store.update_mcp_access_endpoint(
            "io.github.test/server", endpoint.id, server_version="2.0.0"
        )


def test_update_mcp_access_endpoint_nonexistent_alias_raises(store):
    _setup_server(store, "io.github.test/server")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://a.com", server_version="1.0.0"
    )
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_access_endpoint("io.github.test/server", endpoint.id, server_alias="fake")


def test_update_mcp_access_endpoint_wrong_server_raises(store):
    _setup_server(store, "io.github.test/server1")
    _setup_server(store, "io.github.test/server2")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server1", "https://a.com", server_version="1.0.0"
    )
    with pytest.raises(MlflowException, match="does not belong"):
        store.update_mcp_access_endpoint("io.github.test/server2", endpoint.id, url="https://b.com")


# --- search_mcp_server_versions pagination ---


def test_search_mcp_server_versions_pagination(store):
    for i in range(5):
        store.create_mcp_server_version(_server_json("io.github.test/server", f"{i}.0.0"))
    page1 = store.search_mcp_server_versions("io.github.test/server", max_results=2)
    assert len(page1) == 2
    assert page1.token is not None
    page2 = store.search_mcp_server_versions(
        "io.github.test/server", max_results=2, page_token=page1.token
    )
    assert len(page2) == 2
    assert page2.token is not None
    page3 = store.search_mcp_server_versions(
        "io.github.test/server", max_results=2, page_token=page2.token
    )
    assert len(page3) == 1
    assert page3.token is None


def test_search_mcp_server_versions_pagination_same_timestamp_uses_version_tiebreaker(
    store, monkeypatch
):
    monkeypatch.setattr(
        "mlflow.store.tracking.mcp_server_registry.sqlalchemy_mixin.get_current_time_millis",
        lambda: 1000,
    )
    for version in ("1.0.0-beta", "1.0.0-alpha", "1.0.0-gamma"):
        store.create_mcp_server_version(_server_json("io.github.test/server", version))

    page1 = store.search_mcp_server_versions("io.github.test/server", max_results=1)
    page2 = store.search_mcp_server_versions(
        "io.github.test/server", max_results=1, page_token=page1.token
    )
    page3 = store.search_mcp_server_versions(
        "io.github.test/server", max_results=1, page_token=page2.token
    )

    assert [v.version for v in page1] == ["1.0.0-alpha"]
    assert [v.version for v in page2] == ["1.0.0-beta"]
    assert [v.version for v in page3] == ["1.0.0-gamma"]
    assert page3.token is None


# --- order_by ---


def test_search_mcp_servers_order_by_name_desc(store):
    store.create_mcp_server("io.github.test/alpha")
    store.create_mcp_server("io.github.test/beta")
    store.create_mcp_server("io.github.test/gamma")
    result = store.search_mcp_servers(order_by=["name DESC"])
    names = [s.name for s in result]
    assert names == ["io.github.test/gamma", "io.github.test/beta", "io.github.test/alpha"]


def test_search_mcp_servers_order_by_name_asc(store):
    store.create_mcp_server("io.github.test/gamma")
    store.create_mcp_server("io.github.test/alpha")
    store.create_mcp_server("io.github.test/beta")
    result = store.search_mcp_servers(order_by=["name ASC"])
    names = [s.name for s in result]
    assert names == ["io.github.test/alpha", "io.github.test/beta", "io.github.test/gamma"]


def test_search_mcp_servers_order_by_default_is_name_asc(store):
    store.create_mcp_server("io.github.test/gamma")
    store.create_mcp_server("io.github.test/alpha")
    store.create_mcp_server("io.github.test/beta")
    result = store.search_mcp_servers()
    names = [s.name for s in result]
    assert names == ["io.github.test/alpha", "io.github.test/beta", "io.github.test/gamma"]


def test_search_mcp_servers_order_by_invalid_key(store):
    with pytest.raises(MlflowException, match="Invalid order_by key"):
        store.search_mcp_servers(order_by=["bogus ASC"])


def test_search_mcp_servers_order_by_duplicate_key(store):
    with pytest.raises(MlflowException, match="Duplicate order_by"):
        store.search_mcp_servers(order_by=["name ASC", "name DESC"])


# --- Additional coverage ---


def test_delete_mcp_access_endpoint_wrong_server_raises(store):
    _setup_server(store, "io.github.test/server1")
    _setup_server(store, "io.github.test/server2")
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server1", "https://a.com", server_version="1.0.0"
    )
    with pytest.raises(MlflowException, match="does not belong"):
        store.delete_mcp_access_endpoint("io.github.test/server2", endpoint.id)


def test_delete_mcp_server_version_tag_not_found_raises(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server_version_tag(
            "io.github.test/server", "1.0.0", "io.github.test/nonexistent"
        )


def test_update_mcp_server_version_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_server_version("io.github.test/nonexistent", "1.0.0", display_name="x")


def test_search_mcp_access_endpoints_pagination(store):
    _setup_server(store, "io.github.test/server")
    for i in range(5):
        store.create_mcp_access_endpoint(
            "io.github.test/server", f"https://{i}.com", server_version="1.0.0"
        )
    page1 = store.search_mcp_access_endpoints(max_results=2)
    assert len(page1) == 2
    assert page1.token is not None
    page2 = store.search_mcp_access_endpoints(max_results=2, page_token=page1.token)
    assert len(page2) == 2
    assert page2.token is not None
    page3 = store.search_mcp_access_endpoints(max_results=2, page_token=page2.token)
    assert len(page3) == 1
    assert page3.token is None


def test_search_mcp_server_versions_order_by(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0-alpha"))
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0-beta"))
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0-gamma"))
    result = store.search_mcp_server_versions("io.github.test/server", order_by=["`version` DESC"])
    versions = [v.version for v in result]
    assert versions == ["1.0.0-gamma", "1.0.0-beta", "1.0.0-alpha"]


def test_search_mcp_server_versions_order_by_version_uses_semver_desc(store):
    for version in ("1.2.0-alpha", "1.2.0", "1.10.0"):
        store.create_mcp_server_version(_server_json("io.github.test/serveremver-desc", version))

    result = store.search_mcp_server_versions(
        "io.github.test/serveremver-desc",
        order_by=["`version` DESC"],
    )
    versions = [v.version for v in result]
    assert versions == ["1.10.0", "1.2.0", "1.2.0-alpha"]


def test_search_mcp_server_versions_order_by_version_uses_semver_asc(store):
    for version in ("1.2.0-alpha", "1.2.0", "1.10.0"):
        store.create_mcp_server_version(_server_json("io.github.test/serveremver-asc", version))

    result = store.search_mcp_server_versions(
        "io.github.test/serveremver-asc",
        order_by=["`version` ASC"],
    )
    versions = [v.version for v in result]
    assert versions == ["1.2.0-alpha", "1.2.0", "1.10.0"]


def test_search_mcp_server_versions_order_by_version_ignores_build_metadata_precedence(store):
    for version in ("1.0.1", "1.0.0+aaa", "1.0.0+zzz"):
        store.create_mcp_server_version(
            _server_json("io.github.test/serveremver-build-desc", version)
        )

    result = store.search_mcp_server_versions(
        "io.github.test/serveremver-build-desc",
        order_by=["`version` DESC"],
    )
    versions = [v.version for v in result]
    assert versions[0] == "1.0.1"
    assert set(versions[1:]) == {"1.0.0+aaa", "1.0.0+zzz"}


def test_search_mcp_server_versions_filter_by_version_equality_uses_exact_string_match(store):
    for version in ("1.0.0-alpha+aaa", "1.0.0-alpha+zzz", "1.0.0"):
        store.create_mcp_server_version(_server_json("io.github.test/serveremver-eq", version))

    result = store.search_mcp_server_versions(
        "io.github.test/serveremver-eq",
        filter_string="version = '1.0.0-alpha+aaa'",
    )
    versions = {v.version for v in result}
    assert versions == {"1.0.0-alpha+aaa"}


def test_search_mcp_server_versions_filter_by_version_inequality_uses_semver_precedence(store):
    for version in ("1.2.0-alpha", "1.2.0", "1.10.0"):
        store.create_mcp_server_version(_server_json("io.github.test/serveremver-gt", version))

    result = store.search_mcp_server_versions(
        "io.github.test/serveremver-gt",
        filter_string="version > '1.2.0-alpha'",
    )
    versions = {v.version for v in result}
    assert versions == {"1.2.0", "1.10.0"}


def test_search_mcp_server_versions_filter_by_version_rejects_like(store):
    store.create_mcp_server_version(_server_json("io.github.test/serveremver-like", "1.2.3"))

    with pytest.raises(MlflowException, match="version only supports semantic comparators"):
        store.search_mcp_server_versions(
            "io.github.test/serveremver-like",
            filter_string="version LIKE '1.%'",
        )


def test_create_mcp_access_endpoint_with_latest_alias(store):
    # Create server with an active version
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )

    # Should be able to create an endpoint with server_alias="latest"
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )

    # Verify the endpoint was created and resolves to the latest version
    assert endpoint.server_alias == "latest"
    assert endpoint.server_version is None
    assert endpoint.resolved_version is not None
    assert endpoint.resolved_version.version == "1.0.0"

    # Create a newer version and verify "latest" now resolves to it
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "2.0.0"), status=MCPStatus.ACTIVE
    )

    # Retrieve the endpoint again to check resolution
    retrieved_endpoint = store.get_mcp_access_endpoint("io.github.test/server", endpoint.id)
    assert retrieved_endpoint.resolved_version.version == "2.0.0"


def test_search_mcp_access_endpoints_with_latest_alias(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )

    # Search should find the endpoint and resolve its version
    endpoints = store.search_mcp_access_endpoints(
        server_name="io.github.test/server",
        server_alias="latest",
    )
    assert len(endpoints) == 1
    assert endpoints[0].server_alias == "latest"
    assert endpoints[0].resolved_version.version == "1.0.0"


def test_get_latest_version_without_active_version_falls_back_to_non_active(store):
    _create_version(store, "io.github.test/server", "1.0.0", status=MCPStatus.DEPRECATED)
    latest = store.get_latest_mcp_server_version("io.github.test/server")
    assert latest.version == "1.0.0"


def test_create_endpoint_latest_alias_uses_non_active_fallback(store):
    _create_version(store, "io.github.test/server", "1.0.0", status=MCPStatus.DEPRECATED)
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )
    assert endpoint.resolved_version is not None
    assert endpoint.resolved_version.version == "1.0.0"


def test_search_endpoint_latest_alias_remains_resolvable_without_active_version(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )
    assert endpoint.resolved_version.version == "1.0.0"

    store.update_mcp_server_version("io.github.test/server", "1.0.0", status=MCPStatus.DEPRECATED)

    endpoints = store.search_mcp_access_endpoints(server_name="io.github.test/server")
    assert len(endpoints) == 1
    assert endpoints[0].id == endpoint.id
    assert endpoints[0].resolved_version is not None
    assert endpoints[0].resolved_version.version == "1.0.0"


def test_update_last_eligible_version_to_draft_keeps_latest_alias_endpoints_when_resolvable(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )

    store.update_mcp_server_version("io.github.test/server", "1.0.0", status=MCPStatus.DRAFT)

    endpoints = store.search_mcp_access_endpoints(server_name="io.github.test/server")
    assert len(endpoints) == 1
    server = store.get_mcp_server("io.github.test/server")
    assert len(server.access_endpoints) == 1
    persisted = store.get_mcp_access_endpoint("io.github.test/server", endpoint.id)
    assert persisted.resolved_version is not None
    assert persisted.resolved_version.version == "1.0.0"


def test_delete_last_eligible_version_cleans_up_latest_alias_endpoints(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server",
        "https://latest.example.com",
        server_alias="latest",
    )
    store.update_mcp_server_version("io.github.test/server", "1.0.0", status=MCPStatus.DEPRECATED)

    store.delete_mcp_server_version("io.github.test/server", "1.0.0")

    assert store.search_mcp_access_endpoints(server_name="io.github.test/server") == []
    server = store.get_mcp_server("io.github.test/server")
    assert server.access_endpoints == []
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_access_endpoint("io.github.test/server", endpoint.id)


def test_search_unfiltered_returns_all_endpoint_types(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "prod", "1.0.0")

    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://direct.example.com", server_version="1.0.0"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://alias.example.com", server_alias="prod"
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://latest.example.com", server_alias="latest"
    )

    endpoints = store.search_mcp_access_endpoints(server_name="io.github.test/server")
    assert len(endpoints) == 3
    urls = {e.url for e in endpoints}
    assert urls == {
        "https://direct.example.com",
        "https://alias.example.com",
        "https://latest.example.com",
    }
    for e in endpoints:
        assert e.resolved_version is not None
        assert e.resolved_version.version == "1.0.0"


def test_update_endpoint_to_latest_alias(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    endpoint = store.create_mcp_access_endpoint(
        "io.github.test/server", "https://example.com", server_version="1.0.0"
    )
    assert endpoint.server_version == "1.0.0"

    updated = store.update_mcp_access_endpoint(
        "io.github.test/server",
        endpoint.id,
        server_alias="latest",
    )
    assert updated.server_alias == "latest"
    assert updated.server_version is None
    assert updated.resolved_version is not None
    assert updated.resolved_version.version == "1.0.0"


def test_get_mcp_server_includes_latest_alias_endpoint(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server", "https://latest.example.com", server_alias="latest"
    )
    server = store.get_mcp_server("io.github.test/server")
    assert len(server.access_endpoints) == 1
    e = server.access_endpoints[0]
    assert e.server_alias == "latest"
    assert e.resolved_version is not None
    assert e.resolved_version.version == "1.0.0"


def test_search_mcp_servers_has_access_endpoints_with_latest_alias(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server1", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_access_endpoint(
        "io.github.test/server1", "https://latest.example.com", server_alias="latest"
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/server2", "1.0.0"), status=MCPStatus.ACTIVE
    )

    result = store.search_mcp_servers(filter_string="has_access_endpoints = 'true'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/server1"
    assert len(result[0].access_endpoints) == 1
    assert result[0].access_endpoints[0].resolved_version.version == "1.0.0"


def test_deleted_versions_excluded_from_get(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version("io.github.test/server", "1.0.0")


def test_deleted_versions_excluded_from_search(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "2.0.0"), status=MCPStatus.ACTIVE
    )
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    result = store.search_mcp_server_versions("io.github.test/server")
    assert len(result) == 1
    assert result[0].version == "2.0.0"


def test_endpoint_resolved_version_direct(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://example.com",
        server_version="1.0.0",
    )
    endpoint = store.search_mcp_access_endpoints(server_name="io.github.test/server")[0]
    assert endpoint.resolved_version is not None
    assert endpoint.resolved_version.version == "1.0.0"
    assert endpoint.resolved_version.name == "io.github.test/server"


def test_endpoint_resolved_version_via_alias(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "prod", "1.0.0")
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server", url="https://example.com", server_alias="prod"
    )
    endpoint = store.search_mcp_access_endpoints(server_name="io.github.test/server")[0]
    assert endpoint.resolved_version is not None
    assert endpoint.resolved_version.version == "1.0.0"


def test_endpoint_resolved_version_on_get(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    endpoint = store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://example.com",
        server_version="1.0.0",
    )
    fetched = store.get_mcp_access_endpoint("io.github.test/server", endpoint.id)
    assert fetched.resolved_version is not None
    assert fetched.resolved_version.version == "1.0.0"


def test_endpoint_resolved_version_on_get_via_alias(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "prod", "1.0.0")
    endpoint = store.create_mcp_access_endpoint(
        server_name="io.github.test/server", url="https://example.com", server_alias="prod"
    )
    fetched = store.get_mcp_access_endpoint("io.github.test/server", endpoint.id)
    assert fetched.resolved_version is not None
    assert fetched.resolved_version.version == "1.0.0"


def test_search_endpoints_filter_by_status(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.create_mcp_server_version(_server_json("io.github.test/server", "2.0.0"))
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://a.example.com",
        server_version="1.0.0",
    )
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://b.example.com",
        server_version="2.0.0",
    )
    result = store.search_mcp_access_endpoints(filter_string="status = 'active'")
    assert len(result) == 1
    assert result[0].url == "https://a.example.com"


def test_search_endpoints_scoped_to_server_version_resolves_version(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "prod", "1.0.0")
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://direct.example.com",
        server_version="1.0.0",
    )
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://alias.example.com",
        server_alias="prod",
    )
    result = store.search_mcp_access_endpoints(
        server_name="io.github.test/server", server_version="1.0.0"
    )
    assert len(result) == 1
    assert result[0].url == "https://direct.example.com"
    assert result[0].resolved_version is not None
    assert result[0].resolved_version.version == "1.0.0"


def test_search_endpoints_scoped_to_server_alias_resolves_version(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    store.set_mcp_server_alias("io.github.test/server", "prod", "1.0.0")
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://direct.example.com",
        server_version="1.0.0",
    )
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://alias.example.com",
        server_alias="prod",
    )
    result = store.search_mcp_access_endpoints(
        server_name="io.github.test/server", server_alias="prod"
    )
    assert len(result) == 1
    assert result[0].url == "https://alias.example.com"
    assert result[0].resolved_version is not None
    assert result[0].resolved_version.version == "1.0.0"


def test_search_servers_numeric_timestamp_filter(store):
    store.create_mcp_server("io.github.test/server")
    result = store.search_mcp_servers(filter_string="created_at > 0")
    assert len(result) == 1


def test_create_endpoint_returns_resolved_version(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    endpoint = store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://example.com",
        server_version="1.0.0",
    )
    assert endpoint.resolved_version is not None
    assert endpoint.resolved_version.version == "1.0.0"


def test_update_endpoint_returns_resolved_version(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "1.0.0"), status=MCPStatus.ACTIVE
    )
    endpoint = store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://example.com",
        server_version="1.0.0",
    )
    updated = store.update_mcp_access_endpoint(
        server_name="io.github.test/server",
        endpoint_id=endpoint.id,
        url="https://new.example.com",
    )
    assert updated.resolved_version is not None
    assert updated.resolved_version.version == "1.0.0"


def test_endpoint_to_deleted_version_hidden(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://example.com",
        server_version="1.0.0",
    )
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    result = store.search_mcp_access_endpoints(server_name="io.github.test/server")
    assert len(result) == 0


def test_has_access_endpoints_excludes_stale_endpoints(store):
    store.create_mcp_server_version(_server_json("io.github.test/server", "1.0.0"))
    store.create_mcp_access_endpoint(
        server_name="io.github.test/server",
        url="https://example.com",
        server_version="1.0.0",
    )
    result = store.search_mcp_servers(filter_string="has_access_endpoints = 'true'")
    assert len(result) == 1
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    result = store.search_mcp_servers(filter_string="has_access_endpoints = 'true'")
    assert len(result) == 0


def test_has_access_endpoints_duplicate_rejected(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="Invalid"):
        store.search_mcp_servers(
            filter_string="has_access_endpoints = true AND has_access_endpoints = false"
        )
