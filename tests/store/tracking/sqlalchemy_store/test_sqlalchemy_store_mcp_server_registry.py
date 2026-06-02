import pytest

from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus, MCPTool
from mlflow.exceptions import MlflowException

pytestmark = pytest.mark.notrackingurimock


def _server_json(name="io.github.test/server", version="1.0.0"):
    return {"name": name, "version": version, "title": f"Test {name}"}


def _setup_server(store, name, versions=("1.0",), aliases=None):
    """Create a server with versions and optional aliases for binding tests."""
    for v in versions:
        store.create_mcp_server_version(_server_json(name, v))
    for alias, ver in (aliases or {}).items():
        store.set_mcp_server_alias(name, alias, ver)


# --- MCPServer CRUD ---


def test_create_mcp_server(store):
    server = store.create_mcp_server("io.github.test/server", description="A test server")
    assert server.name == "io.github.test/server"
    assert server.description == "A test server"
    assert server.creation_timestamp is not None


def test_create_mcp_server_duplicate_raises(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_mcp_server("io.github.test/server")
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_mcp_server_empty_name_raises(store):
    with pytest.raises(MlflowException, match="must not be empty") as exc:
        store.create_mcp_server("")
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_mcp_server_with_icons(store):
    icons = [{"src": "https://example.com/icon.png", "sizes": "32x32"}]
    server = store.create_mcp_server("io.github.test/server", icons=icons)
    assert server.icons == icons


def test_get_mcp_server(store):
    store.create_mcp_server("io.github.test/server", description="desc")
    server = store.get_mcp_server("io.github.test/server")
    assert server.name == "io.github.test/server"
    assert server.description == "desc"


def test_get_mcp_server_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_mcp_server("nonexistent")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_get_mcp_server_with_tags(store):
    store.create_mcp_server("io.github.test/server")
    store.set_mcp_server_tag("io.github.test/server", "team", "platform")
    server = store.get_mcp_server("io.github.test/server")
    assert server.tags == {"team": "platform"}


def test_get_mcp_server_access_bindings_include_resolved_versions(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_alias("s", "prod", "1.0")
    store.create_mcp_access_binding("s", "https://direct.example.com", server_version="1.0")
    store.create_mcp_access_binding("s", "https://alias.example.com", server_alias="prod")
    server = store.get_mcp_server("s")
    bindings = {b.endpoint_url: b for b in server.access_bindings}
    assert bindings["https://direct.example.com"].resolved_version is not None
    assert bindings["https://direct.example.com"].resolved_version.version == "1.0"
    assert bindings["https://alias.example.com"].resolved_version is not None
    assert bindings["https://alias.example.com"].resolved_version.version == "1.0"


def test_search_mcp_servers_access_bindings_include_resolved_versions(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_alias("s", "prod", "1.0")
    store.create_mcp_access_binding("s", "https://alias.example.com", server_alias="prod")
    server = store.search_mcp_servers()[0]
    assert len(server.access_bindings) == 1
    assert server.access_bindings[0].resolved_version is not None
    assert server.access_bindings[0].resolved_version.version == "1.0"


def test_search_mcp_servers_empty(store):
    result = store.search_mcp_servers()
    assert len(result) == 0
    assert result.token is None


def test_search_mcp_servers_returns_all(store):
    store.create_mcp_server("server-a")
    store.create_mcp_server("server-b")
    result = store.search_mcp_servers()
    assert len(result) == 2


def test_search_mcp_servers_pagination(store):
    for i in range(5):
        store.create_mcp_server(f"server-{i:02d}")
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
    store.create_mcp_server("io.github.test/server", description="old")
    updated = store.update_mcp_server("io.github.test/server", description="new")
    assert updated.description == "new"


def test_update_mcp_server_display_name(store):
    store.create_mcp_server("io.github.test/server")
    updated = store.update_mcp_server("io.github.test/server", display_name="My Server")
    assert updated.display_name == "My Server"


def test_update_mcp_server_returns_complete_entity(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_tag("s", "env", "prod")
    store.set_mcp_server_alias("s", "stable", "1.0")
    updated = store.update_mcp_server("s", description="new desc")
    assert updated.description == "new desc"
    assert updated.status == MCPStatus.ACTIVE
    assert updated.tags == {"env": "prod"}
    assert updated.aliases == {"stable": "1.0"}


def test_update_mcp_server_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_server("nonexistent", description="x")


def test_update_mcp_server_latest_version(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/server", "2.0.0"), status=MCPStatus.ACTIVE
    )
    updated = store.update_mcp_server("io.github.test/server", latest_version="2.0.0")
    assert updated.latest_version == "2.0.0"


def test_update_mcp_server_latest_version_nonexistent(store):
    store.create_mcp_server("s")
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_server("s", latest_version="9.9.9")


def test_update_mcp_server_latest_version_draft_rejected(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    with pytest.raises(MlflowException, match="Cannot pin"):
        store.update_mcp_server("s", latest_version="1.0")


def test_get_mcp_server_resolved_status(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.DRAFT)
    server = store.get_mcp_server("s")
    assert server.status == MCPStatus.ACTIVE


def test_get_mcp_server_resolved_status_pinned(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.DEPRECATED)
    store.update_mcp_server("s", latest_version="2.0")
    server = store.get_mcp_server("s")
    assert server.status == MCPStatus.DEPRECATED
    assert server.latest_version == "2.0"


def test_get_mcp_server_resolved_status_no_versions(store):
    store.create_mcp_server("s")
    server = store.get_mcp_server("s")
    assert server.status is None


def test_delete_mcp_server(store):
    store.create_mcp_server("io.github.test/server")
    store.delete_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server("io.github.test/server")


def test_delete_mcp_server_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server("nonexistent")


def test_delete_mcp_server_cascades_to_versions(store):
    store.create_mcp_server_version(_server_json("s1", "1.0"))
    store.delete_mcp_server("s1")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version("s1", "1.0")


def test_delete_mcp_server_cascades_to_tags(store):
    store.create_mcp_server("s1")
    store.set_mcp_server_tag("s1", "k", "v")
    store.delete_mcp_server("s1")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server("s1")


def test_delete_mcp_server_cascades_to_aliases(store):
    store.create_mcp_server_version(_server_json("s1", "1.0"))
    store.set_mcp_server_alias("s1", "stable", "1.0")
    store.delete_mcp_server("s1")
    # Recreate the server and verify the old alias is gone (was cascade-deleted).
    store.create_mcp_server_version(_server_json("s1", "1.0"))
    server = store.get_mcp_server("s1")
    assert server.aliases == {}


def test_delete_mcp_server_cascades_to_bindings(store):
    _setup_server(store, "s1")
    store.create_mcp_access_binding("s1", "https://a.com", server_version="1.0")
    store.delete_mcp_server("s1")
    result = store.search_mcp_access_bindings(server_name="s1")
    assert len(result) == 0


# --- MCPServerVersion CRUD ---


def test_create_mcp_server_version(store):
    sv = store.create_mcp_server_version(
        _server_json(),
        source="https://github.com/org/repo",
        status=MCPStatus.ACTIVE,
    )
    assert sv.name == "io.github.test/server"
    assert sv.version == "1.0.0"
    assert sv.status == MCPStatus.ACTIVE
    assert sv.source == "https://github.com/org/repo"
    assert sv.server_json == _server_json()


def test_create_mcp_server_version_auto_creates_parent(store):
    store.create_mcp_server_version(_server_json("new-server", "1.0"))
    server = store.get_mcp_server("new-server")
    assert server.name == "new-server"


def test_create_mcp_server_version_with_existing_parent(store):
    store.create_mcp_server("io.github.test/server", description="existing")
    store.create_mcp_server_version(_server_json())
    server = store.get_mcp_server("io.github.test/server")
    assert server.description == "existing"


def test_create_mcp_server_version_duplicate_raises(store):
    store.create_mcp_server_version(_server_json())
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_mcp_server_version(_server_json())
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_mcp_server_version_deleted_version_not_reusable(store):
    original = _server_json("s", "1.0")
    replacement = {
        "name": "s",
        "version": "1.0",
        "title": "Replacement Title",
        "description": "Replacement payload with same identity",
    }
    store.create_mcp_server_version(original)
    store.delete_mcp_server_version("s", "1.0")
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_mcp_server_version(replacement)
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_mcp_server_version_missing_name_raises(store):
    with pytest.raises(MlflowException, match="name.*version") as exc:
        store.create_mcp_server_version({"version": "1.0"})
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_mcp_server_version_missing_version_raises(store):
    with pytest.raises(MlflowException, match="name.*version") as exc:
        store.create_mcp_server_version({"name": "test"})
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_mcp_server_version_with_tools(store):
    tools = [MCPTool(name="web_search", description="Search the web")]
    sv = store.create_mcp_server_version(_server_json(), tools=tools)
    assert sv.tools is not None
    assert len(sv.tools) == 1
    assert sv.tools[0].name == "web_search"


def test_create_mcp_server_version_with_status(store):
    sv = store.create_mcp_server_version(_server_json(), status=MCPStatus.ACTIVE)
    assert sv.status == MCPStatus.ACTIVE


def test_create_mcp_server_version_with_source(store):
    sv = store.create_mcp_server_version(_server_json(), source="https://github.com/org/repo")
    assert sv.source == "https://github.com/org/repo"


def test_get_mcp_server_version(store):
    store.create_mcp_server_version(_server_json())
    sv = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert sv.version == "1.0.0"


def test_get_mcp_server_version_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version("nonexistent", "1.0")


def test_get_latest_mcp_server_version_pinned(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.ACTIVE)
    store.update_mcp_server("s", latest_version="1.0")
    latest = store.get_latest_mcp_server_version("s")
    assert latest.version == "1.0"


def test_get_latest_mcp_server_version_fallback(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.update_mcp_server_version("s", "1.0", status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"))
    latest = store.get_latest_mcp_server_version("s")
    assert latest.version == "1.0"


def test_get_latest_mcp_server_version_same_timestamp_tiebreaker(store, monkeypatch):
    """When two versions share the same created_at, version DESC breaks the tie.

    This must match the ordering in _latest_candidates_query to keep
    get_latest_mcp_server_version and resolved_status consistent.
    """
    monkeypatch.setattr(
        "mlflow.store.tracking.mcp_server_registry.sqlalchemy_mixin.get_current_time_millis",
        lambda: 1000,
    )
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.ACTIVE)
    latest = store.get_latest_mcp_server_version("s")
    assert latest.version == "2.0"

    server = store.get_mcp_server("s")
    assert server.status == MCPStatus.ACTIVE.value


def test_get_latest_mcp_server_version_skips_draft(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    with pytest.raises(MlflowException, match="No eligible"):
        store.get_latest_mcp_server_version("s")


def test_get_latest_mcp_server_version_server_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_latest_mcp_server_version("nonexistent")


def test_get_mcp_server_version_by_alias(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_alias("s", "stable", "1.0")
    sv = store.get_mcp_server_version_by_alias("s", "stable")
    assert sv.version == "1.0"


def test_get_mcp_server_version_by_latest_alias(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    sv = store.get_mcp_server_version_by_alias("s", "latest")
    assert sv.version == "1.0"


def test_get_mcp_server_version_by_alias_not_found(store):
    store.create_mcp_server("s")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version_by_alias("s", "nonexistent")


def test_search_mcp_server_versions(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.create_mcp_server_version(_server_json("s", "2.0"))
    result = store.search_mcp_server_versions("s")
    assert len(result) == 2


def test_search_mcp_server_versions_scoped(store):
    store.create_mcp_server_version(_server_json("s1", "1.0"))
    store.create_mcp_server_version(_server_json("s2", "1.0"))
    result = store.search_mcp_server_versions("s1")
    assert len(result) == 1
    assert result[0].name == "s1"


def test_update_mcp_server_version_status(store):
    store.create_mcp_server_version(_server_json())
    updated = store.update_mcp_server_version(
        "io.github.test/server", "1.0.0", status=MCPStatus.ACTIVE
    )
    assert updated.status == MCPStatus.ACTIVE


def test_update_mcp_server_version_to_draft_clears_latest_pin(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.ACTIVE)
    store.update_mcp_server("s", latest_version="2.0")
    store.update_mcp_server_version("s", "2.0", status=MCPStatus.DRAFT)
    server = store.get_mcp_server("s")
    assert server.latest_version is None
    assert server.status == MCPStatus.ACTIVE


def test_update_mcp_server_version_invalid_transition(store):
    store.create_mcp_server_version(_server_json(), status=MCPStatus.ACTIVE)
    with pytest.raises(MlflowException, match="Invalid status transition") as exc:
        store.update_mcp_server_version("io.github.test/server", "1.0.0", status=MCPStatus.DELETED)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_mcp_server_version_display_name(store):
    store.create_mcp_server_version(_server_json())
    updated = store.update_mcp_server_version("io.github.test/server", "1.0.0", display_name="v1")
    assert updated.display_name == "v1"


def test_update_mcp_server_version_tools(store):
    store.create_mcp_server_version(_server_json())
    tools = [MCPTool(name="calculator")]
    updated = store.update_mcp_server_version("io.github.test/server", "1.0.0", tools=tools)
    assert len(updated.tools) == 1
    assert updated.tools[0].name == "calculator"


def test_update_mcp_server_version_returns_complete_entity(store):
    store.create_mcp_server_version(_server_json(), status=MCPStatus.ACTIVE)
    store.set_mcp_server_version_tag("io.github.test/server", "1.0.0", "env", "prod")
    updated = store.update_mcp_server_version(
        "io.github.test/server", "1.0.0", display_name="Updated"
    )
    assert updated.display_name == "Updated"
    assert updated.status == MCPStatus.ACTIVE
    assert updated.server_json == _server_json()
    assert updated.tags == {"env": "prod"}


def test_delete_mcp_server_version_soft_delete(store):
    store.create_mcp_server_version(_server_json())
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version("io.github.test/server", "1.0.0")


def test_delete_mcp_server_version_clears_latest_pin(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.DEPRECATED)
    store.update_mcp_server("s", latest_version="2.0")
    store.delete_mcp_server_version("s", "2.0")
    server = store.get_mcp_server("s")
    assert server.latest_version is None
    assert server.status == MCPStatus.ACTIVE


def test_delete_mcp_server_version_active_raises(store):
    store.create_mcp_server_version(_server_json(), status=MCPStatus.ACTIVE)
    with pytest.raises(MlflowException, match="Invalid status transition"):
        store.delete_mcp_server_version("io.github.test/server", "1.0.0")


def test_delete_mcp_server_version_cleans_up_aliases_and_bindings(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.set_mcp_server_alias("s", "stable", "1.0")
    store.create_mcp_access_binding("s", "https://direct.example.com", server_version="1.0")
    store.create_mcp_access_binding("s", "https://alias.example.com", server_alias="stable")
    store.delete_mcp_server_version("s", "1.0")
    server = store.get_mcp_server("s")
    assert server.aliases == {}
    assert len(store.search_mcp_access_bindings(server_name="s")) == 0


def test_delete_mcp_server_version_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server_version("nonexistent", "1.0")


# --- MCPAccessBinding CRUD ---


def test_create_mcp_access_binding_with_version(store):
    _setup_server(store, "s")
    binding = store.create_mcp_access_binding("s", "https://mcp.example.com", server_version="1.0")
    assert binding.server_name == "s"
    assert binding.endpoint_url == "https://mcp.example.com"
    assert binding.server_version == "1.0"
    assert binding.transport_type == MCPRemoteTransportType.STREAMABLE_HTTP


def test_create_mcp_access_binding_with_alias(store):
    _setup_server(store, "s", aliases={"stable": "1.0"})
    binding = store.create_mcp_access_binding("s", "https://mcp.example.com", server_alias="stable")
    assert binding.server_alias == "stable"


def test_create_mcp_access_binding_nonexistent_version_raises(store):
    _setup_server(store, "s")
    with pytest.raises(MlflowException, match="not found"):
        store.create_mcp_access_binding("s", "https://mcp.example.com", server_version="9.9")


def test_create_mcp_access_binding_nonexistent_alias_raises(store):
    _setup_server(store, "s")
    with pytest.raises(MlflowException, match="not found"):
        store.create_mcp_access_binding("s", "https://mcp.example.com", server_alias="fake")


def test_create_mcp_access_binding_deleted_version_raises(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.delete_mcp_server_version("s", "1.0")
    with pytest.raises(MlflowException, match="deleted MCP server version"):
        store.create_mcp_access_binding("s", "https://mcp.example.com", server_version="1.0")


def test_create_mcp_access_binding_neither_raises(store):
    _setup_server(store, "s")
    with pytest.raises(MlflowException, match="Exactly one"):
        store.create_mcp_access_binding("s", "https://mcp.example.com")


def test_create_mcp_access_binding_both_raises(store):
    _setup_server(store, "s", aliases={"stable": "1.0"})
    with pytest.raises(MlflowException, match="Exactly one"):
        store.create_mcp_access_binding(
            "s",
            "https://mcp.example.com",
            server_version="1.0",
            server_alias="stable",
        )


def test_create_mcp_access_binding_server_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store.create_mcp_access_binding(
            "nonexistent", "https://mcp.example.com", server_version="1.0"
        )


def test_get_mcp_access_binding_not_found_raises(store):
    _setup_server(store, "s")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_access_binding("s", 99999)


def test_search_mcp_access_bindings_all(store):
    _setup_server(store, "s1")
    _setup_server(store, "s2")
    store.create_mcp_access_binding("s1", "https://a.com", server_version="1.0")
    store.create_mcp_access_binding("s2", "https://b.com", server_version="1.0")
    result = store.search_mcp_access_bindings()
    assert len(result) == 2


def test_search_mcp_access_bindings_by_server(store):
    _setup_server(store, "s1")
    _setup_server(store, "s2")
    store.create_mcp_access_binding("s1", "https://a.com", server_version="1.0")
    store.create_mcp_access_binding("s2", "https://b.com", server_version="1.0")
    result = store.search_mcp_access_bindings(server_name="s1")
    assert len(result) == 1
    assert result[0].server_name == "s1"


def test_search_mcp_access_bindings_by_version(store):
    _setup_server(store, "s", versions=("1.0", "2.0"))
    store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    store.create_mcp_access_binding("s", "https://b.com", server_version="2.0")
    result = store.search_mcp_access_bindings(server_version="1.0")
    assert len(result) == 1


def test_delete_mcp_access_binding(store):
    _setup_server(store, "s")
    binding = store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    store.delete_mcp_access_binding("s", binding.binding_id)
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_access_binding("s", binding.binding_id)


# --- Tags ---


def test_set_and_get_mcp_server_tag(store):
    store.create_mcp_server("s")
    store.set_mcp_server_tag("s", "env", "prod")
    server = store.get_mcp_server("s")
    assert server.tags == {"env": "prod"}


def test_upsert_mcp_server_tag(store):
    store.create_mcp_server("s")
    store.set_mcp_server_tag("s", "env", "dev")
    store.set_mcp_server_tag("s", "env", "prod")
    server = store.get_mcp_server("s")
    assert server.tags == {"env": "prod"}


def test_delete_mcp_server_tag(store):
    store.create_mcp_server("s")
    store.set_mcp_server_tag("s", "env", "prod")
    store.delete_mcp_server_tag("s", "env")
    server = store.get_mcp_server("s")
    assert server.tags == {}


def test_delete_mcp_server_tag_not_found_raises(store):
    store.create_mcp_server("s")
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server_tag("s", "nonexistent")


def test_set_and_get_mcp_server_version_tag(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.set_mcp_server_version_tag("s", "1.0", "env", "prod")
    sv = store.get_mcp_server_version("s", "1.0")
    assert sv.tags == {"env": "prod"}


def test_upsert_mcp_server_version_tag(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.set_mcp_server_version_tag("s", "1.0", "env", "dev")
    store.set_mcp_server_version_tag("s", "1.0", "env", "prod")
    sv = store.get_mcp_server_version("s", "1.0")
    assert sv.tags == {"env": "prod"}


def test_delete_mcp_server_version_tag(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.set_mcp_server_version_tag("s", "1.0", "env", "prod")
    store.delete_mcp_server_version_tag("s", "1.0", "env")
    sv = store.get_mcp_server_version("s", "1.0")
    assert sv.tags == {}


# --- Aliases ---


def test_set_and_resolve_mcp_server_alias(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.set_mcp_server_alias("s", "stable", "1.0")
    sv = store.get_mcp_server_version_by_alias("s", "stable")
    assert sv.version == "1.0"


def test_upsert_mcp_server_alias_retargets(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.create_mcp_server_version(_server_json("s", "2.0"))
    store.set_mcp_server_alias("s", "stable", "1.0")
    store.set_mcp_server_alias("s", "stable", "2.0")
    sv = store.get_mcp_server_version_by_alias("s", "stable")
    assert sv.version == "2.0"


def test_set_mcp_server_alias_latest_reserved(store):
    store.create_mcp_server("s")
    with pytest.raises(MlflowException, match="reserved") as exc:
        store.set_mcp_server_alias("s", "latest", "1.0")
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_set_mcp_server_alias_to_deleted_version_raises(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.delete_mcp_server_version("s", "1.0")
    with pytest.raises(MlflowException, match="Cannot set alias"):
        store.set_mcp_server_alias("s", "stable", "1.0")


def test_delete_mcp_server_alias(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.set_mcp_server_alias("s", "stable", "1.0")
    store.delete_mcp_server_alias("s", "stable")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version_by_alias("s", "stable")


def test_delete_mcp_server_alias_cleans_up_alias_bindings(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_alias("s", "stable", "1.0")
    binding = store.create_mcp_access_binding(
        "s",
        "https://alias.example.com",
        server_alias="stable",
    )
    store.delete_mcp_server_alias("s", "stable")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_access_binding("s", binding.binding_id)
    assert len(store.search_mcp_access_bindings(server_name="s")) == 0
    server = store.get_mcp_server("s")
    assert server.aliases == {}
    assert server.access_bindings == []


def test_delete_mcp_server_alias_not_found_raises(store):
    store.create_mcp_server("s")
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server_alias("s", "nonexistent")


def test_set_mcp_server_alias_nonexistent_version_raises(store):
    store.create_mcp_server("s")
    with pytest.raises(MlflowException, match="not found"):
        store.set_mcp_server_alias("s", "stable", "nonexistent")


def test_aliases_appear_on_server(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.set_mcp_server_alias("s", "stable", "1.0")
    server = store.get_mcp_server("s")
    assert "stable" in server.aliases
    assert server.aliases["stable"] == "1.0"


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
    store.create_mcp_server_version(_server_json(), status=from_status)
    updated = store.update_mcp_server_version("io.github.test/server", "1.0.0", status=to_status)
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
        store.update_mcp_server_version("io.github.test/server", "1.0.0", status=to_status)


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


def test_search_mcp_servers_filter_by_tag(store):
    store.create_mcp_server("s1")
    store.create_mcp_server("s2")
    store.create_mcp_server("s3")
    store.set_mcp_server_tag("s1", "env", "prod")
    store.set_mcp_server_tag("s2", "env", "staging")
    store.set_mcp_server_tag("s3", "env", "prod")
    result = store.search_mcp_servers(filter_string="tags.env = 'prod'")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"s1", "s3"}


def test_search_mcp_servers_filter_by_multiple_tags(store):
    store.create_mcp_server("s1")
    store.create_mcp_server("s2")
    store.set_mcp_server_tag("s1", "env", "prod")
    store.set_mcp_server_tag("s1", "team", "ai-hub")
    store.set_mcp_server_tag("s2", "env", "prod")
    store.set_mcp_server_tag("s2", "team", "other")
    result = store.search_mcp_servers(filter_string="tags.env = 'prod' AND tags.team = 'ai-hub'")
    assert len(result) == 1
    assert result[0].name == "s1"


def test_search_mcp_servers_filter_attribute_and_tag(store):
    store.create_mcp_server("io.github.test/alpha")
    store.create_mcp_server("io.github.test/beta")
    store.set_mcp_server_tag("io.github.test/alpha", "env", "prod")
    store.set_mcp_server_tag("io.github.test/beta", "env", "prod")
    result = store.search_mcp_servers(filter_string="name LIKE '%alpha%' AND tags.env = 'prod'")
    assert len(result) == 1
    assert result[0].name == "io.github.test/alpha"


def test_search_mcp_servers_filter_has_access_bindings_true(store):
    _setup_server(store, "s1")
    store.create_mcp_server("s2")
    _setup_server(store, "s3")
    store.create_mcp_access_binding("s1", "https://a.com", server_version="1.0")
    store.create_mcp_access_binding("s3", "https://b.com", server_version="1.0")
    result = store.search_mcp_servers(filter_string="has_access_bindings = 'true'")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"s1", "s3"}


def test_search_mcp_servers_filter_by_status_uses_resolved_latest(store):
    store.create_mcp_server_version(_server_json("s1", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s1", "2.0"), status=MCPStatus.DEPRECATED)
    store.create_mcp_server_version(_server_json("s2", "1.0"), status=MCPStatus.ACTIVE)
    result = store.search_mcp_servers(filter_string="status = 'active'")
    names = {s.name for s in result}
    assert names == {"s2"}, "s1's latest is deprecated (2.0), should not match active"


def test_search_mcp_servers_filter_by_status_pinned(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.DEPRECATED)
    store.update_mcp_server("s", latest_version="1.0")
    result = store.search_mcp_servers(filter_string="status = 'active'")
    assert len(result) == 1, "pinned to 1.0 which is active"
    assert result[0].name == "s"


def test_search_mcp_servers_filter_by_status_no_versions(store):
    store.create_mcp_server("s1")
    store.create_mcp_server_version(_server_json("s2", "1.0"), status=MCPStatus.ACTIVE)
    result = store.search_mcp_servers(filter_string="status = 'active'")
    assert len(result) == 1
    assert result[0].name == "s2"


def test_search_mcp_servers_empty_filter_string(store):
    store.create_mcp_server("s1")
    store.create_mcp_server("s2")
    result = store.search_mcp_servers(filter_string="")
    assert len(result) == 2


def test_search_mcp_servers_filter_has_access_bindings_false(store):
    _setup_server(store, "s1")
    store.create_mcp_server("s2")
    _setup_server(store, "s3")
    store.create_mcp_access_binding("s1", "https://a.com", server_version="1.0")
    store.create_mcp_access_binding("s3", "https://b.com", server_version="1.0")
    result = store.search_mcp_servers(filter_string="has_access_bindings = 'false'")
    assert len(result) == 1
    assert result[0].name == "s2"


def test_search_mcp_access_bindings_filter_by_transport_type(store):
    _setup_server(store, "s")
    store.create_mcp_access_binding(
        "s",
        "https://a.com",
        server_version="1.0",
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
    )
    store.create_mcp_access_binding(
        "s",
        "https://b.com",
        server_version="1.0",
        transport_type=MCPRemoteTransportType.SSE,
    )
    result = store.search_mcp_access_bindings(filter_string="transport_type = 'streamable-http'")
    assert len(result) == 1
    assert result[0].endpoint_url == "https://a.com"


def test_search_mcp_servers_filter_by_status_in(store):
    store.create_mcp_server_version(_server_json("s1", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s2", "1.0"), status=MCPStatus.DRAFT)
    store.create_mcp_server_version(_server_json("s3", "1.0"), status=MCPStatus.DEPRECATED)
    result = store.search_mcp_servers(filter_string="status IN ('active', 'deprecated')")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"s1", "s3"}


def test_search_mcp_server_versions_filter_by_status_in(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.DRAFT)
    store.create_mcp_server_version(_server_json("s", "3.0"), status=MCPStatus.DEPRECATED)
    result = store.search_mcp_server_versions(
        "s", filter_string="status IN ('active', 'deprecated')"
    )
    assert len(result) == 2
    versions = {v.version for v in result}
    assert versions == {"1.0", "3.0"}


def test_search_mcp_servers_filter_by_status(store):
    store.create_mcp_server_version(_server_json("s1", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s2", "1.0"), status=MCPStatus.DRAFT)
    store.create_mcp_server_version(_server_json("s3", "1.0"), status=MCPStatus.ACTIVE)
    result = store.search_mcp_servers(filter_string="status = 'active'")
    assert len(result) == 2
    names = {s.name for s in result}
    assert names == {"s1", "s3"}


def test_search_mcp_server_versions_filter_by_status(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.DRAFT)
    store.create_mcp_server_version(_server_json("s", "3.0"), status=MCPStatus.ACTIVE)
    result = store.search_mcp_server_versions("s", filter_string="status = 'active'")
    assert len(result) == 2
    versions = {v.version for v in result}
    assert versions == {"1.0", "3.0"}


def test_search_mcp_servers_filter_invalid_attribute(store):
    with pytest.raises(MlflowException, match="Invalid attribute key"):
        store.search_mcp_servers(filter_string="bogus = 'x'")


# --- get_mcp_access_binding happy path ---


def test_get_mcp_access_binding(store):
    _setup_server(store, "s")
    created = store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    fetched = store.get_mcp_access_binding("s", created.binding_id)
    assert fetched.binding_id == created.binding_id
    assert fetched.server_name == "s"
    assert fetched.endpoint_url == "https://a.com"
    assert fetched.server_version == "1.0"
    assert fetched.transport_type == MCPRemoteTransportType.STREAMABLE_HTTP


# --- update_mcp_access_binding ---


def test_update_mcp_access_binding_version_clears_alias(store):
    _setup_server(store, "s", versions=("1.0", "2.0"), aliases={"stable": "1.0"})
    binding = store.create_mcp_access_binding("s", "https://a.com", server_alias="stable")
    assert binding.server_alias == "stable"
    updated = store.update_mcp_access_binding("s", binding.binding_id, server_version="2.0")
    assert updated.server_version == "2.0"
    assert updated.server_alias is None


def test_update_mcp_access_binding_alias_clears_version(store):
    _setup_server(store, "s", aliases={"prod": "1.0"})
    binding = store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    assert binding.server_version == "1.0"
    updated = store.update_mcp_access_binding("s", binding.binding_id, server_alias="prod")
    assert updated.server_alias == "prod"
    assert updated.server_version is None


def test_update_mcp_access_binding_endpoint_and_transport(store):
    _setup_server(store, "s")
    binding = store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    updated = store.update_mcp_access_binding(
        "s",
        binding.binding_id,
        endpoint_url="https://b.com",
        transport_type=MCPRemoteTransportType.SSE,
    )
    assert updated.endpoint_url == "https://b.com"
    assert updated.transport_type == MCPRemoteTransportType.SSE


def test_update_mcp_access_binding_both_version_and_alias_raises(store):
    _setup_server(store, "s", aliases={"stable": "1.0"})
    binding = store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    with pytest.raises(MlflowException, match="Cannot set both"):
        store.update_mcp_access_binding(
            "s", binding.binding_id, server_version="1.0", server_alias="stable"
        )


def test_update_mcp_access_binding_nonexistent_version_raises(store):
    _setup_server(store, "s")
    binding = store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_access_binding("s", binding.binding_id, server_version="9.9")


def test_update_mcp_access_binding_deleted_version_raises(store):
    _setup_server(store, "s", versions=("1.0", "2.0"))
    binding = store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    store.delete_mcp_server_version("s", "2.0")
    with pytest.raises(MlflowException, match="deleted MCP server version"):
        store.update_mcp_access_binding("s", binding.binding_id, server_version="2.0")


def test_update_mcp_access_binding_nonexistent_alias_raises(store):
    _setup_server(store, "s")
    binding = store.create_mcp_access_binding("s", "https://a.com", server_version="1.0")
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_access_binding("s", binding.binding_id, server_alias="fake")


def test_update_mcp_access_binding_wrong_server_raises(store):
    _setup_server(store, "s1")
    _setup_server(store, "s2")
    binding = store.create_mcp_access_binding("s1", "https://a.com", server_version="1.0")
    with pytest.raises(MlflowException, match="does not belong"):
        store.update_mcp_access_binding("s2", binding.binding_id, endpoint_url="https://b.com")


# --- search_mcp_server_versions pagination ---


def test_search_mcp_server_versions_pagination(store):
    for i in range(5):
        store.create_mcp_server_version(_server_json("s", f"{i}.0"))
    page1 = store.search_mcp_server_versions("s", max_results=2)
    assert len(page1) == 2
    assert page1.token is not None
    page2 = store.search_mcp_server_versions("s", max_results=2, page_token=page1.token)
    assert len(page2) == 2
    assert page2.token is not None
    page3 = store.search_mcp_server_versions("s", max_results=2, page_token=page2.token)
    assert len(page3) == 1
    assert page3.token is None


# --- order_by ---


def test_search_mcp_servers_order_by_name_desc(store):
    store.create_mcp_server("alpha")
    store.create_mcp_server("beta")
    store.create_mcp_server("gamma")
    result = store.search_mcp_servers(order_by=["name DESC"])
    names = [s.name for s in result]
    assert names == ["gamma", "beta", "alpha"]


def test_search_mcp_servers_order_by_name_asc(store):
    store.create_mcp_server("gamma")
    store.create_mcp_server("alpha")
    store.create_mcp_server("beta")
    result = store.search_mcp_servers(order_by=["name ASC"])
    names = [s.name for s in result]
    assert names == ["alpha", "beta", "gamma"]


def test_search_mcp_servers_order_by_default_is_name_asc(store):
    store.create_mcp_server("gamma")
    store.create_mcp_server("alpha")
    store.create_mcp_server("beta")
    result = store.search_mcp_servers()
    names = [s.name for s in result]
    assert names == ["alpha", "beta", "gamma"]


def test_search_mcp_servers_order_by_invalid_key(store):
    with pytest.raises(MlflowException, match="Invalid order_by key"):
        store.search_mcp_servers(order_by=["bogus ASC"])


def test_search_mcp_servers_order_by_duplicate_key(store):
    with pytest.raises(MlflowException, match="Duplicate order_by"):
        store.search_mcp_servers(order_by=["name ASC", "name DESC"])


# --- Additional coverage ---


def test_delete_mcp_access_binding_wrong_server_raises(store):
    _setup_server(store, "s1")
    _setup_server(store, "s2")
    binding = store.create_mcp_access_binding("s1", "https://a.com", server_version="1.0")
    with pytest.raises(MlflowException, match="does not belong"):
        store.delete_mcp_access_binding("s2", binding.binding_id)


def test_delete_mcp_server_version_tag_not_found_raises(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    with pytest.raises(MlflowException, match="not found"):
        store.delete_mcp_server_version_tag("s", "1.0", "nonexistent")


def test_update_mcp_server_version_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_mcp_server_version("nonexistent", "1.0", display_name="x")


def test_search_mcp_access_bindings_pagination(store):
    _setup_server(store, "s")
    for i in range(5):
        store.create_mcp_access_binding("s", f"https://{i}.com", server_version="1.0")
    page1 = store.search_mcp_access_bindings(max_results=2)
    assert len(page1) == 2
    assert page1.token is not None
    page2 = store.search_mcp_access_bindings(max_results=2, page_token=page1.token)
    assert len(page2) == 2
    assert page2.token is not None
    page3 = store.search_mcp_access_bindings(max_results=2, page_token=page2.token)
    assert len(page3) == 1
    assert page3.token is None


def test_search_mcp_server_versions_order_by(store):
    store.create_mcp_server_version(_server_json("s", "alpha"))
    store.create_mcp_server_version(_server_json("s", "beta"))
    store.create_mcp_server_version(_server_json("s", "gamma"))
    result = store.search_mcp_server_versions("s", order_by=["`version` DESC"])
    versions = [v.version for v in result]
    assert versions == ["gamma", "beta", "alpha"]


def test_deleted_versions_excluded_from_get(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.delete_mcp_server_version("s", "1.0")
    with pytest.raises(MlflowException, match="not found"):
        store.get_mcp_server_version("s", "1.0")


def test_deleted_versions_excluded_from_search(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.create_mcp_server_version(_server_json("s", "2.0"), status=MCPStatus.ACTIVE)
    store.delete_mcp_server_version("s", "1.0")
    result = store.search_mcp_server_versions("s")
    assert len(result) == 1
    assert result[0].version == "2.0"


def test_binding_resolved_version_direct(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://example.com", server_version="1.0"
    )
    binding = store.search_mcp_access_bindings(server_name="s")[0]
    assert binding.resolved_version is not None
    assert binding.resolved_version.version == "1.0"
    assert binding.resolved_version.name == "s"


def test_binding_resolved_version_via_alias(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_alias("s", "prod", "1.0")
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://example.com", server_alias="prod"
    )
    binding = store.search_mcp_access_bindings(server_name="s")[0]
    assert binding.resolved_version is not None
    assert binding.resolved_version.version == "1.0"


def test_binding_resolved_version_on_get(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    binding = store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://example.com", server_version="1.0"
    )
    fetched = store.get_mcp_access_binding("s", binding.binding_id)
    assert fetched.resolved_version is not None
    assert fetched.resolved_version.version == "1.0"


def test_binding_resolved_version_on_get_via_alias(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_alias("s", "prod", "1.0")
    binding = store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://example.com", server_alias="prod"
    )
    fetched = store.get_mcp_access_binding("s", binding.binding_id)
    assert fetched.resolved_version is not None
    assert fetched.resolved_version.version == "1.0"


def test_search_bindings_filter_by_status(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.create_mcp_server_version(_server_json("s", "2.0"))
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://a.example.com", server_version="1.0"
    )
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://b.example.com", server_version="2.0"
    )
    result = store.search_mcp_access_bindings(filter_string="status = 'active'")
    assert len(result) == 1
    assert result[0].endpoint_url == "https://a.example.com"


def test_search_bindings_scoped_to_server_version_resolves_version(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_alias("s", "prod", "1.0")
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://direct.example.com", server_version="1.0"
    )
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://alias.example.com", server_alias="prod"
    )
    result = store.search_mcp_access_bindings(server_name="s", server_version="1.0")
    assert len(result) == 1
    assert result[0].endpoint_url == "https://direct.example.com"
    assert result[0].resolved_version is not None
    assert result[0].resolved_version.version == "1.0"


def test_search_bindings_scoped_to_server_alias_resolves_version(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    store.set_mcp_server_alias("s", "prod", "1.0")
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://direct.example.com", server_version="1.0"
    )
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://alias.example.com", server_alias="prod"
    )
    result = store.search_mcp_access_bindings(server_name="s", server_alias="prod")
    assert len(result) == 1
    assert result[0].endpoint_url == "https://alias.example.com"
    assert result[0].resolved_version is not None
    assert result[0].resolved_version.version == "1.0"


def test_search_servers_numeric_timestamp_filter(store):
    store.create_mcp_server("s")
    result = store.search_mcp_servers(filter_string="created_at > 0")
    assert len(result) == 1


def test_create_binding_returns_resolved_version(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    binding = store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://example.com", server_version="1.0"
    )
    assert binding.resolved_version is not None
    assert binding.resolved_version.version == "1.0"


def test_update_binding_returns_resolved_version(store):
    store.create_mcp_server_version(_server_json("s", "1.0"), status=MCPStatus.ACTIVE)
    binding = store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://example.com", server_version="1.0"
    )
    updated = store.update_mcp_access_binding(
        server_name="s", binding_id=binding.binding_id, endpoint_url="https://new.example.com"
    )
    assert updated.resolved_version is not None
    assert updated.resolved_version.version == "1.0"


def test_binding_to_deleted_version_hidden(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://example.com", server_version="1.0"
    )
    store.delete_mcp_server_version("s", "1.0")
    result = store.search_mcp_access_bindings(server_name="s")
    assert len(result) == 0


def test_has_access_bindings_excludes_stale_bindings(store):
    store.create_mcp_server_version(_server_json("s", "1.0"))
    store.create_mcp_access_binding(
        server_name="s", endpoint_url="https://example.com", server_version="1.0"
    )
    result = store.search_mcp_servers(filter_string="has_access_bindings = 'true'")
    assert len(result) == 1
    store.delete_mcp_server_version("s", "1.0")
    result = store.search_mcp_servers(filter_string="has_access_bindings = 'true'")
    assert len(result) == 0


def test_has_access_bindings_duplicate_rejected(store):
    store.create_mcp_server("s")
    with pytest.raises(MlflowException, match="Invalid"):
        store.search_mcp_servers(
            filter_string="has_access_bindings = true AND has_access_bindings = false"
        )
