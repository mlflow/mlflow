from pathlib import Path

import pytest

from mlflow.entities.mcp_server import MCPStatus
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    store = SqlAlchemyStore(MLFLOW_TRACKING_URI.get(), artifact_uri.as_uri())
    try:
        yield store
    finally:
        store._dispose_engine()


def _server_json(name: str, version: str) -> dict[str, str]:
    return {"name": name, "version": version, "title": f"Test {name}"}


def test_db_backend_mcp_latest_prefers_semver_prerelease_ordering(store):
    for version in ("1.0.0-alpha.2", "1.0.0-alpha.10", "1.0.0-beta.1"):
        store.create_mcp_server_version(
            _server_json("io.github.test/backend-prerelease", version),
            status=MCPStatus.ACTIVE,
        )

    latest = store.get_latest_mcp_server_version("io.github.test/backend-prerelease")
    assert latest.version == "1.0.0-beta.1"

    server = store.get_mcp_server("io.github.test/backend-prerelease")
    assert server.latest_version == "1.0.0-beta.1"
    assert server.status == MCPStatus.ACTIVE


def test_db_backend_mcp_latest_uses_build_metadata_as_final_tiebreaker(store):
    for version in ("1.0.0+abc", "1.0.0+xyz"):
        store.create_mcp_server_version(
            _server_json("io.github.test/backend-build-meta", version),
            status=MCPStatus.ACTIVE,
        )

    latest = store.get_latest_mcp_server_version("io.github.test/backend-build-meta")
    assert latest.version == "1.0.0+xyz"

    server = store.get_mcp_server("io.github.test/backend-build-meta")
    assert server.latest_version == "1.0.0+xyz"


def test_db_backend_mcp_latest_respects_ascii_prerelease_order(store):
    for version in ("1.0.0-Z", "1.0.0-a"):
        store.create_mcp_server_version(
            _server_json("io.github.test/backend-ascii", version),
            status=MCPStatus.ACTIVE,
        )

    latest = store.get_latest_mcp_server_version("io.github.test/backend-ascii")
    assert latest.version == "1.0.0-a"

    server = store.get_mcp_server("io.github.test/backend-ascii")
    assert server.latest_version == "1.0.0-a"


def test_db_backend_mcp_latest_falls_back_to_highest_non_active_semver(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/backend-fallback", "1.2.0"),
        status=MCPStatus.DEPRECATED,
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/backend-fallback", "1.3.0"),
        status=MCPStatus.DRAFT,
    )

    latest = store.get_latest_mcp_server_version("io.github.test/backend-fallback")
    assert latest.version == "1.3.0"
    assert latest.status == MCPStatus.DRAFT

    server = store.get_mcp_server("io.github.test/backend-fallback")
    assert server.latest_version == "1.3.0"
    assert server.status == MCPStatus.DRAFT


def test_db_backend_mcp_latest_prefers_active_pool_over_higher_non_active_versions(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/backend-active-priority", "1.0.0"),
        status=MCPStatus.ACTIVE,
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/backend-active-priority", "99.0.0"),
        status=MCPStatus.DEPRECATED,
    )

    latest = store.get_latest_mcp_server_version("io.github.test/backend-active-priority")
    assert latest.version == "1.0.0"
    assert latest.status == MCPStatus.ACTIVE

    server = store.get_mcp_server("io.github.test/backend-active-priority")
    assert server.latest_version == "1.0.0"
    assert server.status == MCPStatus.ACTIVE


def test_db_backend_mcp_status_filter_uses_resolved_latest_status(store):
    store.create_mcp_server_version(
        _server_json("io.github.test/backend-status-active", "1.0.0"),
        status=MCPStatus.ACTIVE,
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/backend-status-active", "2.0.0"),
        status=MCPStatus.DEPRECATED,
    )
    store.create_mcp_server_version(
        _server_json("io.github.test/backend-status-draft", "1.0.0"),
        status=MCPStatus.DRAFT,
    )

    active_results = store.search_mcp_servers(filter_string="status = 'active'")
    active_names = {server.name for server in active_results}
    assert "io.github.test/backend-status-active" in active_names
    assert "io.github.test/backend-status-draft" not in active_names

    draft_results = store.search_mcp_servers(filter_string="status = 'draft'")
    draft_names = {server.name for server in draft_results}
    assert "io.github.test/backend-status-draft" in draft_names
