import pytest
from sqlalchemy.exc import IntegrityError

from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.store.workspace.dbmodels.models import SqlWorkspace
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


@pytest.fixture
def workspace_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")

    db_path = tmp_path / "workspace.sqlite"
    uri = f"sqlite:///{db_path}"
    store = SqlAlchemyStore(uri)

    with store.ManagedSessionMaker() as session:
        try:
            session.add(
                SqlWorkspace(
                    name=DEFAULT_WORKSPACE_NAME,
                    description="Default workspace",
                )
            )
            session.commit()
        except IntegrityError:
            session.rollback()

    try:
        yield store
    finally:
        store._engine.dispose()


def _workspace_rows(store):
    with store.ManagedSessionMaker() as session:
        return {
            (row.name, row.description)
            for row in session.query(SqlWorkspace).order_by(SqlWorkspace.name).all()
        }


def test_list_workspaces_returns_all(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="Team A"))
    workspace_store.create_workspace(Workspace(name="team-b", description=None))

    workspaces = workspace_store.list_workspaces()
    rows = {(ws.name, ws.description) for ws in workspaces}
    default_description = next(desc for name, desc in rows if name == DEFAULT_WORKSPACE_NAME)
    assert rows == {
        (DEFAULT_WORKSPACE_NAME, default_description),
        ("team-a", "Team A"),
        ("team-b", None),
    }


def test_get_workspace_success(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="Team A"))

    workspace = workspace_store.get_workspace("team-a")
    assert workspace.name == "team-a"
    assert workspace.description == "Team A"


def test_get_workspace_not_found(workspace_store):
    with pytest.raises(MlflowException, match="Workspace 'unknown' not found") as exc:
        workspace_store.get_workspace("unknown")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_create_workspace_persists_record(workspace_store):
    created = workspace_store.create_workspace(
        Workspace(name="team-a", description="Team A", default_artifact_root="s3://root/team-a"),
    )
    assert created.name == "team-a"
    assert created.description == "Team A"
    assert created.default_artifact_root == "s3://root/team-a"
    assert ("team-a", "Team A") in _workspace_rows(workspace_store)


def test_create_workspace_duplicate_raises(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with pytest.raises(
        MlflowException,
        match="Workspace 'team-a' already exists\\.",
    ) as exc:
        workspace_store.create_workspace(Workspace(name="team-a", description=None))
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_workspace_invalid_name_raises(workspace_store):
    with pytest.raises(
        MlflowException,
        match="Workspace name 'Team-A' must match the pattern",
    ) as exc:
        workspace_store.create_workspace(Workspace(name="Team-A", description=None))
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_workspace_changes_description(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", description="new description"),
    )
    assert updated.description == "new description"
    assert ("team-a", "new description") in _workspace_rows(workspace_store)


def test_update_workspace_sets_default_artifact_root(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", default_artifact_root="s3://bucket/team-a"),
    )
    assert updated.default_artifact_root == "s3://bucket/team-a"
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.default_artifact_root == "s3://bucket/team-a"


def test_update_workspace_can_clear_default_artifact_root(workspace_store):
    workspace_store.create_workspace(
        Workspace(name="team-a", description="old", default_artifact_root="s3://bucket/team-a")
    )

    # Empty string signals "clear this field"
    cleared = workspace_store.update_workspace(
        Workspace(name="team-a", default_artifact_root=""),
    )
    assert cleared.default_artifact_root is None
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.default_artifact_root is None


def test_delete_workspace_removes_empty_workspace(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    workspace_store.delete_workspace("team-a")
    rows = _workspace_rows(workspace_store)
    assert ("team-a", None) not in rows
    default_ws = workspace_store.get_default_workspace()
    assert (DEFAULT_WORKSPACE_NAME, default_ws.description) in rows


def test_delete_default_workspace_rejected(workspace_store):
    with pytest.raises(
        MlflowException,
        match=f"Cannot delete the reserved '{DEFAULT_WORKSPACE_NAME}' workspace",
    ) as exc:
        workspace_store.delete_workspace(DEFAULT_WORKSPACE_NAME)
    assert exc.value.error_code == "INVALID_STATE"


def test_update_workspace_not_found(workspace_store):
    with pytest.raises(
        MlflowException,
        match="Workspace 'unknown' not found",
    ) as exc:
        workspace_store.update_workspace(Workspace(name="unknown", description="new description"))
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_delete_workspace_not_found(workspace_store):
    with pytest.raises(
        MlflowException,
        match="Workspace 'unknown' not found",
    ) as exc:
        workspace_store.delete_workspace("unknown")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_resolve_artifact_root_returns_default(workspace_store):
    default_root = "/default/path"
    assert workspace_store.resolve_artifact_root(default_root, DEFAULT_WORKSPACE_NAME) == (
        default_root,
        True,
    )
    workspace_store.create_workspace(Workspace(name="team-a", description=None))
    assert workspace_store.resolve_artifact_root(default_root, workspace_name="team-a") == (
        default_root,
        True,
    )


def test_resolve_artifact_root_prefers_workspace_override(workspace_store):
    workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description=None,
            default_artifact_root="s3://team-a-artifacts",
        )
    )

    resolved_root, should_append = workspace_store.resolve_artifact_root(
        "/default/path", workspace_name="team-a"
    )
    assert resolved_root == "s3://team-a-artifacts"
    assert not should_append


def test_resolve_artifact_root_cache_updates_on_override_change(workspace_store):
    default_root = "/default/path"
    workspace_store.create_workspace(Workspace(name="team-cache", description=None))

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        default_root,
        True,
    )

    workspace_store.update_workspace(
        Workspace(name="team-cache", default_artifact_root="s3://cache/team")
    )

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        "s3://cache/team",
        False,
    )


def test_resolve_artifact_root_cache_handles_delete_and_recreate(workspace_store):
    default_root = "/default/path"
    workspace_store.create_workspace(
        Workspace(name="team-cache", description=None, default_artifact_root="s3://cache/a")
    )

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        "s3://cache/a",
        False,
    )

    workspace_store.delete_workspace("team-cache")
    workspace_store.create_workspace(
        Workspace(name="team-cache", description=None, default_artifact_root="s3://cache/b")
    )

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        "s3://cache/b",
        False,
    )


def test_resolve_artifact_root_cache_clears_when_override_removed(workspace_store):
    default_root = "/default/path"
    workspace_store.create_workspace(
        Workspace(name="team-cache", description=None, default_artifact_root="s3://cache/a")
    )

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        "s3://cache/a",
        False,
    )

    workspace_store.update_workspace(Workspace(name="team-cache", default_artifact_root=""))

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        default_root,
        True,
    )


def test_get_default_workspace_returns_default(workspace_store):
    default_ws = workspace_store.get_default_workspace()
    assert default_ws.name == DEFAULT_WORKSPACE_NAME
    assert default_ws.description is not None
