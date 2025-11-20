import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.store.db import utils as db_utils
from mlflow.store.workspace.dbmodels.models import SqlWorkspace
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


@pytest.fixture
def workspace_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")

    db_path = tmp_path / "workspace.sqlite"
    uri = f"sqlite:///{db_path}"
    engine = sa.create_engine(uri)

    # Initialize full MLflow schema for tracking + workspace tables.
    db_utils._initialize_tables(engine)

    # Seed the default workspace entry
    with engine.begin() as conn:
        try:
            conn.execute(
                SqlWorkspace.__table__.insert().values(
                    name=DEFAULT_WORKSPACE_NAME,
                    description="Default workspace",
                )
            )
        except IntegrityError:
            pass

    store = SqlAlchemyStore(uri)

    try:
        yield store
    finally:
        store._engine.dispose()
        engine.dispose()


def _workspace_rows(store):
    with store.ManagedSessionMaker() as session:
        return {
            (row.name, row.description)
            for row in session.query(SqlWorkspace).order_by(SqlWorkspace.name).all()
        }


def test_list_workspaces_returns_all(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="Team A"), request=None)
    workspace_store.create_workspace(Workspace(name="team-b", description=None), request=None)

    workspaces = workspace_store.list_workspaces(request=None)
    rows = {(ws.name, ws.description) for ws in workspaces}
    default_description = next(desc for name, desc in rows if name == DEFAULT_WORKSPACE_NAME)
    assert rows == {
        (DEFAULT_WORKSPACE_NAME, default_description),
        ("team-a", "Team A"),
        ("team-b", None),
    }


def test_get_workspace_success(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="Team A"), request=None)

    workspace = workspace_store.get_workspace("team-a", request=None)
    assert workspace.name == "team-a"
    assert workspace.description == "Team A"


def test_get_workspace_not_found(workspace_store):
    with pytest.raises(MlflowException, match="Workspace 'unknown' not found") as exc:
        workspace_store.get_workspace("unknown", request=None)
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_create_workspace_persists_record(workspace_store):
    created = workspace_store.create_workspace(
        Workspace(name="team-a", description="Team A"),
        request=None,
    )
    assert created.name == "team-a"
    assert created.description == "Team A"
    assert ("team-a", "Team A") in _workspace_rows(workspace_store)


def test_create_workspace_duplicate_raises(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None), request=None)

    with pytest.raises(
        MlflowException,
        match="Workspace 'team-a' already exists\\.",
    ) as exc:
        workspace_store.create_workspace(Workspace(name="team-a", description=None), request=None)
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_update_workspace_changes_description(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"), request=None)

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", description="new description"),
        request=None,
    )
    assert updated.description == "new description"
    assert ("team-a", "new description") in _workspace_rows(workspace_store)


def test_delete_workspace_removes_empty_workspace(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None), request=None)

    workspace_store.delete_workspace("team-a", request=None)
    rows = _workspace_rows(workspace_store)
    assert ("team-a", None) not in rows
    default_ws = workspace_store.get_default_workspace(request=None)
    assert (DEFAULT_WORKSPACE_NAME, default_ws.description) in rows


def test_delete_default_workspace_rejected(workspace_store):
    with pytest.raises(
        MlflowException,
        match=f"Cannot delete the reserved '{DEFAULT_WORKSPACE_NAME}' workspace",
    ) as exc:
        workspace_store.delete_workspace(DEFAULT_WORKSPACE_NAME, request=None)
    assert exc.value.error_code == "INVALID_STATE"


def test_update_workspace_not_found(workspace_store):
    with pytest.raises(
        MlflowException,
        match="Workspace 'unknown' not found",
    ) as exc:
        workspace_store.update_workspace(
            Workspace(name="unknown", description="new description"), request=None
        )
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_delete_workspace_not_found(workspace_store):
    with pytest.raises(
        MlflowException,
        match="Workspace 'unknown' not found",
    ) as exc:
        workspace_store.delete_workspace("unknown", request=None)
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_resolve_artifact_root_returns_default(workspace_store):
    default_root = "/default/path"
    assert workspace_store.resolve_artifact_root(default_root) == (default_root, True)
    assert workspace_store.resolve_artifact_root(default_root, workspace_name="team-a") == (
        default_root,
        True,
    )


def test_get_default_workspace_returns_default(workspace_store):
    default_ws = workspace_store.get_default_workspace(request=None)
    assert default_ws.name == DEFAULT_WORKSPACE_NAME
    assert default_ws.description is not None
