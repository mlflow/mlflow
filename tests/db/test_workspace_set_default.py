# Cross-dialect (Docker matrix) tests for `delete_workspace(SET_DEFAULT)`, which reassigns
# a workspace's resources to the default workspace and then deletes it.
#
# These exist because the conflict preflight and the UPDATE behind it depend on how each
# engine treats NULL, and that differs. Unlike the SQLite-only twin in
# tests/store/workspace/test_sqlalchemy_store.py, the fixtures here build their stores from
# MLFLOW_TRACKING_URI, so they really do run against PostgreSQL, MySQL and SQL Server.
#
# NOTE: all tests here share one database with the rest of tests/db, so each test uses ids
# unique to itself and removes its own rows afterwards -- an endpoint left behind in the
# default workspace would be visible to every later test.

import uuid

import pytest
import sqlalchemy as sa

from mlflow.entities.workspace import Workspace, WorkspaceDeletionMode
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES, MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlGatewayEndpoint
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore as WorkspaceStore
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

pytestmark = pytest.mark.notrackingurimock

# SQL Server's UNIQUE constraints count two NULLs as duplicates, so it permits only one
# NULL per unique key. Every other engine treats NULLs as distinct.
_NULLS_ARE_DUPLICATES = {"mssql"}


@pytest.fixture(autouse=True)
def _enable_workspaces(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")


@pytest.fixture
def workspace_store(_enable_workspaces):
    store = WorkspaceStore(MLFLOW_TRACKING_URI.get())
    try:
        yield store
    finally:
        store._engine.dispose()


@pytest.fixture
def dialect():
    return sa.create_engine(MLFLOW_TRACKING_URI.get()).dialect.name


def test_db_backend_set_default_merges_two_unnamed_gateway_endpoints(workspace_store, dialect):
    """Records today's behaviour for gateway endpoints with no name.

    An endpoint's name is optional in the database, so these two rows coexist:

        endpoint_id  name  workspace
        <a>          NULL  <team>
        <d>          NULL  default

    Deleting <team> with SET_DEFAULT moves <a> across, leaving both unnamed endpoints in
    `default`. On SQLite, PostgreSQL and MySQL the database allows that, because
    `UNIQUE (workspace, name)` does not treat two NULLs as duplicates. SQL Server does
    treat them as duplicates, so there the move is refused -- by the database, surfaced
    through the existing IntegrityError handler.

    Either way the preflight must not invent a conflict of its own: comparing the two
    names in Python, where None == None, would refuse the merge on every engine.

    TODO: confirm with maintainers that two unnamed endpoints sharing one workspace is
    intended and not merely tolerated.
    """
    suffix = uuid.uuid4().hex[:8]
    team = f"team-{suffix}"
    moved_id = f"e-moved-{suffix}"
    resident_id = f"e-resident-{suffix}"

    workspace_store.create_workspace(Workspace(name=team, description=None))
    try:
        with workspace_store.ManagedSessionMaker(read_only=False) as session:
            session.add_all([
                SqlGatewayEndpoint(endpoint_id=moved_id, name=None, workspace=team),
                SqlGatewayEndpoint(
                    endpoint_id=resident_id, name=None, workspace=DEFAULT_WORKSPACE_NAME
                ),
            ])

        if dialect in _NULLS_ARE_DUPLICATES:
            with pytest.raises(MlflowException, match="integrity constraints"):
                workspace_store.delete_workspace(team, mode=WorkspaceDeletionMode.SET_DEFAULT)
            expected = {(moved_id, team), (resident_id, DEFAULT_WORKSPACE_NAME)}
        else:
            workspace_store.delete_workspace(team, mode=WorkspaceDeletionMode.SET_DEFAULT)
            expected = {
                (moved_id, DEFAULT_WORKSPACE_NAME),
                (resident_id, DEFAULT_WORKSPACE_NAME),
            }

        with workspace_store.ManagedSessionMaker() as session:
            actual = {
                (endpoint.endpoint_id, endpoint.workspace)
                for endpoint in session.query(SqlGatewayEndpoint).filter(
                    SqlGatewayEndpoint.endpoint_id.in_([moved_id, resident_id])
                )
            }
        assert actual == expected
    finally:
        with workspace_store.ManagedSessionMaker(read_only=False) as session:
            session.query(SqlGatewayEndpoint).filter(
                SqlGatewayEndpoint.endpoint_id.in_([moved_id, resident_id])
            ).delete(synchronize_session=False)
        if any(w.name == team for w in workspace_store.list_workspaces()):
            workspace_store.delete_workspace(team, mode=WorkspaceDeletionMode.CASCADE)


def test_db_backend_set_default_still_blocks_a_real_name_clash(workspace_store, dialect):
    # A genuine duplicate name is a conflict on every
    # engine, so the preflight must refuse it on all four and leave both rows in place.
    suffix = uuid.uuid4().hex[:8]
    team = f"team-{suffix}"
    shared_name = f"gpt-{suffix}"
    moved_id = f"e-clash-a-{suffix}"
    resident_id = f"e-clash-d-{suffix}"

    workspace_store.create_workspace(Workspace(name=team, description=None))
    try:
        with workspace_store.ManagedSessionMaker(read_only=False) as session:
            session.add_all([
                SqlGatewayEndpoint(endpoint_id=moved_id, name=shared_name, workspace=team),
                SqlGatewayEndpoint(
                    endpoint_id=resident_id,
                    name=shared_name,
                    workspace=DEFAULT_WORKSPACE_NAME,
                ),
            ])

        with pytest.raises(MlflowException, match="already exist in the default workspace"):
            workspace_store.delete_workspace(team, mode=WorkspaceDeletionMode.SET_DEFAULT)

        with workspace_store.ManagedSessionMaker() as session:
            still_there = {
                (endpoint.endpoint_id, endpoint.workspace)
                for endpoint in session.query(SqlGatewayEndpoint).filter(
                    SqlGatewayEndpoint.endpoint_id.in_([moved_id, resident_id])
                )
            }
        assert still_there == {(moved_id, team), (resident_id, DEFAULT_WORKSPACE_NAME)}
    finally:
        with workspace_store.ManagedSessionMaker(read_only=False) as session:
            session.query(SqlGatewayEndpoint).filter(
                SqlGatewayEndpoint.endpoint_id.in_([moved_id, resident_id])
            ).delete(synchronize_session=False)
        if any(w.name == team for w in workspace_store.list_workspaces()):
            workspace_store.delete_workspace(team, mode=WorkspaceDeletionMode.CASCADE)
