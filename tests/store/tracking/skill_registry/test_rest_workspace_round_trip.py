# End-to-end workspace round trips for the Skill REST client.
#
# test_rest_mixin.py checks that the client sends the workspace header (against a mocked
# transport), and tests/server/test_skill_registry_api.py checks that responses carry the
# workspace (against a mocked store). The round-trip fixture in test_rest_mixin.py replaces
# `http_request`, which is where the header is attached, so its requests carry none.
#
# Here the real `http_request` runs, and each request passes through the server's workspace
# middleware into a workspace-aware store. That proves the two halves agree: what a caller
# does in one workspace is stored in that workspace and cannot be seen or changed from another.

from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.server.fastapi_app import (
    add_fastapi_workspace_middleware,
    add_registry_exception_handlers,
)
from mlflow.server.skill_registry_api import skill_registry_router
from mlflow.store.tracking.dbmodels.models import (
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
)
from mlflow.store.tracking.rest_store import RestStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore as WorkspaceSqlAlchemyStore
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME, WORKSPACE_HEADER_NAME

_HOST = "https://registry.example.com"
_GIT_SOURCE = {"source_type": "git", "source": "https://example.com/skills.git"}


@pytest.fixture
def registry(tmp_path: Path, db_uri: str, monkeypatch):
    """
    Yield a REST client store, the workspace-aware store behind the server, and the workspace
    header of every request the client sent.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    workspace_store = WorkspaceSqlAlchemyStore(db_uri)
    for name in ("team-a", "team-b"):
        workspace_store.create_workspace(Workspace(name=name))
    db_store = WorkspaceAwareSqlAlchemyStore(db_uri, (tmp_path / "artifacts").as_uri())

    app = FastAPI()
    add_fastapi_workspace_middleware(app)
    add_registry_exception_handlers(app)
    app.include_router(skill_registry_router, prefix="/api/3.0/mlflow/skills")
    rest_store = RestStore(lambda: MlflowHostCreds(_HOST))
    sent_workspaces = []

    with TestClient(app) as http_client:

        def send(method, url, *args, headers=None, **kwargs):
            sent_workspaces.append(headers.get(WORKSPACE_HEADER_NAME))
            # The client and the server share this process, and the client's workspace is
            # also exported to the environment. Hide it while the server handles the request,
            # so the header is the only way the server can learn the workspace.
            with WorkspaceContext(None):
                return http_client.request(
                    method,
                    url.removeprefix(_HOST),
                    headers=headers,
                    json=kwargs.get("json"),
                    params=kwargs.get("params"),
                )

        with (
            mock.patch(
                "mlflow.server.workspace_helpers._get_workspace_store",
                return_value=workspace_store,
            ),
            mock.patch("mlflow.server.handlers._get_tracking_store", return_value=db_store),
            mock.patch.object(rest_store, "_probe_workspace_support", return_value=True),
            mock.patch("mlflow.utils.rest_utils._get_http_response_with_retries", side_effect=send),
        ):
            yield rest_store, db_store, sent_workspaces


def _stored_workspaces(db_store, model, name="reviewer"):
    # Read the workspace column directly, bypassing the store's workspace scoping.
    with db_store.ManagedSessionMaker() as session:
        return sorted(
            workspace for (workspace,) in session.query(model.workspace).filter(model.name == name)
        )


def test_rest_writes_are_stored_in_the_callers_workspace(registry):
    rest_store, db_store, sent_workspaces = registry
    with WorkspaceContext("team-a"):
        skill = rest_store.create_skill("reviewer", description="Reviews code")
        version = rest_store.create_skill_version("reviewer", **_GIT_SOURCE)
        rest_store.set_skill_tag("reviewer", key="team", value="platform")
        rest_store.set_skill_alias("reviewer", alias="production", version=version.version)

    assert skill.workspace == "team-a"
    assert version.workspace == "team-a"
    for model in (SqlSkill, SqlSkillVersion, SqlSkillTag, SqlSkillAlias):
        assert _stored_workspaces(db_store, model) == ["team-a"]
    assert sent_workspaces == ["team-a"] * 4


def test_rest_reads_return_the_callers_workspace_only(registry):
    rest_store, _, _ = registry
    with WorkspaceContext("team-a"):
        rest_store.create_skill("reviewer", description="Team A's reviewer")
        rest_store.create_skill_version("reviewer", **_GIT_SOURCE)
        rest_store.set_skill_alias("reviewer", alias="production", version=1)

    with WorkspaceContext("team-b"):
        assert list(rest_store.search_skills()) == []
        for read in (
            lambda: rest_store.get_skill("reviewer"),
            lambda: rest_store.get_skill_version("reviewer", 1),
            lambda: rest_store.get_latest_skill_version("reviewer"),
            lambda: rest_store.get_skill_version_by_alias("reviewer", "production"),
        ):
            with pytest.raises(MlflowException, match="not found") as exc_info:
                read()
            assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    with WorkspaceContext("team-a"):
        fetched = rest_store.get_skill("reviewer")
        assert [s.name for s in rest_store.search_skills()] == ["reviewer"]
    assert fetched.workspace == "team-a"
    assert fetched.description == "Team A's reviewer"
    assert fetched.aliases == {"production": 1}


@pytest.mark.parametrize(
    "write",
    [
        lambda store: store.update_skill("reviewer", description="Taken over"),
        lambda store: store.delete_skill("reviewer"),
        lambda store: store.set_skill_tag("reviewer", key="team", value="b"),
        lambda store: store.set_skill_alias("reviewer", alias="production", version=1),
        lambda store: store.delete_skill_version("reviewer", 1),
    ],
    ids=[
        "update_skill",
        "delete_skill",
        "set_skill_tag",
        "set_skill_alias",
        "delete_skill_version",
    ],
)
def test_rest_writes_cannot_reach_another_workspace(registry, write):
    rest_store, db_store, _ = registry
    with WorkspaceContext("team-a"):
        rest_store.create_skill("reviewer", description="Team A's reviewer")
        rest_store.create_skill_version("reviewer", status="draft", **_GIT_SOURCE)
        before = rest_store.get_skill("reviewer")

    with WorkspaceContext("team-b"):
        with pytest.raises(MlflowException, match="not found") as exc_info:
            write(rest_store)
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    with WorkspaceContext("team-a"):
        assert rest_store.get_skill("reviewer") == before
        assert rest_store.get_skill_version("reviewer", 1).status.value == "draft"
    assert _stored_workspaces(db_store, SqlSkill) == ["team-a"]


def test_the_same_skill_identity_is_independent_in_each_workspace(registry):
    rest_store, db_store, _ = registry
    for workspace, versions in (("team-a", 2), ("team-b", 1)):
        with WorkspaceContext(workspace):
            rest_store.create_skill("reviewer", description=f"Owned by {workspace}")
            for _ in range(versions):
                rest_store.create_skill_version("reviewer", **_GIT_SOURCE)

    assert _stored_workspaces(db_store, SqlSkill) == ["team-a", "team-b"]
    for workspace, versions in (("team-a", 2), ("team-b", 1)):
        with WorkspaceContext(workspace):
            skill = rest_store.get_skill("reviewer")
            latest = rest_store.get_latest_skill_version("reviewer")
        assert (skill.workspace, skill.description) == (workspace, f"Owned by {workspace}")
        assert (latest.workspace, latest.version) == (workspace, versions)


def test_rest_call_without_a_workspace_uses_the_default_workspace(registry):
    rest_store, db_store, sent_workspaces = registry
    skill = rest_store.create_skill("reviewer")

    assert sent_workspaces == [None]
    assert skill.workspace == DEFAULT_WORKSPACE_NAME
    assert _stored_workspaces(db_store, SqlSkill) == [DEFAULT_WORKSPACE_NAME]


def test_rest_call_for_an_unknown_workspace_is_rejected_before_the_store(registry):
    rest_store, db_store, sent_workspaces = registry
    with WorkspaceContext("team-missing"):
        with pytest.raises(MlflowException, match="team-missing") as exc_info:
            rest_store.create_skill("reviewer")

    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    assert sent_workspaces == ["team-missing"]
    assert _stored_workspaces(db_store, SqlSkill) == []
