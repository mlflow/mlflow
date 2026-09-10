from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from mlflow.entities.workspace import Workspace, WorkspaceDeletionMode
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    SqlSkill,
    SqlSkillVersion,
)
from mlflow.store.workspace.dbmodels.models import SqlWorkspace
from mlflow.store.workspace.sqlalchemy_store import _WORKSPACE_ROOT_MODELS, SqlAlchemyStore
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


@pytest.fixture
def workspace_store(db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")

    store = SqlAlchemyStore(db_uri)

    with store.ManagedSessionMaker(read_only=False) as session:
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
        Workspace(
            name="team-a",
            description="Team A",
            default_artifact_root="s3://root/team-a",
            trace_archival_location="s3://archive/team-a",
            trace_archival_retention="30d",
        ),
    )
    assert created.name == "team-a"
    assert created.description == "Team A"
    assert created.default_artifact_root == "s3://root/team-a"
    assert created.trace_archival_location == "s3://archive/team-a"
    assert created.trace_archival_retention == "30d"
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


def test_create_workspace_invalid_trace_archival_location_raises(workspace_store):
    with pytest.raises(MlflowException, match="proxy-only `mlflow-artifacts:` scheme") as exc:
        workspace_store.create_workspace(
            Workspace(
                name="team-a",
                description=None,
                trace_archival_location="mlflow-artifacts:/archive/team-a",
            )
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_workspace_unsupported_trace_archival_location_raises(workspace_store):
    class UnsupportedArchiveRepo(ArtifactRepository):
        def log_artifact(self, local_file, artifact_path=None):
            raise NotImplementedError

        def log_artifacts(self, local_dir, artifact_path=None):
            raise NotImplementedError

        def list_artifacts(self, path=None):
            raise NotImplementedError

    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=UnsupportedArchiveRepo("dbfs:/archive/team-a"),
    ):
        with pytest.raises(
            MlflowException,
            match="does not support deleting archived payloads",
        ) as exc:
            workspace_store.create_workspace(
                Workspace(
                    name="team-a",
                    description=None,
                    trace_archival_location="dbfs:/archive/team-a",
                )
            )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_workspace_invalid_trace_archival_retention_raises(workspace_store):
    with pytest.raises(MlflowException, match="Trace archival retention must") as exc:
        workspace_store.create_workspace(
            Workspace(
                name="team-a",
                description=None,
                trace_archival_retention="thirty-days",
            )
        )
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


def test_update_workspace_sets_trace_archival_location(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", trace_archival_location="s3://archive/team-a")
    )
    assert updated.trace_archival_location == "s3://archive/team-a"
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.trace_archival_location == "s3://archive/team-a"


def test_update_workspace_invalid_trace_archival_location_raises(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    with pytest.raises(MlflowException, match="proxy-only `mlflow-artifacts:` scheme") as exc:
        workspace_store.update_workspace(
            Workspace(name="team-a", trace_archival_location="mlflow-artifacts:/archive/team-a")
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_workspace_unsupported_trace_archival_location_raises(workspace_store):
    class UnsupportedArchiveRepo(ArtifactRepository):
        def log_artifact(self, local_file, artifact_path=None):
            raise NotImplementedError

        def log_artifacts(self, local_dir, artifact_path=None):
            raise NotImplementedError

        def list_artifacts(self, path=None):
            raise NotImplementedError

    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=UnsupportedArchiveRepo("dbfs:/archive/team-a"),
    ):
        with pytest.raises(
            MlflowException,
            match="does not support deleting archived payloads",
        ) as exc:
            workspace_store.update_workspace(
                Workspace(name="team-a", trace_archival_location="dbfs:/archive/team-a")
            )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_workspace_can_clear_trace_archival_location(workspace_store):
    workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description="old",
            trace_archival_location="s3://archive/team-a",
        )
    )

    cleared = workspace_store.update_workspace(Workspace(name="team-a", trace_archival_location=""))
    assert cleared.trace_archival_location is None
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.trace_archival_location is None


def test_update_workspace_sets_trace_archival_retention(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", trace_archival_retention="14d")
    )
    assert updated.trace_archival_retention == "14d"
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.trace_archival_retention == "14d"


def test_update_workspace_invalid_trace_archival_retention_raises(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    with pytest.raises(MlflowException, match="Trace archival retention must") as exc:
        workspace_store.update_workspace(
            Workspace(name="team-a", trace_archival_retention="thirty-days")
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_workspace_can_clear_trace_archival_retention(workspace_store):
    workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description="old",
            trace_archival_retention="30d",
        )
    )

    cleared = workspace_store.update_workspace(
        Workspace(name="team-a", trace_archival_retention="")
    )
    assert cleared.trace_archival_retention is None
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.trace_archival_retention is None


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


def test_resolve_trace_archival_config_returns_defaults(workspace_store):
    config = workspace_store.resolve_trace_archival_config(
        default_trace_archival_root="s3://archive/default",
        default_retention="30d",
        workspace_name=DEFAULT_WORKSPACE_NAME,
    )
    assert config.config.location == "s3://archive/default"
    assert config.append_workspace_prefix
    assert config.config.retention == "30d"


def test_resolve_trace_archival_config_prefers_workspace_overrides(workspace_store):
    workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description=None,
            trace_archival_location="s3://archive/team-a",
            trace_archival_retention="14d",
        )
    )

    config = workspace_store.resolve_trace_archival_config(
        default_trace_archival_root="s3://archive/default",
        default_retention="30d",
        workspace_name="team-a",
    )
    assert config.config.location == "s3://archive/team-a"
    assert not config.append_workspace_prefix
    assert config.config.retention == "14d"


def test_resolve_trace_archival_config_cache_updates_on_override_change(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-cache", description=None))

    initial = workspace_store.resolve_trace_archival_config(
        default_trace_archival_root="s3://archive/default",
        default_retention="30d",
        workspace_name="team-cache",
    )
    assert initial.config.location == "s3://archive/default"
    assert initial.append_workspace_prefix
    assert initial.config.retention == "30d"

    workspace_store.update_workspace(
        Workspace(
            name="team-cache",
            trace_archival_location="s3://archive/team-cache",
            trace_archival_retention="7d",
        )
    )

    updated = workspace_store.resolve_trace_archival_config(
        default_trace_archival_root="s3://archive/default",
        default_retention="30d",
        workspace_name="team-cache",
    )
    assert updated.config.location == "s3://archive/team-cache"
    assert not updated.append_workspace_prefix
    assert updated.config.retention == "7d"


def test_get_default_workspace_returns_default(workspace_store):
    default_ws = workspace_store.get_default_workspace()
    assert default_ws.name == DEFAULT_WORKSPACE_NAME
    assert default_ws.description is not None


def test_delete_workspace_reassigns_resources_to_default(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "exp-in-team-a", "ws": "team-a"},
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.SET_DEFAULT)

    with workspace_store.ManagedSessionMaker() as session:
        row = session.execute(
            sa.text("SELECT workspace FROM experiments WHERE name = :name"),
            {"name": "exp-in-team-a"},
        ).fetchone()
        assert row[0] == DEFAULT_WORKSPACE_NAME


def test_delete_workspace_fails_on_naming_conflict(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "shared-exp", "ws": "team-a"},
        )
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "shared-exp", "ws": DEFAULT_WORKSPACE_NAME},
        )

    with pytest.raises(MlflowException, match="already exist in the default workspace") as exc:
        workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.SET_DEFAULT)
    assert exc.value.error_code == "INVALID_STATE"

    # Workspace should still exist (transaction rolled back)
    ws = workspace_store.get_workspace("team-a")
    assert ws.name == "team-a"


def _insert_skill(session, *, workspace, organization, name):
    session.execute(
        sa.text(
            "INSERT INTO skills (workspace, organization, name, creation_timestamp, "
            "last_updated_timestamp) VALUES (:ws, :org, :name, 0, 0)"
        ),
        {"ws": workspace, "org": organization, "name": name},
    )


def test_delete_workspace_set_default_allows_same_name_different_organization(workspace_store):
    # Skills are keyed on (organization, name), so the same name under different
    # organizations is not a conflict when merging into the default workspace.
    workspace_store.create_workspace(Workspace(name="team-a", description=None))
    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        _insert_skill(session, workspace="team-a", organization="acme", name="code-review")
        _insert_skill(
            session, workspace=DEFAULT_WORKSPACE_NAME, organization="other", name="code-review"
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.SET_DEFAULT)

    with workspace_store.ManagedSessionMaker() as session:
        moved = session.execute(
            sa.text("SELECT COUNT(*) FROM skills WHERE workspace = :ws"),
            {"ws": DEFAULT_WORKSPACE_NAME},
        ).scalar_one()
    assert moved == 2


def test_delete_workspace_set_default_blocks_same_organization_and_name(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))
    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        _insert_skill(session, workspace="team-a", organization="acme", name="code-review")
        _insert_skill(
            session, workspace=DEFAULT_WORKSPACE_NAME, organization="acme", name="code-review"
        )

    with pytest.raises(MlflowException, match="already exist in the default workspace"):
        workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.SET_DEFAULT)
    assert workspace_store.get_workspace("team-a").name == "team-a"


def test_workspace_root_models_order_agent_plugin_before_skill():
    # CASCADE workspace-delete relies on this ordering: a plugin version's member
    # rows (agent_plugin_version_members -> skill_versions, NO ACTION) must be deleted
    # before the skill versions they reference. There is no ORM relationship for
    # SQLAlchemy to derive the dependency, so only this list order enforces it.
    assert _WORKSPACE_ROOT_MODELS.index(SqlAgentPlugin) < _WORKSPACE_ROOT_MODELS.index(SqlSkill)


def test_delete_workspace_cascade_removes_skill_and_plugin_graph(workspace_store):
    # End-to-end guard for the ordering above: a plugin whose member references a
    # skill version, all in one workspace, must CASCADE-delete cleanly.
    workspace_store.create_workspace(Workspace(name="team-a", description=None))
    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.add(SqlSkill(workspace="team-a", organization="acme", name="code-review"))
        session.add(
            SqlSkillVersion(
                workspace="team-a",
                organization="acme",
                name="code-review",
                version=1,
                source_type="git",
                source="s.git",
            )
        )
        session.add(SqlAgentPlugin(workspace="team-a", organization="acme", name="pr"))
        session.add(
            SqlAgentPluginVersion(
                workspace="team-a",
                organization="acme",
                name="pr",
                version="1.0.0",
                plugin_json={"name": "pr", "version": "1.0.0"},
                source_type="assembled",
                source="assembled",
            )
        )
        # Flush the parents first: the member has no ORM relationship to
        # skill_versions (only a DB FK), so its insert must follow that row.
        session.flush()
        session.add(
            SqlAgentPluginVersionMember(
                plugin_workspace="team-a",
                plugin_organization="acme",
                plugin_name="pr",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name="code-review",
                member_version=1,
            )
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    with workspace_store.ManagedSessionMaker() as session:
        for model in (
            SqlSkill,
            SqlSkillVersion,
            SqlAgentPlugin,
            SqlAgentPluginVersion,
            SqlAgentPluginVersionMember,
        ):
            assert session.query(model).count() == 0


def test_delete_workspace_cascade_removes_resources(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "exp-in-team-a", "ws": "team-a"},
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    with workspace_store.ManagedSessionMaker() as session:
        row = session.execute(
            sa.text("SELECT count(*) FROM experiments WHERE name = :name"),
            {"name": "exp-in-team-a"},
        ).scalar()
        assert row == 0

    with pytest.raises(MlflowException, match="not found"):
        workspace_store.get_workspace("team-a")


def test_delete_workspace_cascade_removes_experiment_with_runs(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (experiment_id, name, workspace, lifecycle_stage) "
                "VALUES (:id, :name, :ws, 'active')"
            ),
            {"id": 999, "name": "exp-with-runs", "ws": "team-a"},
        )
        session.execute(
            sa.text(
                "INSERT INTO runs (run_uuid, name, experiment_id, lifecycle_stage, status, "
                "source_type, start_time, end_time) "
                "VALUES (:run_id, :name, :exp_id, 'active', 'FINISHED', 'LOCAL', 0, 0)"
            ),
            {"run_id": "run-in-team-a", "name": "test-run", "exp_id": 999},
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    with workspace_store.ManagedSessionMaker() as session:
        exp_count = session.execute(
            sa.text("SELECT count(*) FROM experiments WHERE name = :name"),
            {"name": "exp-with-runs"},
        ).scalar()
        assert exp_count == 0
        run_count = session.execute(
            sa.text("SELECT count(*) FROM runs WHERE run_uuid = :run_id"),
            {"run_id": "run-in-team-a"},
        ).scalar()
        assert run_count == 0


def test_delete_workspace_restrict_blocks_when_resources_exist(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "exp-in-team-a", "ws": "team-a"},
        )

    with pytest.raises(MlflowException, match="still contains") as exc:
        workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.RESTRICT)
    assert exc.value.error_code == "INVALID_STATE"

    # Workspace and resources should still exist
    ws = workspace_store.get_workspace("team-a")
    assert ws.name == "team-a"
    with workspace_store.ManagedSessionMaker() as session:
        row = session.execute(
            sa.text("SELECT workspace FROM experiments WHERE name = :name"),
            {"name": "exp-in-team-a"},
        ).fetchone()
        assert row[0] == "team-a"


def test_delete_workspace_restrict_allows_empty_workspace(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.RESTRICT)

    with pytest.raises(MlflowException, match="not found"):
        workspace_store.get_workspace("team-a")
