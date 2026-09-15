import uuid
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy as sa

from mlflow.entities import ExperimentTag, TraceInfo
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.entities.workspace import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.store.db.workspace_move import _SPEC_BY_MODEL, MoveResult, move_resources
from mlflow.store.model_registry.sqlalchemy_workspace_store import (
    WorkspaceAwareSqlAlchemyStore as WorkspaceAwareRegistryStore,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    SqlSkill,
    SqlSkillVersion,
)
from mlflow.store.workspace.sqlalchemy_store import (
    SqlAlchemyStore as WorkspaceStore,
)
from mlflow.tracking._tracking_service import utils as tracking_utils
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


@pytest.fixture(autouse=True)
def _enable_workspaces(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")


@pytest.fixture
def tracking_store(tmp_path, db_uri, _enable_workspaces):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = tracking_utils._get_sqlalchemy_store(db_uri, artifact_dir.as_uri())
    try:
        yield store
    finally:
        store._dispose_engine()


@pytest.fixture
def registry_store(db_uri, _enable_workspaces):
    store = WorkspaceAwareRegistryStore(db_uri)
    try:
        yield store
    finally:
        store.engine.dispose()


@pytest.fixture
def workspace_store(db_uri, _enable_workspaces):
    store = WorkspaceStore(db_uri)
    try:
        yield store
    finally:
        store._engine.dispose()


@pytest.fixture
def engine(tracking_store):
    return tracking_store.engine


def _create_workspace(ws_store, name):
    ws_store.create_workspace(Workspace(name=name))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def test_move_experiments_by_name(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        tracking_store.create_experiment("exp-1")
        tracking_store.create_experiment("exp-2")
        tracking_store.create_experiment("exp-3")

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        names=["exp-1", "exp-3"],
    )

    assert result.names == ["exp-1", "exp-3"]
    assert result.row_count == 2

    with WorkspaceContext("team-a"):
        assert tracking_store.get_experiment_by_name("exp-1") is not None
        assert tracking_store.get_experiment_by_name("exp-3") is not None
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        assert tracking_store.get_experiment_by_name("exp-2") is not None
        assert tracking_store.get_experiment_by_name("exp-1") is None


def test_move_experiment_by_tag(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        exp_id_1 = tracking_store.create_experiment("exp-1")
        tracking_store.create_experiment("exp-2")
        tracking_store.set_experiment_tag(exp_id_1, ExperimentTag("team", "team-a"))

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        tags=[("team", "team-a")],
    )

    assert result.names == ["exp-1"]
    assert result.row_count == 1


def test_error_name_and_tag_mutually_exclusive(workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="experiments",
            names=["exp-1"],
            tags=[("team", "team-a")],
        )


def test_move_experiment_by_multiple_tags_and(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        exp_id_1 = tracking_store.create_experiment("exp-1")
        exp_id_2 = tracking_store.create_experiment("exp-2")
        tracking_store.set_experiment_tag(exp_id_1, ExperimentTag("team", "team-a"))
        tracking_store.set_experiment_tag(exp_id_1, ExperimentTag("env", "prod"))
        tracking_store.set_experiment_tag(exp_id_2, ExperimentTag("team", "team-a"))

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        tags=[("team", "team-a"), ("env", "prod")],
    )

    assert result.names == ["exp-1"]
    assert result.row_count == 1


def test_move_all_experiments_no_filter(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        tracking_store.create_experiment("exp-1")
        tracking_store.create_experiment("exp-2")

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
    )

    assert len(result.names) >= 2
    assert "exp-1" in result.names
    assert "exp-2" in result.names
    assert result.row_count >= 2


# ---------------------------------------------------------------------------
# Registered models
# ---------------------------------------------------------------------------


def test_move_model_by_tag(registry_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        registry_store.create_registered_model("model-1")
        registry_store.create_registered_model("model-2")
        registry_store.set_registered_model_tag("model-1", RegisteredModelTag("team", "team-a"))

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="registered_models",
        tags=[("team", "team-a")],
    )

    assert result.names == ["model-1"]
    assert result.row_count == 1

    with WorkspaceContext("team-a"):
        assert registry_store.get_registered_model("model-1") is not None
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        assert registry_store.get_registered_model("model-2") is not None


def test_move_model_cascades_to_child_tables(registry_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        registry_store.create_registered_model("model-1")
        mv1 = registry_store.create_model_version(
            "model-1", "s3://bucket/v1", run_id=uuid.uuid4().hex
        )
        registry_store.create_model_version("model-1", "s3://bucket/v2", run_id=uuid.uuid4().hex)
        registry_store.set_registered_model_tag("model-1", RegisteredModelTag("stage", "prod"))
        registry_store.set_registered_model_alias("model-1", "champion", str(mv1.version))
        registry_store.set_model_version_tag(
            "model-1", str(mv1.version), ModelVersionTag("metric", "0.95")
        )

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="registered_models",
        names=["model-1"],
    )

    assert result.names == ["model-1"]
    assert result.row_count == 1

    with WorkspaceContext("team-a"):
        rm = registry_store.get_registered_model("model-1")
        assert rm.tags == {"stage": "prod"}
        assert rm.aliases == {"champion": mv1.version}

        versions = registry_store.search_model_versions(filter_string="name='model-1'")
        assert len(versions) == 2

        mv = registry_store.get_model_version("model-1", str(mv1.version))
        assert mv.tags == {"metric": "0.95"}


# ---------------------------------------------------------------------------
# Conflict detection, dry run, and error handling
# ---------------------------------------------------------------------------


def test_conflict_detection(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        tracking_store.create_experiment("dup-exp")
    with WorkspaceContext("team-a"):
        tracking_store.create_experiment("dup-exp")

    with pytest.raises(RuntimeError, match="already exist in workspace"):
        move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="experiments",
            names=["dup-exp"],
        )

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        assert tracking_store.get_experiment_by_name("dup-exp") is not None


def test_dry_run_does_not_modify(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        tracking_store.create_experiment("exp-1")

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        names=["exp-1"],
        dry_run=True,
    )

    assert result.names == ["exp-1"]
    assert result.row_count == 1

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        assert tracking_store.get_experiment_by_name("exp-1") is not None


def test_validation_errors(workspace_store, engine):
    with pytest.raises(MlflowException, match="not found"):
        move_resources(
            engine,
            workspace_store,
            source_workspace="nonexistent",
            target_workspace=DEFAULT_WORKSPACE_NAME,
            resource_type="experiments",
        )

    with pytest.raises(MlflowException, match="not found"):
        move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="nonexistent",
            resource_type="experiments",
        )

    with pytest.raises(RuntimeError, match="must be different"):
        move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace=DEFAULT_WORKSPACE_NAME,
            resource_type="experiments",
        )


def test_error_tag_on_unsupported_resource_type(workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with pytest.raises(RuntimeError, match="does not support tag filtering"):
        move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="webhooks",
            tags=[("team", "team-a")],
        )


def test_noop_when_nothing_matches(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        tracking_store.create_experiment("exp-1")

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        names=["nonexistent"],
    )
    assert result == MoveResult(names=[], row_count=0)

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        tags=[("team", "nonexistent")],
    )
    assert result == MoveResult(names=[], row_count=0)


# ---------------------------------------------------------------------------
# All resource types
# ---------------------------------------------------------------------------


def test_move_all_resource_types(
    tracking_store, registry_store, workspace_store, engine, monkeypatch
):
    from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
    from mlflow.store.jobs.sqlalchemy_workspace_store import (
        WorkspaceAwareSqlAlchemyJobStore,
    )

    monkeypatch.setattr(
        "mlflow.store.model_registry.sqlalchemy_store._validate_webhook_url", lambda url: None
    )
    _create_workspace(workspace_store, "target")
    job_store = WorkspaceAwareSqlAlchemyJobStore(str(engine.url))

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        tracking_store.create_experiment("move-exp")
        tracking_store.create_dataset("move-ds")
        registry_store.create_registered_model("move-model")
        registry_store.create_webhook(
            name="move-webhook",
            url="https://example.com/hook",
            events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
        )
        job_store.create_job(job_name="move-job", params="{}")

    resource_names = {
        "experiments": "move-exp",
        "evaluation_datasets": "move-ds",
        "registered_models": "move-model",
        "webhooks": "move-webhook",
        "jobs": "move-job",
    }

    for resource_type, name in resource_names.items():
        result = move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="target",
            resource_type=resource_type,
            names=[name],
        )
        assert name in result.names, f"Expected {name!r} in moved list for {resource_type}"
        assert result.row_count >= 1, f"Expected row_count >= 1 for {resource_type}"


# ---------------------------------------------------------------------------
# Spec coverage
# ---------------------------------------------------------------------------


def test_all_workspace_root_models_have_spec():
    from mlflow.store.tracking.dbmodels.models import (
        SqlGatewayBudgetPolicy,
        SqlGatewayEndpoint,
        SqlGatewayGuardrail,
        SqlGatewayModelDefinition,
        SqlGatewaySecret,
    )
    from mlflow.store.workspace.sqlalchemy_store import _WORKSPACE_ROOT_MODELS

    # Gateway resources are intentionally excluded due to inter-table FK
    # dependencies that make moving them independently unsafe.
    _INTENTIONALLY_OMITTED = {
        SqlGatewaySecret,
        SqlGatewayEndpoint,
        SqlGatewayModelDefinition,
        SqlGatewayBudgetPolicy,
        SqlGatewayGuardrail,
    }

    missing = {
        model.__name__
        for model in _WORKSPACE_ROOT_MODELS
        if model not in _SPEC_BY_MODEL and model not in _INTENTIONALLY_OMITTED
    }
    assert not missing, (
        f"These models are in _WORKSPACE_ROOT_MODELS but have no entry in "
        f"_SPEC_BY_MODEL (mlflow/store/db/workspace_move.py): {sorted(missing)}. "
        f"Add a _ResourceSpec so move-resources can handle them, or add "
        f"to _INTENTIONALLY_OMITTED if they cannot be moved independently."
    )


# ---------------------------------------------------------------------------
# Artifact root retargeting (--artifact-policy retarget)
# ---------------------------------------------------------------------------


def _seed_experiment_with_artifacts(tracking_store, name):
    """Create an experiment in the default workspace with a run artifact file, a
    logged model and a trace, so tests can verify what retargeting changes and
    what it leaves untouched.
    """
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        exp_id = tracking_store.create_experiment(name)
        run = tracking_store.create_run(exp_id, user_id="u", start_time=0, tags=[], run_name="run")
        run_dir = Path(local_file_uri_to_path(run.info.artifact_uri))
        run_dir.mkdir(parents=True)
        (run_dir / "model.txt").write_text("weights")
        model = tracking_store.create_logged_model(experiment_id=exp_id)
        trace_info = TraceInfo(
            trace_id=f"tr-{uuid.uuid4().hex}",
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=1_000,
            execution_duration=2_000,
            state=TraceState.OK,
        )
        tracking_store.start_trace(trace_info)
    return exp_id, run, model, trace_info


def test_move_experiments_artifact_policy_retarget(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    exp_id, run, model, trace_info = _seed_experiment_with_artifacts(tracking_store, "exp-rt")
    server_root = tracking_store.artifact_root_uri
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        old_location = tracking_store.get_experiment(exp_id).artifact_location

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        names=["exp-rt"],
        artifact_policy="retarget",
        default_artifact_root=server_root,
    )

    assert result.retarget_root == f"{server_root}/workspaces/team-a"
    new_location = f"{server_root}/workspaces/team-a/{exp_id}"

    with WorkspaceContext("team-a"):
        experiment = tracking_store.get_experiment(exp_id)
        assert experiment.artifact_location == new_location

        # Everything already logged keeps its absolute URIs and stays readable.
        moved_run = tracking_store.get_run(run.info.run_id)
        assert moved_run.info.artifact_uri == run.info.artifact_uri
        old_file = Path(local_file_uri_to_path(moved_run.info.artifact_uri)) / "model.txt"
        assert old_file.read_text() == "weights"
        moved_model = tracking_store.get_logged_model(model.model_id)
        assert moved_model.artifact_location == model.artifact_location
        moved_trace = tracking_store.get_trace_info(trace_info.trace_id)
        assert moved_trace.tags[MLFLOW_ARTIFACT_LOCATION].startswith(old_location)

        # New runs land under the retargeted root.
        new_run = tracking_store.create_run(
            exp_id, user_id="u", start_time=0, tags=[], run_name="run2"
        )
        assert new_run.info.artifact_uri.startswith(new_location)


def test_move_experiments_artifact_policy_preserve_leaves_uris(
    tracking_store, workspace_store, engine
):
    _create_workspace(workspace_store, "team-a")
    exp_id, run, _, _ = _seed_experiment_with_artifacts(tracking_store, "exp-keep")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        old_location = tracking_store.get_experiment(exp_id).artifact_location

    move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        names=["exp-keep"],
    )

    with WorkspaceContext("team-a"):
        assert tracking_store.get_experiment(exp_id).artifact_location == old_location
        assert tracking_store.get_run(run.info.run_id).info.artifact_uri == run.info.artifact_uri


def test_move_retarget_dry_run_makes_no_changes(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    exp_id, _, _, _ = _seed_experiment_with_artifacts(tracking_store, "exp-dry")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        old_location = tracking_store.get_experiment(exp_id).artifact_location

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        names=["exp-dry"],
        dry_run=True,
        artifact_policy="retarget",
        default_artifact_root=tracking_store.artifact_root_uri,
    )

    assert result.retarget_root is not None
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        assert tracking_store.get_experiment_by_name("exp-dry") is not None
        assert tracking_store.get_experiment(exp_id).artifact_location == old_location


def test_move_retarget_overrides_custom_artifact_locations(tracking_store, workspace_store, engine):
    # Experiment-level artifact roots cannot be set while workspaces are enabled, so
    # a custom location can only predate workspaces and is repointed like any other.
    _create_workspace(workspace_store, "team-a")
    exp_id, _, _, _ = _seed_experiment_with_artifacts(tracking_store, "exp-custom")
    server_root = tracking_store.artifact_root_uri
    experiments_table = sa.Table("experiments", sa.MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        conn.execute(
            experiments_table
            .update()
            .where(experiments_table.c.experiment_id == int(exp_id))
            .values(artifact_location="s3://custom-bucket/some/path")
        )

    move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        names=["exp-custom"],
        artifact_policy="retarget",
        default_artifact_root=server_root,
    )

    with WorkspaceContext("team-a"):
        location = tracking_store.get_experiment(exp_id).artifact_location
        assert location == f"{server_root}/workspaces/team-a/{exp_id}"


def test_move_retarget_errors_without_artifact_root(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    _seed_experiment_with_artifacts(tracking_store, "exp-noroot")

    with pytest.raises(RuntimeError, match="Cannot determine the artifact root"):
        move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="experiments",
            names=["exp-noroot"],
            artifact_policy="retarget",
        )

    # The root is resolved before any writes, so nothing moved.
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        assert tracking_store.get_experiment_by_name("exp-noroot") is not None


def test_move_retarget_rejected_for_non_experiments(engine, workspace_store):
    _create_workspace(workspace_store, "team-a")
    with pytest.raises(RuntimeError, match="only supported for --resource-type experiments"):
        move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="registered_models",
            artifact_policy="retarget",
            default_artifact_root="s3://root",
        )


def test_move_retarget_uses_workspace_artifact_root(
    tracking_store, workspace_store, engine, tmp_path
):
    workspace_root = (tmp_path / "team-b-root").as_uri()
    workspace_store.create_workspace(Workspace(name="team-b", default_artifact_root=workspace_root))
    exp_id, _, _, _ = _seed_experiment_with_artifacts(tracking_store, "exp-wsroot")

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-b",
        resource_type="experiments",
        names=["exp-wsroot"],
        artifact_policy="retarget",
        default_artifact_root=tracking_store.artifact_root_uri,
    )

    # A workspace-level artifact root is used as is, without the workspaces/<name> suffix.
    assert result.retarget_root == workspace_root
    with WorkspaceContext("team-b"):
        location = tracking_store.get_experiment(exp_id).artifact_location
        assert location == f"{workspace_root}/{exp_id}"


def test_move_retarget_with_tag_filter(tracking_store, workspace_store, engine):
    # The tag subquery in the name filter is scoped to the source workspace, so
    # the retarget must resolve the matched experiments before the workspace flip.
    _create_workspace(workspace_store, "team-a")
    exp_id, _, _, _ = _seed_experiment_with_artifacts(tracking_store, "exp-tagged")
    server_root = tracking_store.artifact_root_uri
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        tracking_store.set_experiment_tag(exp_id, ExperimentTag("team", "team-a"))
        untagged_id = tracking_store.create_experiment("exp-untagged")

    result = move_resources(
        engine,
        workspace_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        tags=[("team", "team-a")],
        artifact_policy="retarget",
        default_artifact_root=server_root,
    )

    assert result.names == ["exp-tagged"]
    with WorkspaceContext("team-a"):
        location = tracking_store.get_experiment(exp_id).artifact_location
        assert location == f"{server_root}/workspaces/team-a/{exp_id}"
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        untagged = tracking_store.get_experiment(untagged_id)
        assert "/workspaces/team-a/" not in untagged.artifact_location


def test_move_validates_and_resolves_through_workspace_provider(
    tracking_store, workspace_store, engine, tmp_path
):
    # The workspace provider is not necessarily backed by the tracking database.
    # Validation and root resolution must go through the provider, so the move
    # works even when the tracking database's workspaces table has no rows.
    exp_id, _, _, _ = _seed_experiment_with_artifacts(tracking_store, "exp-ext")
    server_root = tracking_store.artifact_root_uri

    provider_store = WorkspaceStore(f"sqlite:///{tmp_path / 'workspaces.db'}")
    _create_workspace(provider_store, "team-ext")
    workspaces_table = sa.Table("workspaces", sa.MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        conn.execute(workspaces_table.delete())

    result = move_resources(
        engine,
        provider_store,
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-ext",
        resource_type="experiments",
        names=["exp-ext"],
        artifact_policy="retarget",
        default_artifact_root=server_root,
    )

    assert result.names == ["exp-ext"]
    assert result.retarget_root == f"{server_root}/workspaces/team-ext"
    experiments_table = sa.Table("experiments", sa.MetaData(), autoload_with=engine)
    with engine.connect() as conn:
        row = conn.execute(
            sa.select(experiments_table.c.workspace, experiments_table.c.artifact_location).where(
                experiments_table.c.experiment_id == int(exp_id)
            )
        ).one()
    assert row == ("team-ext", f"{server_root}/workspaces/team-ext/{exp_id}")


def test_move_retarget_failure_rolls_back_move(tracking_store, workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    exp_id, _, _, _ = _seed_experiment_with_artifacts(tracking_store, "exp-fail")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        old_location = tracking_store.get_experiment(exp_id).artifact_location

    # Fail the workspace flip, which runs after the artifact location update,
    # to verify both writes share the move transaction.
    executed: list[str] = []
    original_execute = sa.engine.Connection.execute

    def failing_execute(self, statement, *args, **kwargs):
        compiled = str(statement)
        executed.append(compiled)
        if "SET workspace=" in compiled:
            raise RuntimeError("boom")
        return original_execute(self, statement, *args, **kwargs)

    with mock.patch.object(sa.engine.Connection, "execute", failing_execute):
        with pytest.raises(RuntimeError, match="boom"):
            move_resources(
                engine,
                workspace_store,
                source_workspace=DEFAULT_WORKSPACE_NAME,
                target_workspace="team-a",
                resource_type="experiments",
                names=["exp-fail"],
                artifact_policy="retarget",
                default_artifact_root=tracking_store.artifact_root_uri,
            )

    assert any("SET artifact_location=" in statement for statement in executed)
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        experiment = tracking_store.get_experiment(exp_id)
        assert experiment.artifact_location == old_location


# ---------------------------------------------------------------------------
# Skill registry (skills / agent_plugins): bundle-aware moves
# ---------------------------------------------------------------------------


def _fk_on_engine(db_uri):
    """A fresh engine that enforces foreign keys (SQLite leaves them off by default),
    so bundle moves are exercised under the same enforcement as MySQL/Postgres.
    """
    eng = sa.create_engine(db_uri)
    if eng.dialect.name == "sqlite":
        sa.event.listen(
            eng, "connect", lambda dbapi, record: dbapi.execute("PRAGMA foreign_keys=ON")
        )
    return eng


def _add_skill(session, workspace, name, *, org="", versions=(1,)):
    session.add(SqlSkill(workspace=workspace, organization=org, name=name))
    session.flush()
    for version in versions:
        session.add(
            SqlSkillVersion(
                workspace=workspace,
                organization=org,
                name=name,
                version=version,
                source_type="git",
                source="https://github.com/acme/skills.git",
                status="active",
            )
        )
    session.flush()


def _add_plugin(session, workspace, name, version_statuses, *, org=""):
    session.add(SqlAgentPlugin(workspace=workspace, organization=org, name=name))
    session.flush()
    for version, status in version_statuses:
        session.add(
            SqlAgentPluginVersion(
                workspace=workspace,
                organization=org,
                name=name,
                version=version,
                plugin_json={"name": name, "version": version},
                source_type="assembled",
                status=status,
            )
        )
    session.flush()


def _add_member(session, workspace, plugin_name, plugin_version, skill_name, *, skill_version=1):
    session.add(
        SqlAgentPluginVersionMember(
            plugin_workspace=workspace,
            plugin_organization="",
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            member_name=skill_name,
            member_organization="",
            member_version=skill_version,
        )
    )
    session.flush()


def _scalar(conn, sql, **params):
    return conn.execute(sa.text(sql), params).scalar()


def test_move_agent_plugin_moves_its_skills(tracking_store, workspace_store, db_uri):
    _create_workspace(workspace_store, "team-a")
    with tracking_store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, DEFAULT_WORKSPACE_NAME, "code-review")
        _add_plugin(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", [("1.0.0", "active")])
        _add_member(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", "1.0.0", "code-review")

    engine = _fk_on_engine(db_uri)
    try:
        result = move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="agent_plugins",
            names=["pr-workflow"],
        )
        assert set(result.names) == {"pr-workflow", "code-review"}
        with engine.begin() as conn:
            assert (
                _scalar(conn, "SELECT workspace FROM agent_plugins WHERE name='pr-workflow'")
                == "team-a"
            )
            assert (
                _scalar(conn, "SELECT workspace FROM skills WHERE name='code-review'") == "team-a"
            )
            assert (
                _scalar(conn, "SELECT workspace FROM skill_versions WHERE name='code-review'")
                == "team-a"
            )
            member = conn.execute(
                sa.text(
                    "SELECT plugin_workspace, member_name FROM agent_plugin_version_members "
                    "WHERE plugin_name='pr-workflow'"
                )
            ).fetchone()
            assert member == ("team-a", "code-review")
    finally:
        engine.dispose()


def test_move_agent_plugin_refuses_when_skill_shared_with_live_plugin(
    tracking_store, workspace_store, db_uri
):
    _create_workspace(workspace_store, "team-a")
    with tracking_store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, DEFAULT_WORKSPACE_NAME, "code-review")
        _add_plugin(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", [("1.0.0", "active")])
        _add_plugin(session, DEFAULT_WORKSPACE_NAME, "release-flow", [("1.0.0", "active")])
        _add_member(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", "1.0.0", "code-review")
        _add_member(session, DEFAULT_WORKSPACE_NAME, "release-flow", "1.0.0", "code-review")

    engine = _fk_on_engine(db_uri)
    try:
        with pytest.raises(RuntimeError, match="release-flow"):
            move_resources(
                engine,
                workspace_store,
                source_workspace=DEFAULT_WORKSPACE_NAME,
                target_workspace="team-a",
                resource_type="agent_plugins",
                names=["pr-workflow"],
            )
        with engine.begin() as conn:
            assert (
                _scalar(conn, "SELECT workspace FROM agent_plugins WHERE name='pr-workflow'")
                == DEFAULT_WORKSPACE_NAME
            )
            assert (
                _scalar(conn, "SELECT workspace FROM skills WHERE name='code-review'")
                == DEFAULT_WORKSPACE_NAME
            )
    finally:
        engine.dispose()


def test_move_agent_plugins_together_moves_shared_skill(tracking_store, workspace_store, db_uri):
    _create_workspace(workspace_store, "team-a")
    with tracking_store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, DEFAULT_WORKSPACE_NAME, "code-review")
        _add_plugin(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", [("1.0.0", "active")])
        _add_plugin(session, DEFAULT_WORKSPACE_NAME, "release-flow", [("1.0.0", "active")])
        _add_member(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", "1.0.0", "code-review")
        _add_member(session, DEFAULT_WORKSPACE_NAME, "release-flow", "1.0.0", "code-review")
    engine = _fk_on_engine(db_uri)
    try:
        result = move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="agent_plugins",
            names=["pr-workflow", "release-flow"],
        )
        assert set(result.names) == {"pr-workflow", "release-flow", "code-review"}
        with engine.begin() as conn:
            for table, name in (
                ("agent_plugins", "pr-workflow"),
                ("agent_plugins", "release-flow"),
                ("skills", "code-review"),
            ):
                assert (
                    _scalar(conn, f"SELECT workspace FROM {table} WHERE name=:n", n=name)
                    == "team-a"
                )
            count = _scalar(
                conn,
                "SELECT count(*) FROM agent_plugin_version_members WHERE plugin_workspace='team-a'",
            )
            assert count == 2
    finally:
        engine.dispose()


def test_move_skill_refuses_when_member_of_live_plugin(tracking_store, workspace_store, db_uri):
    _create_workspace(workspace_store, "team-a")
    with tracking_store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, DEFAULT_WORKSPACE_NAME, "code-review")
        _add_plugin(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", [("1.0.0", "active")])
        _add_member(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", "1.0.0", "code-review")

    engine = _fk_on_engine(db_uri)
    try:
        with pytest.raises(RuntimeError, match="pr-workflow"):
            move_resources(
                engine,
                workspace_store,
                source_workspace=DEFAULT_WORKSPACE_NAME,
                target_workspace="team-a",
                resource_type="skills",
                names=["code-review"],
            )
        with engine.begin() as conn:
            assert (
                _scalar(conn, "SELECT workspace FROM skills WHERE name='code-review'")
                == DEFAULT_WORKSPACE_NAME
            )
    finally:
        engine.dispose()


def test_move_standalone_skill(tracking_store, workspace_store, db_uri):
    _create_workspace(workspace_store, "team-a")
    with tracking_store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, DEFAULT_WORKSPACE_NAME, "code-review")

    engine = _fk_on_engine(db_uri)
    try:
        result = move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="skills",
            names=["code-review"],
        )
        assert result.names == ["code-review"]
        with engine.begin() as conn:
            assert (
                _scalar(conn, "SELECT workspace FROM skills WHERE name='code-review'") == "team-a"
            )
            assert (
                _scalar(conn, "SELECT workspace FROM skill_versions WHERE name='code-review'")
                == "team-a"
            )
    finally:
        engine.dispose()


def test_move_agent_plugin_purges_dead_external_membership(tracking_store, workspace_store, db_uri):
    _create_workspace(workspace_store, "team-a")
    with tracking_store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, DEFAULT_WORKSPACE_NAME, "code-review")
        _add_plugin(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", [("1.0.0", "active")])
        # release-flow's only reference to code-review is via a soft-deleted version,
        # so it must not block the move, and its stale link must be purged.
        _add_plugin(session, DEFAULT_WORKSPACE_NAME, "release-flow", [("0.9.0", "deleted")])
        _add_member(session, DEFAULT_WORKSPACE_NAME, "pr-workflow", "1.0.0", "code-review")
        _add_member(session, DEFAULT_WORKSPACE_NAME, "release-flow", "0.9.0", "code-review")

    engine = _fk_on_engine(db_uri)
    try:
        result = move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="agent_plugins",
            names=["pr-workflow"],
        )
        assert set(result.names) == {"pr-workflow", "code-review"}
        with engine.begin() as conn:
            assert (
                _scalar(conn, "SELECT workspace FROM agent_plugins WHERE name='pr-workflow'")
                == "team-a"
            )
            assert (
                _scalar(conn, "SELECT workspace FROM skills WHERE name='code-review'") == "team-a"
            )
            # release-flow stayed behind; its stale link to the moved skill was purged.
            assert (
                _scalar(conn, "SELECT workspace FROM agent_plugins WHERE name='release-flow'")
                == DEFAULT_WORKSPACE_NAME
            )
            stale = _scalar(
                conn,
                "SELECT count(*) FROM agent_plugin_version_members WHERE plugin_name = :n",
                n="release-flow",
            )
            assert stale == 0
    finally:
        engine.dispose()


def test_move_plugin_with_org_name_syntax(tracking_store, workspace_store, db_uri):
    _create_workspace(workspace_store, "team-a")
    with tracking_store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, DEFAULT_WORKSPACE_NAME, "code-review", org="acme")
        _add_plugin(
            session, DEFAULT_WORKSPACE_NAME, "pr-workflow", [("1.0.0", "active")], org="acme"
        )
        session.add(
            SqlAgentPluginVersionMember(
                plugin_workspace=DEFAULT_WORKSPACE_NAME,
                plugin_organization="acme",
                plugin_name="pr-workflow",
                plugin_version="1.0.0",
                member_name="code-review",
                member_organization="acme",
                member_version=1,
            )
        )
        session.flush()

    engine = _fk_on_engine(db_uri)
    try:
        result = move_resources(
            engine,
            workspace_store,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="team-a",
            resource_type="agent_plugins",
            names=["@acme/pr-workflow"],
        )
        assert set(result.names) == {"acme/pr-workflow", "acme/code-review"}
        with engine.begin() as conn:
            assert (
                _scalar(
                    conn,
                    "SELECT workspace FROM agent_plugins WHERE name=:n AND organization=:o",
                    n="pr-workflow",
                    o="acme",
                )
                == "team-a"
            )
            assert (
                _scalar(
                    conn,
                    "SELECT workspace FROM skills WHERE name=:n AND organization=:o",
                    n="code-review",
                    o="acme",
                )
                == "team-a"
            )
    finally:
        engine.dispose()
