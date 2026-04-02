import uuid

import pytest

from mlflow.entities import ExperimentTag
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.workspace import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.store.db.workspace_move import _SPEC_BY_MODEL, MoveResult, move_resources
from mlflow.store.model_registry.sqlalchemy_workspace_store import (
    WorkspaceAwareSqlAlchemyStore as WorkspaceAwareRegistryStore,
)
from mlflow.store.workspace.sqlalchemy_store import (
    SqlAlchemyStore as WorkspaceStore,
)
from mlflow.tracking._tracking_service import utils as tracking_utils
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


def test_validation_errors(engine):
    with pytest.raises(RuntimeError, match="does not exist"):
        move_resources(
            engine,
            source_workspace="nonexistent",
            target_workspace=DEFAULT_WORKSPACE_NAME,
            resource_type="experiments",
        )

    with pytest.raises(RuntimeError, match="does not exist"):
        move_resources(
            engine,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace="nonexistent",
            resource_type="experiments",
        )

    with pytest.raises(RuntimeError, match="must be different"):
        move_resources(
            engine,
            source_workspace=DEFAULT_WORKSPACE_NAME,
            target_workspace=DEFAULT_WORKSPACE_NAME,
            resource_type="experiments",
        )


def test_error_tag_on_unsupported_resource_type(workspace_store, engine):
    _create_workspace(workspace_store, "team-a")
    with pytest.raises(RuntimeError, match="does not support tag filtering"):
        move_resources(
            engine,
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
        source_workspace=DEFAULT_WORKSPACE_NAME,
        target_workspace="team-a",
        resource_type="experiments",
        names=["nonexistent"],
    )
    assert result == MoveResult(names=[], row_count=0)

    result = move_resources(
        engine,
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
