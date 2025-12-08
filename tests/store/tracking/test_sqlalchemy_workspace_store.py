import math
import time
import uuid
from pathlib import Path
from unittest import mock

import pytest

from mlflow.entities import (
    Dataset,
    DatasetInput,
    Experiment,
    ExperimentTag,
    InputTag,
    LoggedModelParameter,
    LoggedModelStatus,
    LoggedModelTag,
    Metric,
    Param,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import (
    SqlEntityAssociation,
    SqlExperiment,
    SqlTraceInfo,
    SqlTraceTag,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.tracing.utils import generate_request_id_v2
from mlflow.tracking._tracking_service import utils as tracking_utils
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.store.tracking.test_sqlalchemy_store import create_test_span


def _now_ms() -> int:
    return int(time.time() * 1000)


def _create_run(
    store: SqlAlchemyStore,
    workspace: str,
    experiment_name: str,
    run_name: str,
    user: str = "user",
):
    with WorkspaceContext(workspace):
        exp_id = store.create_experiment(experiment_name)
        run = store.create_run(
            exp_id,
            user_id=user,
            start_time=_now_ms(),
            tags=[],
            run_name=run_name,
        )
    return exp_id, run


@pytest.fixture
def workspace_tracking_store(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    backend_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = tracking_utils._get_sqlalchemy_store(backend_uri, artifact_dir.as_uri())
    store.tracking_uri = backend_uri
    store.artifact_root_uri = artifact_dir.as_uri()
    try:
        yield store
    finally:
        store._dispose_engine()


def test_sqlalchemy_store_returns_workspace_aware_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    backend_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = tracking_utils._get_sqlalchemy_store(backend_uri, artifact_dir.as_uri())
    try:
        assert isinstance(store, WorkspaceAwareSqlAlchemyStore)
        assert store.supports_workspaces() is True
    finally:
        store._dispose_engine()


def test_sqlalchemy_store_is_single_tenant_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    backend_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = tracking_utils._get_sqlalchemy_store(backend_uri, artifact_dir.as_uri())
    try:
        assert not isinstance(store, WorkspaceAwareSqlAlchemyStore)
        assert store.supports_workspaces() is False
    finally:
        store._dispose_engine()


def test_experiments_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-a"):
        exp_a_id = workspace_tracking_store.create_experiment("exp-in-a")
        duplicate_a_id = workspace_tracking_store.create_experiment("shared-name")

    with WorkspaceContext("team-b"):
        exp_b_id = workspace_tracking_store.create_experiment("exp-in-b")
        duplicate_b_id = workspace_tracking_store.create_experiment("shared-name")

    with WorkspaceContext("team-a"):
        exp_a = workspace_tracking_store.get_experiment(exp_a_id)
        assert exp_a.name == "exp-in-a"
        assert exp_a.workspace == "team-a"

        experiments = workspace_tracking_store.search_experiments(ViewType.ACTIVE_ONLY)
        assert {exp.name for exp in experiments} == {"exp-in-a", "shared-name"}

        assert workspace_tracking_store.get_experiment_by_name("exp-in-b") is None
        with pytest.raises(
            MlflowException, match=f"No Experiment with id={exp_b_id} exists"
        ) as excinfo:
            workspace_tracking_store.get_experiment(exp_b_id)
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

        duplicate_a = workspace_tracking_store.get_experiment(duplicate_a_id)
        assert duplicate_a.name == "shared-name"
        assert duplicate_a.workspace == "team-a"

    with WorkspaceContext("team-b"):
        experiments = workspace_tracking_store.search_experiments(ViewType.ACTIVE_ONLY)
        assert {exp.name for exp in experiments} == {"exp-in-b", "shared-name"}
        assert workspace_tracking_store.get_experiment_by_name("exp-in-a") is None
        duplicate_b = workspace_tracking_store.get_experiment(duplicate_b_id)
        assert duplicate_b.name == "shared-name"
        assert duplicate_b.workspace == "team-b"


def test_runs_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-a"):
        exp_a_id = workspace_tracking_store.create_experiment("exp-a")
        run_a = workspace_tracking_store.create_run(
            exp_a_id,
            user_id="alice",
            start_time=_now_ms(),
            tags=[],
            run_name="run-a",
        )

    with WorkspaceContext("team-b"):
        exp_b_id = workspace_tracking_store.create_experiment("exp-b")
        run_b = workspace_tracking_store.create_run(
            exp_b_id,
            user_id="bob",
            start_time=_now_ms(),
            tags=[],
            run_name="run-b",
        )

        with pytest.raises(
            MlflowException, match=f"No Experiment with id={exp_a_id} exists"
        ) as excinfo:
            workspace_tracking_store.create_run(
                exp_a_id,
                user_id="bob",
                start_time=_now_ms(),
                tags=[],
                run_name=None,
            )
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

        with pytest.raises(
            MlflowException, match=f"Run with id={run_a.info.run_id} not found"
        ) as excinfo:
            workspace_tracking_store.get_run(run_a.info.run_id)
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

        runs_b = workspace_tracking_store.search_runs([exp_b_id], None, ViewType.ALL)
        assert {run.info.run_id for run in runs_b} == {run_b.info.run_id}

        runs_from_other_workspace = workspace_tracking_store.search_runs(
            [exp_a_id], None, ViewType.ALL
        )
        assert runs_from_other_workspace == []

    with WorkspaceContext("team-a"):
        fetched = workspace_tracking_store.get_run(run_a.info.run_id)
        assert fetched.info.experiment_id == exp_a_id


def test_search_datasets_is_workspace_scoped(workspace_tracking_store):
    exp_a_id, run_a = _create_run(workspace_tracking_store, "team-a", "exp-a", "run-a")
    exp_b_id, run_b = _create_run(workspace_tracking_store, "team-b", "exp-b", "run-b")

    dataset_a = Dataset(
        name="dataset-a",
        digest="digest-a",
        source_type="delta",
        source="source-a",
    )
    dataset_b = Dataset(
        name="dataset-b",
        digest="digest-b",
        source_type="delta",
        source="source-b",
    )

    with WorkspaceContext("team-a"):
        workspace_tracking_store.log_inputs(
            run_a.info.run_id,
            [DatasetInput(dataset_a, [InputTag(MLFLOW_DATASET_CONTEXT, "train")])],
        )

    with WorkspaceContext("team-b"):
        workspace_tracking_store.log_inputs(
            run_b.info.run_id,
            [DatasetInput(dataset_b, [InputTag(MLFLOW_DATASET_CONTEXT, "train")])],
        )

        summaries = workspace_tracking_store._search_datasets([exp_b_id, exp_a_id])
        assert {
            (summary.experiment_id, summary.name, summary.digest, summary.context)
            for summary in summaries
        } == {(str(exp_b_id), dataset_b.name, dataset_b.digest, "train")}

        assert workspace_tracking_store._search_datasets([exp_a_id]) == []

    with WorkspaceContext("team-a"):
        summaries = workspace_tracking_store._search_datasets([exp_a_id, exp_b_id])
        assert {
            (summary.experiment_id, summary.name, summary.digest, summary.context)
            for summary in summaries
        } == {(str(exp_a_id), dataset_a.name, dataset_a.digest, "train")}

        assert workspace_tracking_store._search_datasets([exp_b_id]) == []


def test_search_datasets_public_api_is_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-a"):
        exp_a_id = workspace_tracking_store.create_experiment("search-exp-a")
        dataset_a = workspace_tracking_store.create_dataset(
            name="dataset-a", experiment_ids=[exp_a_id]
        )
        workspace_tracking_store.upsert_dataset_records(
            dataset_a.dataset_id,
            [{"inputs": {"x": 1}, "outputs": {"y": "a"}}],
        )

    with WorkspaceContext("team-b"):
        exp_b_id = workspace_tracking_store.create_experiment("search-exp-b")
        dataset_b = workspace_tracking_store.create_dataset(
            name="dataset-b", experiment_ids=[exp_b_id]
        )
        workspace_tracking_store.upsert_dataset_records(
            dataset_b.dataset_id,
            [{"inputs": {"x": 2}, "outputs": {"y": "b"}}],
        )

    with WorkspaceContext("team-a"):
        results = workspace_tracking_store.search_datasets()
        assert {d.name for d in results} == {"dataset-a"}

        results = workspace_tracking_store.search_datasets(experiment_ids=[exp_a_id, exp_b_id])
        assert {d.name for d in results} == {"dataset-a"}

        records, _ = workspace_tracking_store._load_dataset_records(dataset_a.dataset_id)
        assert len(records) == 1
        assert records[0].inputs == {"x": 1}
        assert records[0].outputs == {"y": "a"}

    with WorkspaceContext("team-b"):
        results = workspace_tracking_store.search_datasets()
        assert {d.name for d in results} == {"dataset-b"}

        results = workspace_tracking_store.search_datasets(experiment_ids=[exp_a_id])
        assert results == []

        records, _ = workspace_tracking_store._load_dataset_records(dataset_b.dataset_id)
        assert len(records) == 1
        assert records[0].inputs == {"x": 2}
        assert records[0].outputs == {"y": "b"}


def test_entity_associations_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-a"):
        exp_a_id = workspace_tracking_store.create_experiment("assoc-exp-a")
        dataset = workspace_tracking_store.create_dataset(
            name="dataset-a",
            experiment_ids=[exp_a_id],
        )

    with WorkspaceContext("team-a"):
        forward = workspace_tracking_store.search_entities_by_source(
            source_ids=dataset.dataset_id,
            source_type=EntityAssociationType.EVALUATION_DATASET,
            destination_type=EntityAssociationType.EXPERIMENT,
        )
        assert forward.to_list() == [exp_a_id]

    with WorkspaceContext("team-b"):
        forward = workspace_tracking_store.search_entities_by_source(
            source_ids=dataset.dataset_id,
            source_type=EntityAssociationType.EVALUATION_DATASET,
            destination_type=EntityAssociationType.EXPERIMENT,
        )
        assert forward.to_list() == []

        reverse = workspace_tracking_store.search_entities_by_destination(
            destination_ids=exp_a_id,
            destination_type=EntityAssociationType.EXPERIMENT,
            source_type=EntityAssociationType.EVALUATION_DATASET,
        )
        assert reverse.to_list() == []


def test_artifact_locations_are_scoped_to_workspace(workspace_tracking_store):
    with WorkspaceContext("team-alpha"):
        exp_id = workspace_tracking_store.create_experiment("alpha-exp")
        experiment = workspace_tracking_store.get_experiment(exp_id)
        assert "/workspaces/team-alpha/" in experiment.artifact_location

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        default_exp_id = workspace_tracking_store.create_experiment("default-exp")
        default_experiment = workspace_tracking_store.get_experiment(default_exp_id)
        assert f"/workspaces/{DEFAULT_WORKSPACE_NAME}/" in default_experiment.artifact_location


def test_serving_artifacts_auto_scopes_workspace_paths(workspace_tracking_store, monkeypatch):
    monkeypatch.setenv("_MLFLOW_SERVER_SERVE_ARTIFACTS", "true")
    workspace_tracking_store.artifact_root_uri = "mlflow-artifacts:/artifacts"

    class RaisingProvider:
        def resolve_artifact_root(self, *_args, **_kwargs):
            raise AssertionError(
                "Workspace provider should not be consulted when serving artifacts"
            )

    monkeypatch.setattr(
        WorkspaceAwareSqlAlchemyStore,
        "_get_workspace_provider_instance",
        lambda self: RaisingProvider(),
    )

    with WorkspaceContext("team-prefix"):
        exp_id = workspace_tracking_store.create_experiment("auto-scoped")
        experiment = workspace_tracking_store.get_experiment(exp_id)
        assert f"/workspaces/team-prefix/{exp_id}" in experiment.artifact_location


def test_serving_artifacts_allows_pre_scoped_roots(workspace_tracking_store, monkeypatch):
    monkeypatch.delenv("_MLFLOW_SERVER_SERVE_ARTIFACTS", raising=False)
    workspace_tracking_store.artifact_root_uri = "mlflow-artifacts:/artifacts"

    class PrefixedProvider:
        def __init__(self, store):
            self.store = store

        def resolve_artifact_root(self, artifact_root, workspace_name):
            scoped = append_to_uri_path(artifact_root, f"workspaces/{workspace_name}")
            return scoped, False

    monkeypatch.setattr(
        WorkspaceAwareSqlAlchemyStore,
        "_get_workspace_provider_instance",
        lambda self: PrefixedProvider(self),
    )

    with WorkspaceContext("team-ready"):
        exp_id = workspace_tracking_store.create_experiment("with-prefix")
        experiment = workspace_tracking_store.get_experiment(exp_id)
    assert "/workspaces/team-ready/" in experiment.artifact_location


def test_default_workspace_experiment_uses_zero_id(workspace_tracking_store):
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        default_experiment = workspace_tracking_store.get_experiment_by_name(
            Experiment.DEFAULT_EXPERIMENT_NAME
        )

    assert default_experiment is not None
    assert default_experiment.experiment_id == SqlAlchemyStore.DEFAULT_EXPERIMENT_ID


def test_default_workspace_experiment_allows_single_tenant_fallback(tmp_path, monkeypatch):
    backend_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    workspace_store = SqlAlchemyStore(backend_uri, artifact_dir.as_uri())
    try:
        with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
            default_ws_experiment = workspace_store.get_experiment_by_name(
                Experiment.DEFAULT_EXPERIMENT_NAME
            )
        assert default_ws_experiment is not None
        assert default_ws_experiment.experiment_id == SqlAlchemyStore.DEFAULT_EXPERIMENT_ID
    finally:
        workspace_store._dispose_engine()

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    single_tenant_store = SqlAlchemyStore(backend_uri, artifact_dir.as_uri())
    try:
        fallback_experiment = single_tenant_store.get_experiment(
            SqlAlchemyStore.DEFAULT_EXPERIMENT_ID
        )
        assert fallback_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
    finally:
        single_tenant_store._dispose_engine()


def test_custom_artifact_location_rejected_in_workspace(workspace_tracking_store):
    with WorkspaceContext("team-delta"):
        with pytest.raises(
            MlflowException,
            match="artifact_location cannot be specified when workspaces are enabled",
        ) as excinfo:
            workspace_tracking_store.create_experiment(
                "delta-exp", artifact_location="file:///tmp/custom"
            )
        assert excinfo.value.error_code == "INVALID_PARAMETER_VALUE"


def test_experiment_lifecycle_operations_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-a"):
        exp_a_id = workspace_tracking_store.create_experiment("lifecycle-exp")

    with WorkspaceContext("team-b"):
        exp_b_id = workspace_tracking_store.create_experiment("other-exp")

    with WorkspaceContext("team-a"):
        workspace_tracking_store.rename_experiment(exp_a_id, "renamed-exp")
        experiment = workspace_tracking_store.get_experiment(exp_a_id)
        assert experiment.name == "renamed-exp"

    with WorkspaceContext("team-b"):
        with pytest.raises(MlflowException, match=f"No Experiment with id={exp_a_id} exists"):
            workspace_tracking_store.rename_experiment(exp_a_id, "fail")
        with pytest.raises(MlflowException, match=f"No Experiment with id={exp_a_id} exists"):
            workspace_tracking_store.delete_experiment(exp_a_id)

    with WorkspaceContext("team-a"):
        workspace_tracking_store.delete_experiment(exp_a_id)
        deleted = workspace_tracking_store.get_experiment(exp_a_id)
        assert deleted.lifecycle_stage == LifecycleStage.DELETED

    with WorkspaceContext("team-b"):
        with pytest.raises(MlflowException, match=f"No Experiment with id={exp_a_id} exists"):
            workspace_tracking_store.restore_experiment(exp_a_id)

    with WorkspaceContext("team-a"):
        workspace_tracking_store.restore_experiment(exp_a_id)
        restored = workspace_tracking_store.get_experiment(exp_a_id)
        assert restored.lifecycle_stage == LifecycleStage.ACTIVE

    with WorkspaceContext("team-b"):
        assert workspace_tracking_store.get_experiment(exp_b_id).name == "other-exp"


def test_experiment_tags_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-a"):
        exp_a_id = workspace_tracking_store.create_experiment("tagged-exp")
        workspace_tracking_store.set_experiment_tag(exp_a_id, ExperimentTag("owner", "team-a"))

    with WorkspaceContext("team-b"):
        exp_b_id = workspace_tracking_store.create_experiment("other-exp")
        with pytest.raises(MlflowException, match=f"No Experiment with id={exp_a_id} exists"):
            workspace_tracking_store.set_experiment_tag(exp_a_id, ExperimentTag("owner", "team-b"))
        with pytest.raises(MlflowException, match=f"No Experiment with id={exp_a_id} exists"):
            workspace_tracking_store.delete_experiment_tag(exp_a_id, "owner")

    with WorkspaceContext("team-a"):
        workspace_tracking_store.delete_experiment_tag(exp_a_id, "owner")
        experiment = workspace_tracking_store.get_experiment(exp_a_id)
        assert "owner" not in experiment.tags

    with WorkspaceContext("team-b"):
        workspace_tracking_store.set_experiment_tag(exp_b_id, ExperimentTag("owner", "team-b"))
        experiment = workspace_tracking_store.get_experiment(exp_b_id)
        assert experiment.tags["owner"] == "team-b"


def test_run_data_logging_enforces_workspaces(workspace_tracking_store):
    exp_a_id, run_a = _create_run(
        workspace_tracking_store, "team-a", "data-exp-a", "run-a", user="alice"
    )

    with WorkspaceContext("team-a"):
        workspace_tracking_store.log_param(run_a.info.run_id, Param("p", "1"))
        workspace_tracking_store.log_metric(run_a.info.run_id, Metric("m", 2.0, _now_ms(), 0))
        workspace_tracking_store.set_tag(run_a.info.run_id, RunTag("t", "team-a"))
        workspace_tracking_store.log_batch(
            run_a.info.run_id,
            metrics=[Metric("m2", 3.0, _now_ms(), 0)],
            params=[Param("p2", "v")],
            tags=[RunTag("t2", "v2")],
        )
        run = workspace_tracking_store.get_run(run_a.info.run_id)
        assert run.data.params["p"] == "1"
        assert run.data.metrics["m"] == 2.0
        assert run.data.tags["t"] == "team-a"

    with WorkspaceContext("team-b"):
        exp_b_id, run_b = _create_run(
            workspace_tracking_store, "team-b", "data-exp-b", "run-b", user="bob"
        )
        for call in (
            lambda: workspace_tracking_store.log_param(run_a.info.run_id, Param("cross", "fail")),
            lambda: workspace_tracking_store.log_metric(
                run_a.info.run_id, Metric("cross", 1.0, _now_ms(), 0)
            ),
            lambda: workspace_tracking_store.set_tag(run_a.info.run_id, RunTag("cross", "fail")),
            lambda: workspace_tracking_store.delete_tag(run_a.info.run_id, "t"),
            lambda: workspace_tracking_store.log_batch(
                run_a.info.run_id,
                metrics=[Metric("cross", 1.0, _now_ms(), 0)],
                params=[Param("cross", "1")],
                tags=[RunTag("cross", "1")],
            ),
        ):
            with pytest.raises(
                MlflowException, match=f"Run with id={run_a.info.run_id} not found"
            ) as excinfo:
                call()
            assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

        workspace_tracking_store.log_param(run_b.info.run_id, Param("pb", "b"))
        assert workspace_tracking_store.get_run(run_b.info.run_id).data.params["pb"] == "b"

    with WorkspaceContext("team-a"):
        run = workspace_tracking_store.get_run(run_a.info.run_id)
        assert "cross" not in run.data.params
        assert run.data.tags["t2"] == "v2"


def test_run_lifecycle_operations_workspace_isolation(workspace_tracking_store):
    _, run_a = _create_run(
        workspace_tracking_store, "team-a", "lifecycle-exp", "run-a", user="alice"
    )

    with WorkspaceContext("team-b"):
        with pytest.raises(MlflowException, match=f"Run with id={run_a.info.run_id} not found"):
            workspace_tracking_store.delete_run(run_a.info.run_id)

    with WorkspaceContext("team-a"):
        workspace_tracking_store.delete_run(run_a.info.run_id)
        deleted = workspace_tracking_store.get_run(run_a.info.run_id)
        assert deleted.info.lifecycle_stage == LifecycleStage.DELETED

    with WorkspaceContext("team-b"):
        for call in (
            lambda: workspace_tracking_store.restore_run(run_a.info.run_id),
            lambda: workspace_tracking_store.update_run_info(
                run_a.info.run_id, RunStatus.FAILED, end_time=_now_ms(), run_name=None
            ),
        ):
            with pytest.raises(MlflowException, match=f"Run with id={run_a.info.run_id} not found"):
                call()

    with WorkspaceContext("team-a"):
        workspace_tracking_store.restore_run(run_a.info.run_id)
        restored = workspace_tracking_store.get_run(run_a.info.run_id)
        assert restored.info.lifecycle_stage == LifecycleStage.ACTIVE

        updated = workspace_tracking_store.update_run_info(
            run_a.info.run_id, RunStatus.FINISHED, end_time=_now_ms(), run_name=None
        )
        assert updated.status == RunStatus.to_string(RunStatus.FINISHED)


def test_search_and_history_calls_are_workspace_scoped(workspace_tracking_store):
    exp_a_id, run_a = _create_run(
        workspace_tracking_store, "team-a", "search-exp-a", "run-a", user="alice"
    )
    exp_b_id, run_b = _create_run(
        workspace_tracking_store, "team-b", "search-exp-b", "run-b", user="bob"
    )

    with WorkspaceContext("team-a"):
        workspace_tracking_store.log_metric(run_a.info.run_id, Metric("metric", 1.0, _now_ms(), 0))
        runs = workspace_tracking_store.search_runs([exp_a_id, exp_b_id], None, ViewType.ALL)
        assert {r.info.run_id for r in runs} == {run_a.info.run_id}

    with WorkspaceContext("team-b"):
        runs = workspace_tracking_store.search_runs([exp_a_id, exp_b_id], None, ViewType.ALL)
        assert {r.info.run_id for r in runs} == {run_b.info.run_id}

        with pytest.raises(MlflowException, match=f"Run with id={run_a.info.run_id} not found"):
            workspace_tracking_store.get_metric_history(run_a.info.run_id, "metric")

    with WorkspaceContext("team-a"):
        history = workspace_tracking_store.get_metric_history(run_a.info.run_id, "metric")
        assert [m.value for m in history] == [1.0]
        with workspace_tracking_store.ManagedSessionMaker() as session:
            infos = workspace_tracking_store._list_run_infos(session, exp_a_id)
            assert len(infos) == 1

    with WorkspaceContext("team-b"):
        with workspace_tracking_store.ManagedSessionMaker() as session:
            infos = workspace_tracking_store._list_run_infos(session, exp_a_id)
            assert infos == []


def test_run_artifact_uris_are_workspace_scoped(workspace_tracking_store):
    _, run_a = _create_run(
        workspace_tracking_store, "team-a", "artifact-exp", "run-a", user="alice"
    )

    with WorkspaceContext("team-a"):
        run = workspace_tracking_store.get_run(run_a.info.run_id)
        artifact_uri = Path(run.info.artifact_uri.replace("file://", ""))
        parts = artifact_uri.parts
        assert "workspaces" in parts
        workspace_index = parts.index("workspaces")
        assert parts[workspace_index + 1] == "team-a"

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        _, default_run = _create_run(
            workspace_tracking_store,
            DEFAULT_WORKSPACE_NAME,
            "default-artifact-exp",
            "run-default",
            user="carol",
        )
        default_artifact_uri = Path(
            workspace_tracking_store.get_run(default_run.info.run_id).info.artifact_uri.replace(
                "file://", ""
            )
        )
        assert "workspaces" in default_artifact_uri.parts
        workspace_index = default_artifact_uri.parts.index("workspaces")
        assert default_artifact_uri.parts[workspace_index + 1] == DEFAULT_WORKSPACE_NAME

    with WorkspaceContext("team-b"):
        with pytest.raises(MlflowException, match=f"Run with id={run_a.info.run_id} not found"):
            workspace_tracking_store.get_run(run_a.info.run_id)


def test_artifact_operations_enforce_workspace_isolation(workspace_tracking_store, tmp_path):
    client = TrackingServiceClient(workspace_tracking_store.tracking_uri)
    _, run_a = _create_run(
        workspace_tracking_store, "team-a", "artifact-exp-client", "run-a", user="alice"
    )
    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("hello")

    with WorkspaceContext("team-a"):
        client.log_artifact(run_a.info.run_id, str(artifact_file))
        artifacts = client.list_artifacts(run_a.info.run_id)
        assert any(info.path == "artifact.txt" for info in artifacts)

        download_dir = tmp_path / "downloaded"
        download_dir.mkdir()
        downloaded_path = client.download_artifacts(
            run_a.info.run_id, "artifact.txt", dst_path=str(download_dir)
        )
        assert Path(downloaded_path).read_text() == "hello"

    with WorkspaceContext("team-b"):
        other_client = TrackingServiceClient(workspace_tracking_store.tracking_uri)
        other_download_dir = tmp_path / "other"
        other_download_dir.mkdir()
        for call in (
            lambda: other_client.log_artifact(run_a.info.run_id, str(artifact_file)),
            lambda: other_client.list_artifacts(run_a.info.run_id),
            lambda: other_client.download_artifacts(
                run_a.info.run_id, "artifact.txt", dst_path=str(other_download_dir)
            ),
        ):
            tracking_utils._artifact_repos_cache.clear()
            with pytest.raises(
                MlflowException, match=f"Run with id={run_a.info.run_id} not found"
            ) as excinfo:
                call()
            assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_get_trace_is_workspace_scoped(workspace_tracking_store):
    trace_id = f"tr-{uuid.uuid4().hex}"
    span = create_test_span(trace_id=trace_id)

    with WorkspaceContext("team-a"):
        exp_id = workspace_tracking_store.create_experiment("trace-exp-a")
        workspace_tracking_store.log_spans(exp_id, [span])
        trace = workspace_tracking_store.get_trace(trace_id)
        assert trace.info.trace_id == trace_id

    with WorkspaceContext("team-b"):
        with pytest.raises(
            MlflowException, match=f"Trace with ID {trace_id} is not found."
        ) as excinfo:
            workspace_tracking_store.get_trace(trace_id)
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_log_spans_update_is_workspace_scoped(workspace_tracking_store):
    trace_id = f"tr-{uuid.uuid4().hex}"
    initial_span = create_test_span(
        trace_id=trace_id,
        start_ns=2_000_000_000,
        end_ns=3_000_000_000,
    )
    earlier_span = create_test_span(
        trace_id=trace_id,
        span_id=222,
        start_ns=1_000_000_000,
        end_ns=4_000_000_000,
    )

    with WorkspaceContext("team-a"):
        exp_id = workspace_tracking_store.create_experiment("trace-exp-workspace-guard")
        workspace_tracking_store.log_spans(exp_id, [initial_span])
        original_trace = workspace_tracking_store.get_trace(trace_id)

        call_state = {"count": 0}

        def workspace_side_effect(*_args, **_kwargs):
            call_state["count"] += 1
            return "team-a" if call_state["count"] == 1 else "team-b"

        with mock.patch.object(
            WorkspaceAwareSqlAlchemyStore,
            "_get_active_workspace",
            side_effect=workspace_side_effect,
        ):
            workspace_tracking_store.log_spans(exp_id, [earlier_span])

        updated_trace = workspace_tracking_store.get_trace(trace_id)
        assert updated_trace.info.request_time == original_trace.info.request_time
        assert updated_trace.info.execution_duration == original_trace.info.execution_duration
        assert len(updated_trace.data.spans) == 2


def test_workspace_startup_rejects_root_ending_with_workspaces(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    backend_uri = f"sqlite:///{tmp_path / 'suffix.db'}"
    bad_root = tmp_path / "base" / "workspaces"
    bad_root.mkdir(parents=True)

    with pytest.raises(
        MlflowException,
        match="ends with the reserved 'workspaces' segment",
    ) as excinfo:
        tracking_utils._get_sqlalchemy_store(backend_uri, bad_root.as_uri())
    assert excinfo.value.error_code == "INVALID_STATE"
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)


def test_workspace_startup_rejects_root_already_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    backend_uri = f"sqlite:///{tmp_path / 'scoped.db'}"
    bad_root = tmp_path / "base" / "workspaces" / "team"
    bad_root.mkdir(parents=True)

    with pytest.raises(
        MlflowException,
        match="is already scoped under the reserved 'workspaces/<name>' prefix",
    ) as excinfo:
        tracking_utils._get_sqlalchemy_store(backend_uri, bad_root.as_uri())
    assert excinfo.value.error_code == "INVALID_STATE"
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)


def test_workspace_startup_detects_existing_reserved_artifact(tmp_path, monkeypatch):
    backend_uri = f"sqlite:///{tmp_path / 'conflict.db'}"
    base_root = tmp_path / "base"
    base_root.mkdir()

    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    legacy_store = tracking_utils._get_sqlalchemy_store(backend_uri, base_root.as_uri())
    legacy_store.tracking_uri = backend_uri
    legacy_store.artifact_root_uri = base_root.as_uri()
    reserved_location = append_to_uri_path(base_root.as_uri(), "workspaces/legacy/1")
    legacy_store.create_experiment("legacy-exp", artifact_location=reserved_location)
    legacy_store._dispose_engine()

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    with pytest.raises(
        MlflowException,
        match="existing experiment artifact location",
    ) as excinfo:
        tracking_utils._get_sqlalchemy_store(backend_uri, base_root.as_uri())
    assert excinfo.value.error_code == "INVALID_STATE"
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)


def test_workspace_startup_ignores_default_experiment_reserved_location(tmp_path, monkeypatch):
    backend_uri = f"sqlite:///{tmp_path / 'default_conflict.db'}"
    base_root = tmp_path / "base"
    base_root.mkdir()

    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    legacy_store = tracking_utils._get_sqlalchemy_store(backend_uri, base_root.as_uri())
    legacy_store.tracking_uri = backend_uri
    legacy_store.artifact_root_uri = base_root.as_uri()

    with legacy_store.ManagedSessionMaker() as session:
        default_exp = (
            session.query(SqlExperiment)
            .filter(SqlExperiment.experiment_id == SqlAlchemyStore.DEFAULT_EXPERIMENT_ID)
            .one()
        )
        default_exp.artifact_location = append_to_uri_path(
            base_root.as_uri(), "workspaces/default/0"
        )
        session.flush()
    legacy_store._dispose_engine()

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    workspace_store = tracking_utils._get_sqlalchemy_store(backend_uri, base_root.as_uri())
    workspace_store._dispose_engine()
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)


def test_single_tenant_startup_rejects_non_default_workspace_experiments(tmp_path, monkeypatch):
    backend_uri = f"sqlite:///{tmp_path / 'multi_tenant.db'}"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    workspace_store = tracking_utils._get_sqlalchemy_store(backend_uri, artifact_root.as_uri())

    with WorkspaceContext("team-startup"):
        workspace_store.create_experiment("team-exp")

    workspace_store._dispose_engine()
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    with pytest.raises(
        MlflowException,
        match="Cannot disable workspaces because experiments exist outside the default workspace",
    ) as excinfo:
        SqlAlchemyStore(backend_uri, artifact_root.as_uri())

    assert excinfo.value.error_code == "INVALID_STATE"
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)


def test_metric_bulk_operations_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-metrics-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-metrics-a")
        run_a = workspace_tracking_store.create_run(exp_a, "alice", _now_ms(), [], "run-a")
        workspace_tracking_store.log_metric(
            run_a.info.run_id, Metric("secret_metric", 42.0, _now_ms(), 0)
        )

    with WorkspaceContext("team-metrics-b"):
        exp_b = workspace_tracking_store.create_experiment("exp-metrics-b")
        run_b = workspace_tracking_store.create_run(exp_b, "bob", _now_ms(), [], "run-b")
        workspace_tracking_store.log_metric(
            run_b.info.run_id, Metric("other_metric", 10.0, _now_ms(), 0)
        )

        result = workspace_tracking_store.get_metric_history_bulk(
            [run_a.info.run_id], "secret_metric", 100
        )
        assert result == []

        with pytest.raises(MlflowException, match="Run with id=.* not found"):
            workspace_tracking_store.get_max_step_for_metric(run_a.info.run_id, "secret_metric")

        with pytest.raises(MlflowException, match="Run with id=.* not found"):
            workspace_tracking_store.get_metric_history_bulk_interval_from_steps(
                run_a.info.run_id, "secret_metric", [0], 100
            )


def test_logged_model_operations_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-model-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-model-a")
        run_a = workspace_tracking_store.create_run(exp_a, "alice", _now_ms(), [], "run-a")
        model_a = workspace_tracking_store.create_logged_model(exp_a, "model-a", run_a.info.run_id)

    with WorkspaceContext("team-model-b"):
        workspace_tracking_store.create_experiment("exp-model-b")

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.get_logged_model(model_a.model_id)

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.delete_logged_model(model_a.model_id)

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.finalize_logged_model(
                model_a.model_id, LoggedModelStatus.READY
            )

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.log_logged_model_params(
                model_a.model_id, [LoggedModelParameter("key", "value")]
            )

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.set_logged_model_tags(
                model_a.model_id, [LoggedModelTag("key", "value")]
            )

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.delete_logged_model_tag(model_a.model_id, "key")


def test_trace_tag_operations_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-trace-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-trace-a")
        trace_id_a = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id_a,
                    experiment_id=int(exp_a),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=0,
                    status="OK",
                )
            )

    with WorkspaceContext("team-trace-b"):
        workspace_tracking_store.create_experiment("exp-trace-b")

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.set_trace_tag(trace_id_a, "key", "value")

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.delete_trace_tag(trace_id_a, "key")


def test_search_traces_is_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-search-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-search-a")
        trace_id_a = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id_a,
                    experiment_id=int(exp_a),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=0,
                    status="OK",
                )
            )

    with WorkspaceContext("team-search-b"):
        exp_b = workspace_tracking_store.create_experiment("exp-search-b")
        trace_id_b = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id_b,
                    experiment_id=int(exp_b),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=0,
                    status="OK",
                )
            )

        # Cross-workspace search returns nothing
        results, _ = workspace_tracking_store.search_traces(locations=[exp_a])
        assert results == []

        # Same-workspace search works
        results, _ = workspace_tracking_store.search_traces(locations=[exp_b])
        assert len(results) == 1
        assert results[0].trace_id == trace_id_b


def test_link_traces_to_run_is_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-link-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-link-a")
        run_a = workspace_tracking_store.create_run(exp_a, "alice", _now_ms(), [], "run-a")
        trace_id_a = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id_a,
                    experiment_id=int(exp_a),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=0,
                    status="OK",
                )
            )

    with WorkspaceContext("team-link-b"):
        exp_b = workspace_tracking_store.create_experiment("exp-link-b")
        run_b = workspace_tracking_store.create_run(exp_b, "bob", _now_ms(), [], "run-b")
        trace_id_b = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id_b,
                    experiment_id=int(exp_b),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=0,
                    status="OK",
                )
            )

        # Cross-workspace trace is silently filtered out
        workspace_tracking_store.link_traces_to_run([trace_id_a], run_b.info.run_id)
        with workspace_tracking_store.ManagedSessionMaker() as session:
            count = (
                session.query(SqlEntityAssociation)
                .filter(SqlEntityAssociation.source_id == trace_id_a)
                .count()
            )
            assert count == 0

        # Cross-workspace run raises error
        with pytest.raises(MlflowException, match="Run with id=.* not found"):
            workspace_tracking_store.link_traces_to_run([trace_id_b], run_a.info.run_id)

        # Same-workspace link works
        workspace_tracking_store.link_traces_to_run([trace_id_b], run_b.info.run_id)
        with workspace_tracking_store.ManagedSessionMaker() as session:
            count = (
                session.query(SqlEntityAssociation)
                .filter(SqlEntityAssociation.source_id == trace_id_b)
                .count()
            )
            assert count == 1


def test_assessment_operations_are_workspace_scoped(workspace_tracking_store):
    from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType, Feedback

    with WorkspaceContext("team-assessment-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-assessment-a")
        trace_id_a = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id_a,
                    experiment_id=int(exp_a),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=100,
                    status="OK",
                )
            )

        feedback_a = Feedback(
            trace_id=trace_id_a,
            name="quality",
            value=True,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="user@example.com"
            ),
        )
        created_assessment_a = workspace_tracking_store.create_assessment(feedback_a)
        assessment_id_a = created_assessment_a.assessment_id

        retrieved = workspace_tracking_store.get_assessment(trace_id_a, assessment_id_a)
        assert retrieved.assessment_id == assessment_id_a

    with WorkspaceContext("team-assessment-b"):
        exp_b = workspace_tracking_store.create_experiment("exp-assessment-b")
        trace_id_b = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id_b,
                    experiment_id=int(exp_b),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=100,
                    status="OK",
                )
            )

        with pytest.raises(MlflowException, match=r"Trace with ID .* not found"):
            workspace_tracking_store.get_assessment(trace_id_a, assessment_id_a)

        with pytest.raises(MlflowException, match=r"Trace with ID .* not found"):
            workspace_tracking_store.update_assessment(
                trace_id_a,
                assessment_id_a,
                name="updated_quality",
            )

        with pytest.raises(MlflowException, match=r"Trace with ID .* not found"):
            workspace_tracking_store.delete_assessment(trace_id_a, assessment_id_a)

    with WorkspaceContext("team-assessment-a"):
        retrieved = workspace_tracking_store.get_assessment(trace_id_a, assessment_id_a)
        assert retrieved.name == "quality"
        assert retrieved.value is True


def test_create_assessment_validates_trace_workspace(workspace_tracking_store):
    from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType, Feedback

    with WorkspaceContext("team-create-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-create-a")
        trace_id_a = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id_a,
                    experiment_id=int(exp_a),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=100,
                    status="OK",
                )
            )

    with WorkspaceContext("team-create-b"):
        workspace_tracking_store.create_experiment("exp-create-b")

        feedback = Feedback(
            trace_id=trace_id_a,
            name="cross_workspace_attempt",
            value=False,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="attacker@example.com"
            ),
        )

        with pytest.raises(MlflowException, match=r"Trace with ID .* not found"):
            workspace_tracking_store.create_assessment(feedback)


def test_calculate_trace_filter_correlation_filters_experiment_ids(workspace_tracking_store):
    with WorkspaceContext("team-corr-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-corr-a")
        for _ in range(5):
            trace_id = generate_request_id_v2()
            with workspace_tracking_store.ManagedSessionMaker() as session:
                session.add(
                    SqlTraceInfo(
                        request_id=trace_id,
                        experiment_id=int(exp_a),
                        timestamp_ms=_now_ms(),
                        execution_time_ms=100,
                        status="OK",
                    )
                )
                session.merge(
                    SqlTraceTag(
                        request_id=trace_id,
                        key="test_tag",
                        value="value_a",
                    )
                )

    with WorkspaceContext("team-corr-b"):
        exp_b = workspace_tracking_store.create_experiment("exp-corr-b")
        for _ in range(3):
            trace_id = generate_request_id_v2()
            with workspace_tracking_store.ManagedSessionMaker() as session:
                session.add(
                    SqlTraceInfo(
                        request_id=trace_id,
                        experiment_id=int(exp_b),
                        timestamp_ms=_now_ms(),
                        execution_time_ms=100,
                        status="OK",
                    )
                )
                session.merge(
                    SqlTraceTag(
                        request_id=trace_id,
                        key="test_tag",
                        value="value_b",
                    )
                )

        result = workspace_tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_a],
            filter_string1='tags.test_tag = "value_a"',
            filter_string2='tags.test_tag = "value_a"',
        )

        assert result.total_count == 0
        assert result.filter1_count == 0
        assert result.filter2_count == 0
        assert result.joint_count == 0

        result = workspace_tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_b],
            filter_string1='tags.test_tag = "value_b"',
            filter_string2='tags.test_tag = "value_b"',
        )

        assert result.total_count == 3
        assert result.filter1_count == 3


def test_calculate_trace_filter_correlation_cross_workspace_ids_filtered(workspace_tracking_store):
    with WorkspaceContext("team-xcorr-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-xcorr-a")
        trace_id = generate_request_id_v2()
        with workspace_tracking_store.ManagedSessionMaker() as session:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id,
                    experiment_id=int(exp_a),
                    timestamp_ms=_now_ms(),
                    execution_time_ms=100,
                    status="OK",
                )
            )

    with WorkspaceContext("team-xcorr-b"):
        workspace_tracking_store.create_experiment("exp-xcorr-b")

        result = workspace_tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_a],
            filter_string1="status = 'OK'",
            filter_string2="status = 'OK'",
        )

        assert result.total_count == 0
        assert result.filter1_count == 0
        assert result.filter2_count == 0
        assert result.joint_count == 0
        assert math.isnan(result.npmi)


def test_dataset_tag_operations_are_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-tag-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-tag-a")
        dataset_a = workspace_tracking_store.create_dataset(
            name="dataset-a",
            experiment_ids=[exp_a],
            tags={"initial": "value"},
        )

        # Same workspace tag operations work
        workspace_tracking_store.set_dataset_tags(dataset_a.dataset_id, {"key1": "value1"})
        updated = workspace_tracking_store.get_dataset(dataset_a.dataset_id)
        assert updated.tags["key1"] == "value1"

        workspace_tracking_store.delete_dataset_tag(dataset_a.dataset_id, "initial")
        updated = workspace_tracking_store.get_dataset(dataset_a.dataset_id)
        assert "initial" not in updated.tags

    with WorkspaceContext("team-tag-b"):
        workspace_tracking_store.create_experiment("exp-tag-b")

        # Cross-workspace set_dataset_tags fails
        with pytest.raises(
            MlflowException,
            match=f"Could not find evaluation dataset with ID {dataset_a.dataset_id}",
        ) as excinfo:
            workspace_tracking_store.set_dataset_tags(dataset_a.dataset_id, {"cross": "fail"})
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

        # Cross-workspace delete_dataset_tag is a no-op (idempotent, doesn't affect other workspace)
        workspace_tracking_store.delete_dataset_tag(dataset_a.dataset_id, "key1")

    # Verify tags are unchanged after cross-workspace attempts
    with WorkspaceContext("team-tag-a"):
        dataset = workspace_tracking_store.get_dataset(dataset_a.dataset_id)
        assert dataset.tags["key1"] == "value1"
        assert "cross" not in dataset.tags


def test_upsert_dataset_records_is_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-records-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-records-a")
        dataset_a = workspace_tracking_store.create_dataset(
            name="dataset-records-a",
            experiment_ids=[exp_a],
        )

        # Same workspace upsert works
        result = workspace_tracking_store.upsert_dataset_records(
            dataset_a.dataset_id,
            [{"inputs": {"x": 1}, "outputs": {"y": 2}}],
        )
        assert result["inserted"] == 1
        assert result["updated"] == 0

    with WorkspaceContext("team-records-b"):
        workspace_tracking_store.create_experiment("exp-records-b")

        # Cross-workspace upsert fails
        with pytest.raises(
            MlflowException,
            match=f"Dataset '{dataset_a.dataset_id}' not found",
        ) as excinfo:
            workspace_tracking_store.upsert_dataset_records(
                dataset_a.dataset_id,
                [{"inputs": {"x": 2}, "outputs": {"y": 3}}],
            )
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    # Verify records are unchanged after cross-workspace attempt
    with WorkspaceContext("team-records-a"):
        records, _ = workspace_tracking_store._load_dataset_records(dataset_a.dataset_id)
        assert len(records) == 1
        assert records[0].inputs == {"x": 1}


def test_load_dataset_records_is_workspace_scoped(workspace_tracking_store):
    with WorkspaceContext("team-load-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-load-a")
        dataset_a = workspace_tracking_store.create_dataset(
            name="dataset-load-a",
            experiment_ids=[exp_a],
        )
        workspace_tracking_store.upsert_dataset_records(
            dataset_a.dataset_id,
            [{"inputs": {"x": 1}, "outputs": {"y": 2}}],
        )

        # Same workspace load works
        records, _ = workspace_tracking_store._load_dataset_records(dataset_a.dataset_id)
        assert len(records) == 1
        assert records[0].inputs == {"x": 1}

    with WorkspaceContext("team-load-b"):
        workspace_tracking_store.create_experiment("exp-load-b")

        # Cross-workspace load fails
        with pytest.raises(
            MlflowException,
            match=f"Dataset '{dataset_a.dataset_id}' not found",
        ) as excinfo:
            workspace_tracking_store._load_dataset_records(dataset_a.dataset_id)
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"
