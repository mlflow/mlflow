import time
import uuid
from pathlib import Path

import pytest

from mlflow.entities import (
    Dataset,
    DatasetInput,
    ExperimentTag,
    InputTag,
    Metric,
    Param,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlExperiment
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service import utils as tracking_utils
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.tracking._workspace.context import WorkspaceContext, clear_workspace
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.uri import append_to_uri_path
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


@pytest.fixture(autouse=True)
def _reset_workspace_context():
    clear_workspace()
    yield
    clear_workspace()


@pytest.fixture
def workspace_tracking_store(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    backend_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = SqlAlchemyStore(backend_uri, artifact_dir.as_uri())
    store.tracking_uri = backend_uri
    store.artifact_root_uri = artifact_dir.as_uri()
    try:
        yield store
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


def test_artifact_locations_are_scoped_to_workspace(workspace_tracking_store):
    with WorkspaceContext("team-alpha"):
        exp_id = workspace_tracking_store.create_experiment("alpha-exp")
        experiment = workspace_tracking_store.get_experiment(exp_id)
        assert "/workspaces/team-alpha/" in experiment.artifact_location

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        default_exp_id = workspace_tracking_store.create_experiment("default-exp")
        default_experiment = workspace_tracking_store.get_experiment(default_exp_id)
        assert f"/workspaces/{DEFAULT_WORKSPACE_NAME}/" in default_experiment.artifact_location


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


def test_workspace_startup_rejects_root_ending_with_workspaces(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    backend_uri = f"sqlite:///{tmp_path / 'suffix.db'}"
    bad_root = tmp_path / "base" / "workspaces"
    bad_root.mkdir(parents=True)

    with pytest.raises(
        MlflowException,
        match="ends with the reserved 'workspaces' segment",
    ) as excinfo:
        SqlAlchemyStore(backend_uri, bad_root.as_uri())
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
        SqlAlchemyStore(backend_uri, bad_root.as_uri())
    assert excinfo.value.error_code == "INVALID_STATE"
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)


def test_workspace_startup_detects_existing_reserved_artifact(tmp_path, monkeypatch):
    backend_uri = f"sqlite:///{tmp_path / 'conflict.db'}"
    base_root = tmp_path / "base"
    base_root.mkdir()

    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    legacy_store = SqlAlchemyStore(backend_uri, base_root.as_uri())
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
        SqlAlchemyStore(backend_uri, base_root.as_uri())
    assert excinfo.value.error_code == "INVALID_STATE"
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)


def test_workspace_startup_ignores_default_experiment_reserved_location(tmp_path, monkeypatch):
    backend_uri = f"sqlite:///{tmp_path / 'default_conflict.db'}"
    base_root = tmp_path / "base"
    base_root.mkdir()

    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    legacy_store = SqlAlchemyStore(backend_uri, base_root.as_uri())
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
    workspace_store = SqlAlchemyStore(backend_uri, base_root.as_uri())
    workspace_store._dispose_engine()
    SqlAlchemyStore._db_uri_sql_alchemy_engine_map.pop(backend_uri, None)


def test_single_tenant_startup_rejects_non_default_workspace_experiments(tmp_path, monkeypatch):
    backend_uri = f"sqlite:///{tmp_path / 'multi_tenant.db'}"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    workspace_store = SqlAlchemyStore(backend_uri, artifact_root.as_uri())

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
