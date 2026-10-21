import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities import Workspace
from mlflow.entities._job import JobProgress
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES, MLFLOW_WORKSPACE
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.builtin_scorers import Completeness
from mlflow.genai.scorers.job import run_online_scoring_scheduler
from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
from mlflow.server.jobs.utils import _exec_job, _for_each_unfinished_job, _SubprocessJobResult
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore
from mlflow.utils.workspace_context import (
    ServerWorkspaceContext,
    WorkspaceContext,
    get_request_workspace,
)
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


def test_sqlalchemy_job_store_defaults_to_legacy_workspace(tmp_path: Path):
    backend_store_uri = f"sqlite:///{tmp_path / 'workspace-default.db'}"
    store = SqlAlchemyJobStore(backend_store_uri)

    job = store.create_job("tests.server.jobs.test_jobs.basic_job_fun", '{"value": 1}')
    assert job.workspace == DEFAULT_WORKSPACE_NAME
    stored = store.get_job(job.job_id)
    assert stored.workspace == DEFAULT_WORKSPACE_NAME


def test_sqlalchemy_job_store_isolates_workspaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    backend_store_uri = f"sqlite:///{tmp_path / 'workspace-aware.db'}"
    store = WorkspaceAwareSqlAlchemyJobStore(backend_store_uri)

    with WorkspaceContext("team-a"):
        job_team_a = store.create_job("tests.server.jobs.test_jobs.basic_job_fun", '{"value": 1}')

    with WorkspaceContext("team-b"):
        job_team_b = store.create_job("tests.server.jobs.test_jobs.basic_job_fun", '{"value": 2}')

    with WorkspaceContext("team-a"):
        fetched_a = store.get_job(job_team_a.job_id)
        assert fetched_a.workspace == "team-a"
        with pytest.raises(MlflowException, match="not found"):
            store.get_job(job_team_b.job_id)
        assert {job.job_id for job in store.list_jobs()} == {job_team_a.job_id}

    with WorkspaceContext("team-b"):
        fetched_b = store.get_job(job_team_b.job_id)
        assert fetched_b.workspace == "team-b"
        assert {job.job_id for job in store.list_jobs()} == {job_team_b.job_id}


def test_scheduler_runs_per_workspace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.delenv(MLFLOW_WORKSPACE.name, raising=False)

    mock_scorer = OnlineScorer(
        name="completeness",
        serialized_scorer=json.dumps(Completeness().model_dump()),
        online_config=OnlineScoringConfig(
            online_scoring_config_id=uuid.uuid4().hex,
            scorer_id=uuid.uuid4().hex,
            sample_rate=1.0,
            experiment_id="exp1",
            filter_string=None,
        ),
    )

    mock_tracking_store = MagicMock()
    workspace_calls = []

    def _get_active_online_scorers():
        # The scheduler shares its process with other periodic tasks' threads, so it must not
        # set the workspace in the process environment.
        workspace_calls.append((get_request_workspace(), os.environ.get(MLFLOW_WORKSPACE.name)))
        return [mock_scorer]

    mock_tracking_store.get_active_online_scorers.side_effect = _get_active_online_scorers

    mock_workspace_store = MagicMock()
    mock_workspace_store.list_workspaces.return_value = [
        Workspace(name="team-a"),
        Workspace(name="team-b"),
    ]

    with (
        patch("mlflow.genai.scorers.job._get_tracking_store", return_value=mock_tracking_store),
        patch(
            "mlflow.server.workspace_helpers._get_workspace_store",
            return_value=mock_workspace_store,
        ),
        patch("mlflow.genai.scorers.job.submit_job") as mock_submit_job,
    ):
        run_online_scoring_scheduler()

    assert workspace_calls == [("team-a", None), ("team-b", None)]
    assert mock_submit_job.call_count == 2


def test_update_status_details_workspace_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    backend_store_uri = f"sqlite:///{tmp_path / 'workspace-metadata.db'}"
    store = WorkspaceAwareSqlAlchemyJobStore(backend_store_uri)

    with WorkspaceContext("team-a"):
        job_a = store.create_job("test_job", '{"value": 1}')
        store.update_status_details(job_a.job_id, {"stage": "team-a-stage"})

    with WorkspaceContext("team-b"):
        job_b = store.create_job("test_job", '{"value": 2}')
        store.update_status_details(job_b.job_id, {"stage": "team-b-stage"})

    with WorkspaceContext("team-a"):
        fetched_a = store.get_job(job_a.job_id)
        assert fetched_a.status_details == {"stage": "team-a-stage"}

        with pytest.raises(MlflowException, match="not found"):
            store.update_status_details(job_b.job_id, {"stage": "attempt"})

    with WorkspaceContext("team-b"):
        fetched_b = store.get_job(job_b.job_id)
        assert fetched_b.status_details == {"stage": "team-b-stage"}


def test_update_job_progress_workspace_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    backend_store_uri = f"sqlite:///{tmp_path / 'workspace-progress.db'}"
    store = WorkspaceAwareSqlAlchemyJobStore(backend_store_uri)

    with WorkspaceContext("team-a"):
        job_a = store.create_job("test_job", '{"value": 1}')
        store.update_job_progress(
            job_a.job_id,
            message="team-a-progress",
            progress=JobProgress(phase="team-a", completed=1, total=2, unit="items"),
        )

    with WorkspaceContext("team-b"):
        job_b = store.create_job("test_job", '{"value": 2}')
        store.update_job_progress(
            job_b.job_id,
            message="team-b-progress",
            progress=JobProgress(phase="team-b", completed=1, total=3, unit="items"),
        )

    with WorkspaceContext("team-a"):
        fetched_a = store.get_job(job_a.job_id)
        assert fetched_a.status_message == "team-a-progress"
        assert fetched_a.progress == JobProgress(phase="team-a", completed=1, total=2, unit="items")

        with pytest.raises(MlflowException, match="not found"):
            store.update_job_progress(
                job_b.job_id,
                message="attempt",
                progress=JobProgress(phase="blocked"),
            )

    with WorkspaceContext("team-b"):
        fetched_b = store.get_job(job_b.job_id)
        assert fetched_b.status_message == "team-b-progress"
        assert fetched_b.progress == JobProgress(phase="team-b", completed=1, total=3, unit="items")


def test_create_job_initializes_metadata_as_none(tmp_path: Path):
    backend_store_uri = f"sqlite:///{tmp_path / 'metadata-init.db'}"
    store = SqlAlchemyJobStore(backend_store_uri)

    job = store.create_job("test_job", '{"param": "value"}')
    assert job.status_details is None
    assert job.error_message is None
    assert job.executor_backend is None
    assert job.lease_expires_at is None
    assert job.status_message is None
    assert job.progress is None
    assert job.progress_updated_at is None
    assert job.token_hash is None
    assert job.scoped_permissions is None

    fetched_job = store.get_job(job.job_id)
    assert fetched_job.status_details is None
    assert fetched_job.error_message is None
    assert fetched_job.executor_backend is None
    assert fetched_job.lease_expires_at is None
    assert fetched_job.status_message is None
    assert fetched_job.progress is None
    assert fetched_job.progress_updated_at is None
    assert fetched_job.token_hash is None
    assert fetched_job.scoped_permissions is None


def test_exec_job_sets_the_workspace_only_for_its_thread(monkeypatch, tmp_path: Path):
    # Huey runs jobs on threads of one consumer process. If a job set its workspace in
    # MLFLOW_WORKSPACE, a concurrent job from another workspace could read it.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.delenv(MLFLOW_WORKSPACE.name, raising=False)
    store = WorkspaceAwareSqlAlchemyJobStore(f"sqlite:///{tmp_path / 'exec-job.db'}")
    with ServerWorkspaceContext("team-a"):
        job = store.create_job("basic_job_fun", json.dumps({"x": 1, "y": 2}))
    seen = []

    def _exec_job_in_subproc(*args, **kwargs):
        # The subprocess gets the workspace from this argument, not from this process's env.
        workspace_arg = args[9]
        seen.append((get_request_workspace(), os.environ.get(MLFLOW_WORKSPACE.name), workspace_arg))
        return _SubprocessJobResult(succeeded=True, result="3")

    with (
        patch("mlflow.server.handlers._get_job_store", return_value=store),
        patch(
            "mlflow.server.jobs.utils.get_job_fn_fullname",
            return_value="tests.server.jobs.test_jobs.basic_job_fun",
        ),
        patch("mlflow.server.jobs.utils._exec_job_in_subproc", side_effect=_exec_job_in_subproc),
    ):
        _exec_job(job.job_id, "team-a", "basic_job_fun", {"x": 1, "y": 2}, None)

    assert seen == [("team-a", None, "team-a")]
    assert os.environ.get(MLFLOW_WORKSPACE.name) is None
    with ServerWorkspaceContext("team-a"):
        assert store.get_job(job.job_id).status == JobStatus.SUCCEEDED


def test_unfinished_job_loop_sets_each_workspace_only_for_its_thread(monkeypatch):
    # Startup recovery and the executor engine's scheduler, which runs every second next to the
    # worker threads, both loop over workspaces this way.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.delenv(MLFLOW_WORKSPACE.name, raising=False)
    seen = []

    def _list_jobs(**kwargs):
        seen.append((get_request_workspace(), os.environ.get(MLFLOW_WORKSPACE.name)))
        return []

    job_store = MagicMock()
    job_store.list_jobs.side_effect = _list_jobs
    workspace_store = MagicMock()
    workspace_store.list_workspaces.return_value = [
        Workspace(name="team-a"),
        Workspace(name="team-b"),
    ]

    with patch(
        "mlflow.server.workspace_helpers._get_workspace_store", return_value=workspace_store
    ):
        _for_each_unfinished_job(job_store, [JobStatus.PENDING], 0, MagicMock())

    assert seen == [("team-a", None), ("team-b", None)]
