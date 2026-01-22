import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.builtin_scorers import Completeness
from mlflow.genai.scorers.job import run_online_scoring_scheduler
from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore
from mlflow.utils.workspace_context import WorkspaceContext, get_request_workspace
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
        workspace_calls.append(get_request_workspace())
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

    assert workspace_calls == ["team-a", "team-b"]
    assert mock_submit_job.call_count == 2
