from pathlib import Path

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore
from mlflow.utils.workspace_context import WorkspaceContext
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
