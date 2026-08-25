from unittest import mock

from mlflow.server.jobs.router import JobExecutorRouter


def _router():
    return JobExecutorRouter(registry=mock.Mock())


def test_non_custom_routes_to_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)
    assert _router().select("invoke_scorer", is_custom_scorer=False) == "local"


def test_custom_with_override_routes_to_custom(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.setenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", "sandbox")
    assert _router().select("invoke_scorer", is_custom_scorer=True) == "sandbox"


def test_custom_without_override_routes_to_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)
    assert _router().select("invoke_scorer", is_custom_scorer=True) == "local"
