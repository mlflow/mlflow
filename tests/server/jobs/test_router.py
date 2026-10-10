from mlflow.server.jobs.router import select_executor_backend


def test_non_custom_routes_to_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)
    assert select_executor_backend(is_custom_scorer=False) == "local"


def test_custom_with_override_routes_to_custom(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.setenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", "sandbox")
    assert select_executor_backend(is_custom_scorer=True) == "sandbox"


def test_custom_without_override_routes_to_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND", "local")
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)
    assert select_executor_backend(is_custom_scorer=True) == "local"
