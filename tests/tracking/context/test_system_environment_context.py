from mlflow.tracking.context.system_environment_context import SystemEnvironmentContext


def test_system_environment_context_in_context(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_CONTEXT", '{"A": "B"}')
    assert SystemEnvironmentContext().in_context()
    monkeypatch.delenv("MLFLOW_RUN_CONTEXT", raising=True)
    assert not SystemEnvironmentContext().in_context()


def test_system_environment_context_tags(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_CONTEXT", '{"A": "B", "C": "D"}')
    assert SystemEnvironmentContext().tags() == {"A": "B", "C": "D"}
