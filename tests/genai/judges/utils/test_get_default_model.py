import pytest

from mlflow.genai.judges.utils import get_default_model


@pytest.mark.parametrize(
    ("env_value", "tracking_uri", "expected"),
    [
        ("openai:/gpt-5", "databricks", "openai:/gpt-5"),
        ("anthropic:/claude-4", "http://localhost:5000", "anthropic:/claude-4"),
        ("", "databricks", "databricks"),
        ("", "http://localhost:5000", "openai:/gpt-4.1-mini"),
        (None, "databricks", "databricks"),
        (None, "http://localhost:5000", "openai:/gpt-4.1-mini"),
    ],
)
def test_get_default_model(monkeypatch, env_value, tracking_uri, expected):
    if env_value is not None:
        monkeypatch.setenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", env_value)
    else:
        monkeypatch.delenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", raising=False)
    monkeypatch.setattr("mlflow.genai.judges.utils.mlflow.get_tracking_uri", lambda: tracking_uri)
    monkeypatch.setattr(
        "mlflow.genai.judges.utils.is_databricks_uri",
        lambda uri: uri == "databricks",
    )
    assert get_default_model() == expected
