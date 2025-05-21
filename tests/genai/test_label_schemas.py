import pytest


def test_labeling_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        import mlflow.genai.label_schemas  # noqa: F401
