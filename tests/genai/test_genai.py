import pytest


def test_namespaced_import_raises_when_agents_not_installed():
    # Ensure that databricks-agents methods renamespaced under mlflow.genai raise an
    # ImportError when the databricks-agents package is not installed.
    import mlflow.genai
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        mlflow.genai.create_dataset("test_schema")

    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        mlflow.genai.get_dataset("test_schema")

    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        mlflow.genai.delete_dataset("test_schema")


def test_namespaced_import_does_not_exist_when_agents_not_installed():
    # Ensure that databricks-agents methods renamespaced under mlflow.genai do not exist
    # when the databricks-agents package is not installed.
    import mlflow.genai
    assert not hasattr(mlflow.genai, "create_labeling_session")
