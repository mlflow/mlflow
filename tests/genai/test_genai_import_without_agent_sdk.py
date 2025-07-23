import pytest

from mlflow.genai.datasets import create_dataset, delete_dataset, get_dataset
from mlflow.genai.experimental.databricks_trace_archival import enable_databricks_trace_archival


# Test `mlflow.genai` namespace
def test_mlflow_genai_star_import_succeeds():
    exec("from mlflow.genai import *")


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


# Test `mlflow.genai.datasets` namespace
def test_mlflow_genai_datasets_star_import_succeeds():
    exec("from mlflow.genai.datasets import *")


def test_create_dataset_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        create_dataset("test_dataset")


def test_get_dataset_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        get_dataset("test_dataset")


def test_delete_dataset_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        delete_dataset("test_dataset")


def test_trace_archival_raises_when_agents_not_installed():
    """Test that ImportError is raised when databricks-agents package is not available."""
    with pytest.raises(
        ImportError,
        match=r"The `mlflow\[databricks\]` package is required to use databricks trace archival",
    ):
        enable_databricks_trace_archival("12345", "catalog", "schema")
