import pytest

from mlflow.genai.datasets import create_dataset, delete_dataset, get_dataset


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
