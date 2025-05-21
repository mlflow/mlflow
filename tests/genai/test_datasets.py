"""This module tests the mlflow.genai.datasets module."""
import pytest

from mlflow.genai.datasets import create_dataset, delete_dataset, get_dataset


def test_create_dataset_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        create_dataset("test_dataset")


def test_get_dataset_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        get_dataset("test_dataset")


def test_delete_dataset_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        delete_dataset("test_dataset")


def test_evaluation_dataset_raises_when_agents_not_installed():
    # A warning is logged when the module is imported, but when EvaluationDataset is explicitly
    # imported, it raises an ImportError.
    with pytest.raises(ImportError, match="cannot import name 'EvaluationDataset'"):
        from mlflow.genai.datasets import EvaluationDataset
