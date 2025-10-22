from unittest.mock import patch

import pytest

from mlflow.genai.datasets import create_dataset, delete_dataset, get_dataset
from mlflow.genai.scorers import (
    delete_scorer,
    get_scorer,
    list_scorers,
)
from mlflow.genai.scorers.base import Scorer


# Test `mlflow.genai` namespace
def test_mlflow_genai_star_import_succeeds():
    exec("from mlflow.genai import *")


def test_namespaced_import_raises_when_agents_not_installed():
    # Ensure that databricks-agents methods renamespaced under mlflow.genai raise an
    # ImportError when the databricks-agents package is not installed.
    import mlflow.genai

    # Mock to simulate Databricks environment without databricks-agents installed
    with patch("mlflow.genai.datasets.is_databricks_default_tracking_uri", return_value=True):
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
    # Mock to simulate Databricks environment without databricks-agents installed
    with patch("mlflow.genai.datasets.is_databricks_default_tracking_uri", return_value=True):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            create_dataset("test_dataset")


def test_get_dataset_raises_when_agents_not_installed():
    # Mock to simulate Databricks environment without databricks-agents installed
    with patch("mlflow.genai.datasets.is_databricks_default_tracking_uri", return_value=True):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            get_dataset("test_dataset")


def test_delete_dataset_raises_when_agents_not_installed():
    # Mock to simulate Databricks environment without databricks-agents installed
    with patch("mlflow.genai.datasets.is_databricks_default_tracking_uri", return_value=True):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            delete_dataset("test_dataset")


class MockScorer(Scorer):
    """Mock scorer for testing purposes."""

    name: str = "mock_scorer"

    def __call__(self, *, outputs=None, **kwargs):
        return {"score": 1.0}


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_list_scorers_raises_when_agents_not_installed(_):
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        list_scorers(experiment_id="test_experiment")


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_get_scorer_raises_when_agents_not_installed(_):
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        get_scorer(name="test_scorer", experiment_id="test_experiment")


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_delete_scorer_raises_when_agents_not_installed(_):
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        delete_scorer(experiment_id="test_experiment", name="test_scorer")
