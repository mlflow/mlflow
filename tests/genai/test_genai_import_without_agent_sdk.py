import pytest

from mlflow.genai.datasets import create_dataset, delete_dataset, get_dataset
from mlflow.genai.scheduled_scorers import (
    ScorerScheduleConfig,
    add_scheduled_scorer,
    delete_scheduled_scorer,
    get_scheduled_scorer,
    list_scheduled_scorers,
    set_scheduled_scorers,
    update_scheduled_scorer,
)
from mlflow.genai.scorers.base import Scorer


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


class MockScorer(Scorer):
    """Mock scorer for testing purposes."""

    name: str = "mock_scorer"

    def __call__(self, *, outputs=None, **kwargs):
        return {"score": 1.0}


def test_add_scheduled_scorer_raises_when_agents_not_installed():
    mock_scorer = MockScorer()

    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        add_scheduled_scorer(
            experiment_id="test_experiment",
            scheduled_scorer_name="test_scorer",
            scorer=mock_scorer,
            sample_rate=0.5,
            filter_string="test_filter",
        )


def test_update_scheduled_scorer_raises_when_agents_not_installed():
    mock_scorer = MockScorer()

    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        update_scheduled_scorer(
            experiment_id="test_experiment",
            scheduled_scorer_name="test_scorer",
            scorer=mock_scorer,
            sample_rate=0.5,
            filter_string="test_filter",
        )


def test_delete_scheduled_scorer_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        delete_scheduled_scorer(
            experiment_id="test_experiment", scheduled_scorer_name="test_scorer"
        )


def test_get_scheduled_scorer_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        get_scheduled_scorer(experiment_id="test_experiment", scheduled_scorer_name="test_scorer")


def test_list_scheduled_scorers_raises_when_agents_not_installed():
    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        list_scheduled_scorers(experiment_id="test_experiment")


def test_set_scheduled_scorers_raises_when_agents_not_installed():
    mock_scorer = MockScorer()
    scheduled_scorer = ScorerScheduleConfig(
        scorer=mock_scorer, scheduled_scorer_name="test_scorer", sample_rate=0.5
    )

    with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
        set_scheduled_scorers(experiment_id="test_experiment", scheduled_scorers=[scheduled_scorer])
