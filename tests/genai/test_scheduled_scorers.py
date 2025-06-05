import pytest

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


class MockScorer(Scorer):
    """Mock scorer for testing purposes."""

    name: str = "mock_scorer"

    def __call__(self, *, outputs=None, **kwargs):
        return {"score": 1.0}


def test_scheduled_scorer_class_instantiation():
    """Test that ScheduledScorer class can be instantiated without import errors."""
    mock_scorer = MockScorer()
    scheduled_scorer = ScorerScheduleConfig(
        scorer=mock_scorer,
        scheduled_scorer_name="test_scorer",
        sample_rate=0.5,
        filter_string="test_filter",
    )

    assert scheduled_scorer.scorer == mock_scorer
    assert scheduled_scorer.scheduled_scorer_name == "test_scorer"
    assert scheduled_scorer.sample_rate == 0.5
    assert scheduled_scorer.filter_string == "test_filter"


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
