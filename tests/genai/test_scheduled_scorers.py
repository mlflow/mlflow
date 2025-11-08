from mlflow.genai.scheduled_scorers import (
    ScorerScheduleConfig,
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
