import pandas as pd
import pytest

from mlflow.metrics import (
    MetricValue,
    ari_grade_level,
    flesch_kincaid_grade_level,
    perplexity,
    toxicity,
)

# test success and failure, logging issues and skipping metrics


@pytest.mark.parametrize(
    "metric", [toxicity, perplexity, flesch_kincaid_grade_level, ari_grade_level]
)
def test_return_type_and_len(metric):
    eval_df = pd.DataFrame({"prediction": ["sentence not", "random text", "a", "c"]})
    result = metric.eval_fn(eval_df, metrics={})
    assert isinstance(result, MetricValue)
    if result.scores:
        assert len(result.scores) == len(eval_df)

    result = metric.eval_fn(pd.DataFrame({"prediction": []}), metrics={})
    assert result is None


def _is_toxic(score):
    return score > 0.5


def test_toxicity():
    eval_df = pd.DataFrame({"prediction": ["A normal sentence", "All women are bad"]})
    result = toxicity.eval_fn(eval_df, metrics={})
    assert not _is_toxic(result.scores[0])
    assert _is_toxic(result.scores[1])
    assert "ratio" in result.aggregate_results


def test_perplexity():
    eval_df = pd.DataFrame({"prediction": ["sentence not", "This is a sentence"]})
    result = perplexity.eval_fn(eval_df, metrics={})
    # A properly structured sentence should have lower perplexity
    assert result.scores[0] > result.scores[1]
    assert "mean" in result.aggregate_results


def test_flesch_kincaid_grade_level():
    eval_df = pd.DataFrame(
        {
            "prediction": [
                "This is a sentence.",
                (
                    "This is a much longer and more complicated sentence than the previous one, "
                    "so this sentence should have a higher grade level score."
                ),
            ]
        }
    )
    result = flesch_kincaid_grade_level.eval_fn(eval_df, metrics={})
    assert result.scores[0] < result.scores[1]
    assert "mean" in result.aggregate_results


def test_ari_grade_level():
    eval_df = pd.DataFrame(
        {
            "prediction": [
                "This is a sentence.",
                (
                    "This is a much longer and more complicated sentence than the previous one, "
                    "so this sentence should have a higher grade level score."
                ),
            ]
        }
    )
    result = ari_grade_level.eval_fn(eval_df, metrics={})
    assert result.scores[0] < result.scores[1]
    assert "mean" in result.aggregate_results
