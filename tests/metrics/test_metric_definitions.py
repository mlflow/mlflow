from unittest import mock

import pandas as pd
import pytest

from mlflow.metrics import (
    MetricValue,
    accuracy,
    ari_grade_level,
    flesch_kincaid_grade_level,
    perplexity,
    rouge1,
    rouge2,
    rougeL,
    rougeLsum,
    toxicity,
)


@pytest.mark.parametrize(
    "metric",
    [
        toxicity,
        perplexity,
        flesch_kincaid_grade_level,
        ari_grade_level,
        accuracy,
        rouge1,
        rouge2,
        rougeL,
        rougeLsum,
    ],
)
def test_return_type_and_len(metric):
    eval_df = pd.DataFrame(
        {
            "prediction": ["sentence not", "random text", "a", "c"],
            "target": ["sentence not", "random text", "a", "c"],
        }
    )
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
    assert result.aggregate_results["ratio"] == 0.5
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] < result.aggregate_results["p90"] < result.scores[1]
    assert "variance" in result.aggregate_results


def test_perplexity():
    eval_df = pd.DataFrame({"prediction": ["sentence not", "This is a sentence"]})
    result = perplexity.eval_fn(eval_df, metrics={})
    # A properly structured sentence should have lower perplexity
    assert result.scores[0] > result.scores[1]
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] > result.aggregate_results["p90"] > result.scores[1]
    assert "variance" in result.aggregate_results


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
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] < result.aggregate_results["p90"] < result.scores[1]
    assert "variance" in result.aggregate_results


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
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] < result.aggregate_results["p90"] < result.scores[1]
    assert "variance" in result.aggregate_results


def test_accuracy():
    eval_df = pd.DataFrame(
        {
            "prediction": ["sentence not", "random text", "a", "c"],
            "target": ["sentence not", "random text", "a", "c"],
        }
    )
    result = accuracy.eval_fn(eval_df, metrics={})
    assert result.aggregate_results[""] == 1.0

    eval_df = pd.DataFrame(
        {
            "prediction": ["not sentence", "random text", "b", "c"],
            "target": ["sentence not", "random text", "a", "c"],
        }
    )
    result = accuracy.eval_fn(eval_df, metrics={})
    assert result.aggregate_results[""] == 0.5


def test_rouge1():
    eval_df = pd.DataFrame({"prediction": ["a", "d c"], "target": ["d", "b c"]})
    result = rouge1.eval_fn(eval_df, metrics={})
    assert result.scores[0] == 0.0
    assert result.scores[1] == 0.5
    assert result.aggregate_results["mean"] == 0.25
    assert result.aggregate_results["p90"] == 0.45
    assert result.aggregate_results["variance"] == 0.0625


def test_rouge2():
    eval_df = pd.DataFrame({"prediction": ["a e", "b c e"], "target": ["a e", "b c d"]})
    result = rouge2.eval_fn(eval_df, metrics={})
    assert result.scores[0] == 1.0
    assert result.scores[1] == 0.5
    assert result.aggregate_results["mean"] == 0.75
    assert result.aggregate_results["p90"] == 0.95
    assert result.aggregate_results["variance"] == 0.0625


def test_rougeL():
    eval_df = pd.DataFrame({"prediction": ["a", "b c"], "target": ["d", "b c"]})
    result = rougeL.eval_fn(eval_df, metrics={})
    assert result.scores[0] == 0.0
    assert result.scores[1] == 1.0
    assert result.aggregate_results["mean"] == 0.5
    assert result.aggregate_results["p90"] == 0.9
    assert result.aggregate_results["variance"] == 0.25


def test_rougeLsum():
    eval_df = pd.DataFrame({"prediction": ["a", "b c"], "target": ["d", "b c"]})
    result = rougeLsum.eval_fn(eval_df, metrics={})
    assert result.scores[0] == 0.0
    assert result.scores[1] == 1.0
    assert result.aggregate_results["mean"] == 0.5
    assert result.aggregate_results["p90"] == 0.9
    assert result.aggregate_results["variance"] == 0.25


def test_fails_to_load_metric():
    eval_df = pd.DataFrame({"prediction": ["random text", "This is a sentence"]})
    e = ImportError("mocked error")
    with mock.patch("evaluate.load", side_effect=e) as mock_load:
        with mock.patch("mlflow.metrics.utils.metric_definitions._logger.warning") as mock_warning:
            toxicity.eval_fn(eval_df, metrics={})
            mock_load.assert_called_once_with("toxicity", module_type="measurement")
            mock_warning.assert_called_once_with(
                f"Failed to load 'toxicity' metric (error: {e!r}), skipping metric logging.",
            )
