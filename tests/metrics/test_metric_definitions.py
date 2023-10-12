from unittest import mock

import pandas as pd
import pytest

from mlflow.metrics import (
    MetricValue,
    ari_grade_level,
    exact_match,
    f1_score,
    flesch_kincaid_grade_level,
    mae,
    mape,
    max_error,
    mse,
    perplexity,
    precision_score,
    r2_score,
    recall_score,
    rmse,
    rouge1,
    rouge2,
    rougeL,
    rougeLsum,
    toxicity,
)


@pytest.mark.parametrize(
    "metric",
    [
        ari_grade_level,
        exact_match,
        flesch_kincaid_grade_level,
        perplexity,
        rouge1,
        rouge2,
        rougeL,
        rougeLsum,
        toxicity,
    ],
)
def test_return_type_and_len_with_target(metric):
    predictions = pd.Series(["sentence not", "random text", "a", "c"])
    targets = pd.Series(["sentence not", "random text", "a", "c"])

    result = metric.eval_fn(predictions, targets, metrics={})
    assert isinstance(result, MetricValue)
    if result.scores:
        assert len(result.scores) == len(predictions) == len(targets)

    predictions = pd.Series([])
    targets = pd.Series([])
    result = metric.eval_fn(predictions, targets, metrics={})
    assert result is None


def _is_toxic(score):
    return score > 0.5


def test_toxicity():
    predictions = pd.Series(["A normal sentence", "All women are bad"])
    result = toxicity.eval_fn(predictions, None, {})
    assert not _is_toxic(result.scores[0])
    assert _is_toxic(result.scores[1])
    assert result.aggregate_results["ratio"] == 0.5
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] < result.aggregate_results["p90"] < result.scores[1]
    assert "variance" in result.aggregate_results


def test_perplexity():
    predictions = pd.Series(["sentence not", "This is a sentence"])
    result = perplexity.eval_fn(predictions, None, {})
    # A properly structured sentence should have lower perplexity
    assert result.scores[0] > result.scores[1]
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] > result.aggregate_results["p90"] > result.scores[1]
    assert "variance" in result.aggregate_results


def test_flesch_kincaid_grade_level():
    predictions = pd.Series(
        [
            "This is a sentence.",
            (
                "This is a much longer and more complicated sentence than the previous one, "
                "so this sentence should have a higher grade level score."
            ),
        ]
    )
    result = flesch_kincaid_grade_level.eval_fn(predictions, None, {})
    assert result.scores[0] < result.scores[1]
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] < result.aggregate_results["p90"] < result.scores[1]
    assert "variance" in result.aggregate_results


def test_ari_grade_level():
    predictions = pd.Series(
        [
            "This is a sentence.",
            (
                "This is a much longer and more complicated sentence than the previous one, "
                "so this sentence should have a higher grade level score."
            ),
        ]
    )
    result = ari_grade_level.eval_fn(predictions, None, {})
    assert result.scores[0] < result.scores[1]
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] < result.aggregate_results["p90"] < result.scores[1]
    assert "variance" in result.aggregate_results


def test_exact_match():
    predictions = pd.Series(["sentence not", "random text", "a", "c"])
    targets = pd.Series(["sentence not", "random text", "a", "c"])

    result = exact_match.eval_fn(predictions, targets, {})
    assert result.aggregate_results["exact_match"] == 1.0

    predictions = pd.Series(["not sentence", "random text", "b", "c"])
    targets = pd.Series(["sentence not", "random text", "a", "c"])
    result = exact_match.eval_fn(predictions, targets, {})
    assert result.aggregate_results["exact_match"] == 0.5


def test_rouge1():
    predictions = pd.Series(["a", "d c"])
    targets = pd.Series(["d", "b c"])
    result = rouge1.eval_fn(predictions, targets, {})
    assert result.scores[0] == 0.0
    assert result.scores[1] == 0.5
    assert result.aggregate_results["mean"] == 0.25
    assert result.aggregate_results["p90"] == 0.45
    assert result.aggregate_results["variance"] == 0.0625


def test_rouge2():
    predictions = pd.Series(["a e", "b c e"])
    targets = pd.Series(["a e", "b c d"])
    result = rouge2.eval_fn(predictions, targets, {})
    assert result.scores[0] == 1.0
    assert result.scores[1] == 0.5
    assert result.aggregate_results["mean"] == 0.75
    assert result.aggregate_results["p90"] == 0.95
    assert result.aggregate_results["variance"] == 0.0625


def test_rougeL():
    predictions = pd.Series(["a", "b c"])
    targets = pd.Series(["d", "b c"])
    result = rougeL.eval_fn(predictions, targets, {})
    assert result.scores[0] == 0.0
    assert result.scores[1] == 1.0
    assert result.aggregate_results["mean"] == 0.5
    assert result.aggregate_results["p90"] == 0.9
    assert result.aggregate_results["variance"] == 0.25


def test_rougeLsum():
    predictions = pd.Series(["a", "b c"])
    targets = pd.Series(["d", "b c"])
    result = rougeLsum.eval_fn(predictions, targets, {})
    assert result.scores[0] == 0.0
    assert result.scores[1] == 1.0
    assert result.aggregate_results["mean"] == 0.5
    assert result.aggregate_results["p90"] == 0.9
    assert result.aggregate_results["variance"] == 0.25


def test_fails_to_load_metric():
    predictions = pd.Series(["random text", "This is a sentence"])
    e = ImportError("mocked error")
    with mock.patch("evaluate.load", side_effect=e) as mock_load:
        with mock.patch("mlflow.metrics.metric_definitions._logger.warning") as mock_warning:
            toxicity.eval_fn(predictions, None, {})
            mock_load.assert_called_once_with("toxicity", module_type="measurement")
            mock_warning.assert_called_once_with(
                f"Failed to load 'toxicity' metric (error: {e!r}), skipping metric logging.",
            )


def test_mae():
    predictions = pd.Series([1.0, 2.0, 0.0])
    targets = pd.Series([1.0, 2.0, 3.0])
    result = mae.eval_fn(predictions, targets, {})
    assert result.aggregate_results["mean_absolute_error"] == 1.0


def test_mse():
    predictions = pd.Series([1.0, 2.0, 0.0])
    targets = pd.Series([1.0, 2.0, 3.0])
    result = mse.eval_fn(predictions, targets, {})
    assert result.aggregate_results["mean_squared_error"] == 3.0


def test_rmse():
    predictions = pd.Series([4.0, 5.0, 0.0])
    targets = pd.Series([1.0, 2.0, 3.0])
    result = rmse.eval_fn(predictions, targets, {})
    assert result.aggregate_results["root_mean_squared_error"] == 3.0


def test_r2_score():
    predictions = pd.Series([1.0, 2.0, 3.0])
    targets = pd.Series([3.0, 2.0, 1.0])
    result = r2_score.eval_fn(predictions, targets, {})
    assert result.aggregate_results["r2_score"] == -3.0


def test_max_error():
    predictions = pd.Series([1.0, 2.0, 3.0])
    targets = pd.Series([3.0, 2.0, 1.0])
    result = max_error.eval_fn(predictions, targets, {})
    assert result.aggregate_results["max_error"] == 2.0


def test_mape_error():
    predictions = pd.Series([1.0, 1.0, 1.0])
    targets = pd.Series([2.0, 2.0, 2.0])
    result = mape.eval_fn(predictions, targets, {})
    assert result.aggregate_results["mean_absolute_percentage_error"] == 0.5


def test_binary_recall_score():
    predictions = pd.Series([0, 0, 1, 1, 0, 0, 0, 1])
    targets = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    result = recall_score.eval_fn(predictions, targets, {})
    assert abs(result.aggregate_results["recall_score"] - 0.5) < 1e-3


def test_binary_precision():
    predictions = pd.Series([0, 0, 1, 1, 0, 0, 0, 1])
    targets = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    result = precision_score.eval_fn(predictions, targets, {})
    assert abs(result.aggregate_results["precision_score"] == 0.666) < 1e-3


def test_binary_f1_score():
    predictions = pd.Series([0, 0, 1, 1, 0, 0, 0, 1])
    targets = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    result = f1_score.eval_fn(predictions, targets, {})
    assert abs(result.aggregate_results["f1_score"] - 0.5713) < 1e-3
