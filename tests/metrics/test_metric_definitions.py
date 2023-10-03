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
        toxicity,
        perplexity,
        flesch_kincaid_grade_level,
        ari_grade_level,
    ],
)
def test_return_type_and_len_without_reference(metric):
    eval_df = pd.DataFrame(
        {
            "prediction": ["sentence not", "random text", "a", "c"],
        }
    )
    result = metric.eval_fn(eval_df["prediction"])

    assert isinstance(result, MetricValue)
    if result.scores:
        assert len(result.scores) == len(eval_df)

    result = metric.eval_fn(pd.Series([]))
    assert result is None


@pytest.mark.parametrize(
    "metric",
    [
        rouge1,
        rouge2,
        rougeL,
        rougeLsum,
    ],
)
def test_return_type_and_len_with_target(metric):
    eval_df = pd.DataFrame(
        {
            "prediction": ["sentence not", "random text", "a", "c"],
            "target": ["sentence not", "random text", "a", "c"],
        }
    )
    result = metric.eval_fn(eval_df["prediction"], eval_df["target"])
    assert isinstance(result, MetricValue)
    if result.scores:
        assert len(result.scores) == len(eval_df)

    eval_df = pd.DataFrame(
        {
            "prediction": [],
            "target": [],
        }
    )
    result = metric.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result is None


def _is_toxic(score):
    return score > 0.5


def test_toxicity():
    eval_df = pd.DataFrame({"prediction": ["A normal sentence", "All women are bad"]})
    result = toxicity.eval_fn(eval_df["prediction"])
    assert not _is_toxic(result.scores[0])
    assert _is_toxic(result.scores[1])
    assert result.aggregate_results["ratio"] == 0.5
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] < result.aggregate_results["p90"] < result.scores[1]
    assert "variance" in result.aggregate_results


def test_perplexity():
    eval_df = pd.DataFrame({"prediction": ["sentence not", "This is a sentence"]})
    result = perplexity.eval_fn(eval_df["prediction"])
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
    result = flesch_kincaid_grade_level.eval_fn(eval_df["prediction"])
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
    result = ari_grade_level.eval_fn(eval_df["prediction"])
    assert result.scores[0] < result.scores[1]
    assert result.aggregate_results["mean"] == (result.scores[0] + result.scores[1]) / 2
    assert result.scores[0] < result.aggregate_results["p90"] < result.scores[1]
    assert "variance" in result.aggregate_results


def test_exact_match():
    eval_df = pd.DataFrame(
        {
            "prediction": ["sentence not", "random text", "a", "c"],
            "target": ["sentence not", "random text", "a", "c"],
        }
    )
    result = exact_match.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.aggregate_results["exact_match"] == 1.0

    eval_df = pd.DataFrame(
        {
            "prediction": ["not sentence", "random text", "b", "c"],
            "target": ["sentence not", "random text", "a", "c"],
        }
    )
    result = exact_match.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.aggregate_results["exact_match"] == 0.5


def test_rouge1():
    eval_df = pd.DataFrame({"prediction": ["a", "d c"], "target": ["d", "b c"]})
    result = rouge1.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.scores[0] == 0.0
    assert result.scores[1] == 0.5
    assert result.aggregate_results["mean"] == 0.25
    assert result.aggregate_results["p90"] == 0.45
    assert result.aggregate_results["variance"] == 0.0625


def test_rouge2():
    eval_df = pd.DataFrame({"prediction": ["a e", "b c e"], "target": ["a e", "b c d"]})
    result = rouge2.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.scores[0] == 1.0
    assert result.scores[1] == 0.5
    assert result.aggregate_results["mean"] == 0.75
    assert result.aggregate_results["p90"] == 0.95
    assert result.aggregate_results["variance"] == 0.0625


def test_rougeL():
    eval_df = pd.DataFrame({"prediction": ["a", "b c"], "target": ["d", "b c"]})
    result = rougeL.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.scores[0] == 0.0
    assert result.scores[1] == 1.0
    assert result.aggregate_results["mean"] == 0.5
    assert result.aggregate_results["p90"] == 0.9
    assert result.aggregate_results["variance"] == 0.25


def test_rougeLsum():
    eval_df = pd.DataFrame({"prediction": ["a", "b c"], "target": ["d", "b c"]})
    result = rougeLsum.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.scores[0] == 0.0
    assert result.scores[1] == 1.0
    assert result.aggregate_results["mean"] == 0.5
    assert result.aggregate_results["p90"] == 0.9
    assert result.aggregate_results["variance"] == 0.25


def test_fails_to_load_metric():
    eval_df = pd.DataFrame({"prediction": ["random text", "This is a sentence"]})
    e = ImportError("mocked error")
    with mock.patch("evaluate.load", side_effect=e) as mock_load:
        with mock.patch("mlflow.metrics.metric_definitions._logger.warning") as mock_warning:
            toxicity.eval_fn(eval_df["prediction"])
            mock_load.assert_called_once_with("toxicity", module_type="measurement")
            mock_warning.assert_called_once_with(
                f"Failed to load 'toxicity' metric (error: {e!r}), skipping metric logging.",
            )


def test_mae():
    eval_df = pd.DataFrame({"prediction": [1.0, 2.0, 0.0], "target": [1.0, 2.0, 3.0]})
    result = mae.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.aggregate_results["mean_absolute_error"] == 1.0


def test_mse():
    eval_df = pd.DataFrame({"prediction": [1.0, 2.0, 0.0], "target": [1.0, 2.0, 3.0]})
    result = mse.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.aggregate_results["mean_squared_error"] == 3.0


def test_rmse():
    eval_df = pd.DataFrame({"prediction": [4.0, 5.0, 0.0], "target": [1.0, 2.0, 3.0]})
    result = rmse.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.aggregate_results["root_mean_squared_error"] == 3.0


def test_r2_score():
    eval_df = pd.DataFrame({"prediction": [1.0, 2.0, 3.0], "target": [3.0, 2.0, 1.0]})
    result = r2_score.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.aggregate_results["r2_score"] == -3.0


def test_max_error():
    eval_df = pd.DataFrame({"prediction": [1.0, 2.0, 3.0], "target": [3.0, 2.0, 1.0]})
    result = max_error.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.aggregate_results["max_error"] == 2.0


def test_mape_error():
    eval_df = pd.DataFrame({"prediction": [1.0, 1.0, 1.0], "target": [2.0, 2.0, 2.0]})
    result = mape.eval_fn(eval_df["prediction"], eval_df["target"])
    assert result.aggregate_results["mean_absolute_percentage_error"] == 0.5


def test_binary_recall_score():
    eval_df = pd.DataFrame(
        {"prediction": [0, 0, 1, 1, 0, 0, 0, 1], "target": [1, 1, 1, 1, 0, 0, 0, 0]}
    )
    result = recall_score.eval_fn(eval_df["prediction"], eval_df["target"])
    assert abs(result.aggregate_results["recall_score"] - 0.5) < 1e-3


def test_binary_precision():
    eval_df = pd.DataFrame(
        {"prediction": [0, 0, 1, 1, 0, 0, 0, 1], "target": [1, 1, 1, 1, 0, 0, 0, 0]}
    )
    result = precision_score.eval_fn(eval_df["prediction"], eval_df["target"])
    assert abs(result.aggregate_results["precision_score"] == 0.666) < 1e-3


def test_binary_f1_score():
    eval_df = pd.DataFrame(
        {"prediction": [0, 0, 1, 1, 0, 0, 0, 1], "target": [1, 1, 1, 1, 0, 0, 0, 0]}
    )
    result = f1_score.eval_fn(eval_df["prediction"], eval_df["target"])
    assert abs(result.aggregate_results["f1_score"] - 0.5713) < 1e-3
