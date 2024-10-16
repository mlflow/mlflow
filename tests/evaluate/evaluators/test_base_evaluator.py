import numpy as np
import pandas as pd

from mlflow.models.evaluation.evaluators.base import (
    _CustomArtifact,
    _evaluate_custom_artifacts,
    _get_aggregate_metrics_values,
)
from mlflow.models.evaluation.evaluators.regressor import _get_regressor_metrics
from mlflow.models.evaluation.utils.metric import MetricDefinition


def test_evaluate_metric_backwards_compatible():
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    builtin_metrics = _get_regressor_metrics(
        eval_df["target"], eval_df["prediction"], sample_weights=None
    )
    metrics = _get_aggregate_metrics_values(builtin_metrics)

    def old_fn(eval_df, builtin_metrics):
        return builtin_metrics["mean_absolute_error"] * 1.5

    eval_fn_args = [eval_df, builtin_metrics]
    res_metric = MetricDefinition(old_fn, "old_fn", 0).evaluate(eval_fn_args)
    assert res_metric.scores is None
    assert res_metric.justifications is None
    assert res_metric.aggregate_results["old_fn"] == builtin_metrics["mean_absolute_error"] * 1.5

    new_eval_fn_args = [eval_df, None, metrics]

    def new_fn(predictions, targets=None, metrics=None):
        return metrics["mean_absolute_error"].aggregate_results["mean_absolute_error"] * 1.5

    res_metric = MetricDefinition(new_fn, "new_fn", 0).evaluate(new_eval_fn_args)
    assert res_metric.scores is None
    assert res_metric.justifications is None
    assert res_metric.aggregate_results["new_fn"] == builtin_metrics["mean_absolute_error"] * 1.5


def test_evaluate_custom_artifacts_success():
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    metrics = _get_regressor_metrics(eval_df["target"], eval_df["prediction"], sample_weights=None)

    def example_custom_artifacts(given_df, _given_metrics, _artifact_dir):
        return {
            "pred_target_abs_diff": np.abs(given_df["prediction"] - given_df["target"]),
            "example_dictionary_artifact": {"a": 1, "b": 2},
        }

    res_artifacts = _evaluate_custom_artifacts(
        _CustomArtifact(example_custom_artifacts, "", 0, ""), eval_df, metrics
    )

    assert isinstance(res_artifacts, dict)
    assert "pred_target_abs_diff" in res_artifacts
    pd.testing.assert_series_equal(
        res_artifacts["pred_target_abs_diff"], np.abs(eval_df["prediction"] - eval_df["target"])
    )

    assert "example_dictionary_artifact" in res_artifacts
    assert res_artifacts["example_dictionary_artifact"] == {"a": 1, "b": 2}
