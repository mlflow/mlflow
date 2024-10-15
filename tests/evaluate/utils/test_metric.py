from unittest import mock

import pandas as pd
import pytest

from mlflow.metrics import MetricValue
from mlflow.models.evaluation.evaluators.base import _get_aggregate_metrics_values
from mlflow.models.evaluation.evaluators.regressor import _get_regressor_metrics
from mlflow.models.evaluation.utils.metric import MetricDefinition


def test_evaluate_custom_metric_incorrect_return_formats():
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    builtin_metrics = _get_regressor_metrics(
        eval_df["target"], eval_df["prediction"], sample_weights=None
    )
    eval_fn_args = [eval_df, builtin_metrics]

    def dummy_fn(*_):
        pass

    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        MetricDefinition(dummy_fn, "dummy_fn", 0, None).evaluate(eval_fn_args)
        mock_warning.assert_called_once_with(
            "Did not log metric 'dummy_fn' at index 0 in the `extra_metrics` parameter"
            " because it returned None."
        )

    def incorrect_return_type(*_):
        return ["stuff"], 3

    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        metric = MetricDefinition(incorrect_return_type, incorrect_return_type.__name__, 0)
        metric.evaluate(eval_fn_args)
        mock_warning.assert_called_once_with(
            f"Did not log metric '{incorrect_return_type.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it did not return a MetricValue."
        )

    def non_list_scores(*_):
        return MetricValue(scores=5)

    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        MetricDefinition(non_list_scores, non_list_scores.__name__, 0).evaluate(eval_fn_args)
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_list_scores.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with scores as a list."
        )

    def non_numeric_scores(*_):
        return MetricValue(scores=[{"val": "string"}])

    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        MetricDefinition(non_numeric_scores, non_numeric_scores.__name__, 0).evaluate(eval_fn_args)
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_numeric_scores.__name__}' at index 0 in the `extra_metrics`"
            " parameter because it must return MetricValue with numeric or string scores."
        )

    def non_list_justifications(*_):
        return MetricValue(justifications="string")

    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        metric = MetricDefinition(non_list_justifications, non_list_justifications.__name__, 0)
        metric.evaluate(eval_fn_args)
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_list_justifications.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with justifications "
            "as a list."
        )

    def non_str_justifications(*_):
        return MetricValue(justifications=[3, 4])

    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        metric = MetricDefinition(non_str_justifications, non_str_justifications.__name__, 0)
        metric.evaluate(eval_fn_args)
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_str_justifications.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with string "
            "justifications."
        )

    def non_dict_aggregates(*_):
        return MetricValue(aggregate_results=[5.0, 4.0])

    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        metric = MetricDefinition(non_dict_aggregates, non_dict_aggregates.__name__, 0)
        metric.evaluate(eval_fn_args)
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_dict_aggregates.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with aggregate_results "
            "as a dict."
        )

    def wrong_type_aggregates(*_):
        return MetricValue(aggregate_results={"toxicity": 0.0, "hi": "hi"})

    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        metric = MetricDefinition(wrong_type_aggregates, wrong_type_aggregates.__name__, 0)
        metric.evaluate(eval_fn_args)
        mock_warning.assert_called_once_with(
            f"Did not log metric '{wrong_type_aggregates.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with aggregate_results "
            "with str keys and numeric values."
        )


@pytest.mark.parametrize(
    "fn",
    [
        (
            lambda eval_df, _: MetricValue(
                scores=eval_df["prediction"].tolist(),
                aggregate_results={"prediction_sum": sum(eval_df["prediction"])},
            )
        ),
        (
            lambda eval_df, _: MetricValue(
                scores=eval_df["prediction"].tolist()[:-1] + [None],
                aggregate_results={"prediction_sum": None, "another_aggregate": 5.0},
            )
        ),
    ],
)
def test_evaluate_custom_metric_lambda(fn):
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    builtin_metrics = _get_regressor_metrics(
        eval_df["target"], eval_df["prediction"], sample_weights=None
    )
    metrics = _get_aggregate_metrics_values(builtin_metrics)
    eval_fn_args = [eval_df, metrics]
    with mock.patch("mlflow.models.evaluation.utils.metric._logger.warning") as mock_warning:
        MetricDefinition(fn, "<lambda>", 0).evaluate(eval_fn_args)
        mock_warning.assert_not_called()


def test_evaluate_custom_metric_success():
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    builtin_metrics = _get_regressor_metrics(
        eval_df["target"], eval_df["prediction"], sample_weights=None
    )

    def example_count_times_1_point_5(predictions, targets=None, metrics=None):
        return MetricValue(
            scores=[score * 1.5 for score in predictions.tolist()],
            justifications=["justification"] * len(predictions),
            aggregate_results={
                "example_count_times_1_point_5": metrics["example_count"].aggregate_results[
                    "example_count"
                ]
                * 1.5
            },
        )

    eval_fn_args = [eval_df["prediction"], None, _get_aggregate_metrics_values(builtin_metrics)]
    res_metric = MetricDefinition(example_count_times_1_point_5, "", 0).evaluate(eval_fn_args)
    assert (
        res_metric.aggregate_results["example_count_times_1_point_5"]
        == builtin_metrics["example_count"] * 1.5
    )
    assert res_metric.scores == [score * 1.5 for score in eval_df["prediction"].tolist()]
    assert res_metric.justifications == ["justification"] * len(eval_df["prediction"])
