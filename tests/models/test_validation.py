from mlflow.models.evaluation import (
    evaluate,
    EvaluationResult,
    ModelEvaluator,
    MetricThreshold,
)
from mlflow.models.evaluation.validation import (
    _MetricValidationResult,
    ModelValidationFailedException,
)
from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry
from unittest import mock
import pytest

# pylint: disable=unused-import
from tests.models.test_evaluation import (
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
)

message_separator = "\n"


class MockEvaluator(ModelEvaluator):
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs):
        raise RuntimeError()

    def evaluate(self, *, model, model_type, dataset, run_id, evaluator_config, **kwargs):
        raise RuntimeError()


@pytest.fixture
def value_threshold_test_spec(request):
    """
    Test specification for value threshold tests:
    :return: (
                metrics: A dictionary mapping scalar metric names to scalar metric values,
                validation_threhsolds: A dictonary mapping scalar metric names
                    to MetricThreshold(threshold=0.2, higher_is_better=True),
                expected_validation_results: A dictonary mapping scalar metric names
                    to _MetricValidationResult
             )
    """
    acc_threshold = MetricThreshold(threshold=0.9, higher_is_better=True)
    acc_validation_result = _MetricValidationResult("accuracy", 0.8, acc_threshold, None)
    acc_validation_result.threshold_failed = True

    f1score_threshold = MetricThreshold(threshold=0.8, higher_is_better=True)
    f1score_validation_result = _MetricValidationResult("f1_score", 0.7, f1score_threshold, None)
    f1score_validation_result.threshold_failed = True

    log_loss_threshold = MetricThreshold(threshold=0.5, higher_is_better=False)
    log_loss_validation_result = _MetricValidationResult("log_loss", 0.3, log_loss_threshold, None)

    l1_loss_threshold = MetricThreshold(threshold=0.3, higher_is_better=False)
    l1_loss_validation_result = _MetricValidationResult(
        "custom_l1_loss", 0.5, l1_loss_threshold, None
    )
    l1_loss_validation_result.threshold_failed = True

    if request.param == "single_metric_not_satisfied_higher_better":
        return ({"accuracy": 0.8}, {"accuracy": acc_threshold}, {"accuracy": acc_validation_result})

    if request.param == "multiple_metrics_not_satisfied_higher_better":
        return (
            {"accuracy": 0.8, "f1_score": 0.7},
            {"accuracy": acc_threshold, "f1_score": f1score_threshold},
            {"accuracy": acc_validation_result, "f1_score": f1score_validation_result},
        )

    if request.param == "single_metric_not_satisfied_lower_better":
        return (
            {"custom_l1_loss": 0.5},
            {"custom_l1_loss": l1_loss_threshold},
            {"custom_l1_loss": l1_loss_validation_result},
        )

    if request.param == "multiple_metrics_not_satisfied_lower_better":
        log_loss_validation_result.candidate_metric_value = 0.8
        log_loss_validation_result.threshold_failed = True
        return (
            {"custom_l1_loss": 0.5, "log_loss": 0.8},
            {"custom_l1_loss": l1_loss_threshold, "log_loss": log_loss_threshold},
            {"custom_l1_loss": l1_loss_validation_result, "log_loss": log_loss_validation_result},
        )

    if request.param == "missing_candidate_metric":
        acc_validation_result.missing_candidate = True
        return ({}, {"accuracy": acc_threshold}, {"accuracy": acc_validation_result})

    if request.param == "multiple_metrics_not_all_satisfied":
        return (
            {"accuracy": 0.8, "f1_score": 0.7, "log_loss": 0.3},
            {
                "accuracy": acc_threshold,
                "f1_score": f1score_threshold,
                "log_loss": log_loss_threshold,
            },
            {"accuracy": acc_validation_result, "f1_score": f1score_validation_result},
        )

    if request.param == "equality_boundary":
        return (
            {"accuracy": 0.9, "log_loss": 0.5},
            {"accuracy": acc_threshold, "log_loss": log_loss_threshold},
            {},
        )

    if request.param == "single_metric_satisfied_higher_better":
        return ({"accuracy": 0.91}, {"accuracy": acc_threshold}, {})

    if request.param == "single_metric_satisfied_lower_better":
        return ({"log_loss": 0.3}, {"log_loss": log_loss_threshold}, {})

    if request.param == "multiple_metrics_all_satisfied":
        return (
            {"accuracy": 0.9, "f1_score": 0.8, "log_loss": 0.3},
            {
                "accuracy": acc_threshold,
                "f1_score": f1score_threshold,
                "log_loss": log_loss_threshold,
            },
            {},
        )


@pytest.mark.parametrize(
    "value_threshold_test_spec",
    [
        ("single_metric_not_satisfied_higher_better"),
        ("multiple_metrics_not_satisfied_higher_better"),
        ("single_metric_not_satisfied_lower_better"),
        ("missing_candidate_metric"),
        ("multiple_metrics_not_satisfied_lower_better"),
        ("multiple_metrics_not_all_satisfied"),
    ],
    indirect=["value_threshold_test_spec"],
)
def test_validation_value_threshold_should_fail(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    value_threshold_test_spec,
):
    metrics, validation_thresholds, expected_validation_results = value_threshold_test_spec
    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        evaluator1_config = {}
        evaluator1_return_value = EvaluationResult(
            metrics=metrics, artifacts={}, baseline_model_metrics=None
        )
        expected_failure_message = message_separator.join(
            map(str, list(expected_validation_results.values()))
        )
        with mock.patch.object(
            MockEvaluator, "can_evaluate", return_value=True
        ) as _, mock.patch.object(
            MockEvaluator, "evaluate", return_value=evaluator1_return_value
        ) as _:
            with pytest.raises(
                ModelValidationFailedException,
                match=expected_failure_message,
            ):
                evaluate(
                    multiclass_logistic_regressor_model_uri,
                    data=iris_dataset._constructor_args["data"],
                    model_type="classifier",
                    targets=iris_dataset._constructor_args["targets"],
                    dataset_name=iris_dataset.name,
                    evaluators="test_evaluator1",
                    evaluator_config=evaluator1_config,
                    validation_thresholds=validation_thresholds,
                    baseline_model=None,
                )


@pytest.mark.parametrize(
    "value_threshold_test_spec",
    [
        ("single_metric_satisfied_higher_better"),
        ("single_metric_satisfied_lower_better"),
        ("equality_boundary"),
        ("multiple_metrics_all_satisfied"),
    ],
    indirect=["value_threshold_test_spec"],
)
def test_validation_value_threshold_should_pass(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    value_threshold_test_spec,
):
    metrics, validation_thresholds, _ = value_threshold_test_spec
    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        evaluator1_config = {}
        evaluator1_return_value = EvaluationResult(
            metrics=metrics, artifacts={}, baseline_model_metrics=None
        )
        with mock.patch.object(
            MockEvaluator, "can_evaluate", return_value=True
        ) as _, mock.patch.object(
            MockEvaluator, "evaluate", return_value=evaluator1_return_value
        ) as _:
            evaluate(
                multiclass_logistic_regressor_model_uri,
                data=iris_dataset._constructor_args["data"],
                model_type="classifier",
                targets=iris_dataset._constructor_args["targets"],
                dataset_name=iris_dataset.name,
                evaluators="test_evaluator1",
                evaluator_config=evaluator1_config,
                validation_thresholds=validation_thresholds,
                baseline_model=None,
            )


@pytest.fixture
def min_absolute_change_threshold_test_spec(request):
    """
    Test specification for min_absolute_change threshold tests:
    :return: (
                metrics: A dictionary mapping scalar metric names to scalar metric values,
                baseline_model_metrics: A dictionary mapping scalar metric names
                    to scalar metric values of baseline_model,
                validation_threhsolds: A dictonary mapping scalar metric names
                    to MetricThreshold(threshold=0.2, higher_is_better=True),
                expected_validation_results: A dictonary mapping scalar metric names
                    to _MetricValidationResult
             )
    """
    acc_threshold = MetricThreshold(min_absolute_change=0.1, higher_is_better=True)
    f1score_threshold = MetricThreshold(min_absolute_change=0.15, higher_is_better=True)
    log_loss_threshold = MetricThreshold(min_absolute_change=-0.1, higher_is_better=False)
    l1_loss_threshold = MetricThreshold(min_absolute_change=-0.15, higher_is_better=False)

    if request.param == "single_metric_not_satisfied_higher_better":
        acc_validation_result = _MetricValidationResult("accuracy", 0.79, acc_threshold, 0.7)
        acc_validation_result.min_absolute_change_failed = True
        return (
            {"accuracy": 0.79},
            {"accuracy": 0.7},
            {"accuracy": acc_threshold},
            {"accuracy": acc_validation_result},
        )

    if request.param == "multiple_metrics_not_satisfied_higher_better":
        acc_validation_result = _MetricValidationResult("accuracy", 0.79, acc_threshold, 0.7)
        acc_validation_result.min_absolute_change_failed = True
        f1score_validation_result = _MetricValidationResult("f1_score", 0.8, f1score_threshold, 0.7)
        f1score_validation_result.min_absolute_change_failed = True
        return (
            {"accuracy": 0.79, "f1_score": 0.8},
            {"accuracy": 0.7, "f1_score": 0.7},
            {"accuracy": acc_threshold, "f1_score": f1score_threshold},
            {"accuracy": acc_validation_result, "f1_score": f1score_validation_result},
        )

    if request.param == "single_metric_not_satisfied_lower_better":
        l1_loss_validation_result = _MetricValidationResult(
            "custom_l1_loss", 0.5, l1_loss_threshold, 0.6
        )
        l1_loss_validation_result.min_absolute_change_failed = True
        return (
            {"custom_l1_loss": 0.5},
            {"custom_l1_loss": 0.6},
            {"custom_l1_loss": l1_loss_threshold},
            {"custom_l1_loss": l1_loss_validation_result},
        )

    if request.param == "multiple_metrics_not_satisfied_lower_better":
        l1_loss_validation_result = _MetricValidationResult(
            "custom_l1_loss", 0.5, l1_loss_threshold, 0.6
        )
        l1_loss_validation_result.min_absolute_change_failed = True
        log_loss_validation_result = _MetricValidationResult(
            "log_loss", 0.45, log_loss_threshold, 0.3
        )
        log_loss_validation_result.min_absolute_change_failed = True
        return (
            {"custom_l1_loss": 0.5, "log_loss": 0.45},
            {"custom_l1_loss": 0.6, "log_loss": 0.3},
            {"custom_l1_loss": l1_loss_threshold, "log_loss": log_loss_threshold},
            {
                "custom_l1_loss": l1_loss_validation_result,
                "log_loss": log_loss_validation_result,
            },
        )

    if request.param == "equality_boundary":
        acc_validation_result = _MetricValidationResult("accuracy", 0.8, acc_threshold, 0.7)
        log_loss_validation_result = _MetricValidationResult(
            "custom_log_loss", 0.2, log_loss_threshold, 0.3
        )
        return (
            {"accuracy": 0.8 + 1e-10, "log_loss": 0.2 - 1e-10},
            {"accuracy": 0.7, "log_loss": 0.3},
            {"accuracy": acc_threshold, "log_loss": log_loss_threshold},
            {},
        )

    if request.param == "single_metric_satisfied_higher_better":
        return ({"accuracy": 0.9 + 1e-2}, {"accuracy": 0.8}, {"accuracy": acc_threshold}, {})

    if request.param == "single_metric_satisfied_lower_better":
        return ({"log_loss": 0.3}, {"log_loss": 0.4 + 1e-3}, {"log_loss": log_loss_threshold}, {})

    if request.param == "multiple_metrics_all_satisfied":
        return (
            {"accuracy": 0.9, "f1_score": 0.8, "log_loss": 0.3},
            {"accuracy": 0.7, "f1_score": 0.6, "log_loss": 0.5},
            {
                "accuracy": acc_threshold,
                "f1_score": f1score_threshold,
                "log_loss": log_loss_threshold,
            },
            {},
        )

    if request.param == "missing_baseline_metric":

        l1_loss_validation_result = _MetricValidationResult(
            "custom_l1_loss", 0.72, l1_loss_threshold, None
        )
        l1_loss_validation_result.missing_baseline = True
        return (
            {"custom_l1_loss": 0.72},
            None,
            {"custom_l1_loss": l1_loss_threshold},
            {"custom_l1_loss": l1_loss_validation_result},
        )


@pytest.mark.parametrize(
    "min_absolute_change_threshold_test_spec",
    [
        ("single_metric_not_satisfied_higher_better"),
        ("multiple_metrics_not_satisfied_higher_better"),
        ("single_metric_not_satisfied_lower_better"),
        ("multiple_metrics_not_satisfied_lower_better"),
        ("missing_baseline_metric"),
    ],
    indirect=["min_absolute_change_threshold_test_spec"],
)
def test_validation_model_comparison_absolute_threshold_should_fail(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    min_absolute_change_threshold_test_spec,
):
    (
        metrics,
        baseline_model_metrics,
        validation_thresholds,
        expected_validation_results,
    ) = min_absolute_change_threshold_test_spec

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        evaluator1_config = {}
        evaluator1_return_value = EvaluationResult(
            metrics=metrics, artifacts={}, baseline_model_metrics=baseline_model_metrics
        )
        expected_failure_message = message_separator.join(
            map(str, list(expected_validation_results.values()))
        )
        with mock.patch.object(
            MockEvaluator, "can_evaluate", return_value=True
        ) as _, mock.patch.object(
            MockEvaluator, "evaluate", return_value=evaluator1_return_value
        ) as _:
            with pytest.raises(
                ModelValidationFailedException,
                match=expected_failure_message,
            ):
                evaluate(
                    multiclass_logistic_regressor_model_uri,
                    data=iris_dataset._constructor_args["data"],
                    model_type="classifier",
                    targets=iris_dataset._constructor_args["targets"],
                    dataset_name=iris_dataset.name,
                    evaluators="test_evaluator1",
                    evaluator_config=evaluator1_config,
                    validation_thresholds=validation_thresholds,
                    baseline_model=multiclass_logistic_regressor_model_uri,
                )


@pytest.mark.parametrize(
    "min_absolute_change_threshold_test_spec",
    [
        ("single_metric_satisfied_higher_better"),
        ("single_metric_satisfied_lower_better"),
        ("equality_boundary"),
        ("multiple_metrics_all_satisfied"),
    ],
    indirect=["min_absolute_change_threshold_test_spec"],
)
def test_validation_model_comparison_absolute_threshold_should_pass(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    min_absolute_change_threshold_test_spec,
):
    (
        metrics,
        baseline_model_metrics,
        validation_thresholds,
        _,
    ) = min_absolute_change_threshold_test_spec
    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        evaluator1_config = {}
        evaluator1_return_value = EvaluationResult(
            metrics=metrics, artifacts={}, baseline_model_metrics=baseline_model_metrics
        )
        with mock.patch.object(
            MockEvaluator, "can_evaluate", return_value=True
        ) as _, mock.patch.object(
            MockEvaluator, "evaluate", return_value=evaluator1_return_value
        ) as _:
            evaluate(
                multiclass_logistic_regressor_model_uri,
                data=iris_dataset._constructor_args["data"],
                model_type="classifier",
                targets=iris_dataset._constructor_args["targets"],
                dataset_name=iris_dataset.name,
                evaluators="test_evaluator1",
                evaluator_config=evaluator1_config,
                validation_thresholds=validation_thresholds,
                baseline_model=multiclass_logistic_regressor_model_uri,
            )


@pytest.fixture
def min_relative_change_threshold_test_spec(request):
    """
    Test specification for min_relative_change threshold tests:
    :return: (
                metrics: A dictionary mapping scalar metric names to scalar metric values,
                baseline_model_metrics: A dictionary mapping scalar metric names
                    to scalar metric values of baseline_model,
                validation_threhsolds: A dictonary mapping scalar metric names
                    to MetricThreshold(threshold=0.2, higher_is_better=True),
                expected_validation_results: A dictonary mapping scalar metric names
                    to _MetricValidationResult
             )
    """
    acc_threshold = MetricThreshold(min_relative_change=0.1, higher_is_better=True)
    f1score_threshold = MetricThreshold(min_relative_change=0.15, higher_is_better=True)
    log_loss_threshold = MetricThreshold(min_relative_change=0.15, higher_is_better=False)
    l1_loss_threshold = MetricThreshold(min_relative_change=0.1, higher_is_better=False)

    if request.param == "single_metric_not_satisfied_higher_better":
        acc_validation_result = _MetricValidationResult("accuracy", 0.75, acc_threshold, 0.7)
        acc_validation_result.min_relative_change_failed = True
        return (
            {"accuracy": 0.75},
            {"accuracy": 0.7},
            {"accuracy": acc_threshold},
            {"accuracy": acc_validation_result},
        )

    if request.param == "multiple_metrics_not_satisfied_higher_better":
        acc_validation_result = _MetricValidationResult("accuracy", 0.53, acc_threshold, 0.5)
        acc_validation_result.min_relative_change_failed = True
        f1score_validation_result = _MetricValidationResult("f1_score", 0.8, f1score_threshold, 0.7)
        f1score_validation_result.min_relative_change_failed = True
        return (
            {"accuracy": 0.53, "f1_score": 0.8},
            {"accuracy": 0.5, "f1_score": 0.7},
            {"accuracy": acc_threshold, "f1_score": f1score_threshold},
            {"accuracy": acc_validation_result, "f1_score": f1score_validation_result},
        )

    if request.param == "single_metric_not_satisfied_lower_better":
        l1_loss_validation_result = _MetricValidationResult(
            "custom_l1_loss", 0.55, l1_loss_threshold, 0.6
        )
        l1_loss_validation_result.min_relative_change_failed = True
        return (
            {"custom_l1_loss": 0.55},
            {"custom_l1_loss": 0.6},
            {"custom_l1_loss": l1_loss_threshold},
            {"custom_l1_loss": l1_loss_validation_result},
        )

    if request.param == "missing_baseline_metric":

        l1_loss_validation_result = _MetricValidationResult(
            "custom_l1_loss", 0.72, l1_loss_threshold, None
        )
        l1_loss_validation_result.missing_baseline = True
        return (
            {"custom_l1_loss": 0.72},
            None,
            {"custom_l1_loss": l1_loss_threshold},
            {"custom_l1_loss": l1_loss_validation_result},
        )

    if request.param == "multiple_metrics_not_satisfied_lower_better":
        l1_loss_validation_result = _MetricValidationResult(
            "custom_l1_loss", 0.72 + 1e-3, l1_loss_threshold, 0.8
        )
        l1_loss_validation_result.min_relative_change_failed = True
        log_loss_validation_result = _MetricValidationResult(
            "log_loss", 0.27 + 1e-5, log_loss_threshold, 0.3
        )
        log_loss_validation_result.min_relative_change_failed = True
        return (
            {"custom_l1_loss": 0.72 + 1e-3, "log_loss": 0.27 + 1e-5},
            {"custom_l1_loss": 0.8, "log_loss": 0.3},
            {"custom_l1_loss": l1_loss_threshold, "log_loss": log_loss_threshold},
            {
                "custom_l1_loss": l1_loss_validation_result,
                "log_loss": log_loss_validation_result,
            },
        )

    if request.param == "equality_boundary":
        acc_validation_result = _MetricValidationResult("accuracy", 0.77, acc_threshold, 0.7)
        log_loss_validation_result = _MetricValidationResult(
            "custom_log_loss", 0.3 * 0.85 - 1e-10, log_loss_threshold, 0.3
        )
        return (
            {"accuracy": 0.77, "log_loss": 0.3 * 0.85 - 1e-10},
            {"accuracy": 0.7, "log_loss": 0.3},
            {"accuracy": acc_threshold, "log_loss": log_loss_threshold},
            {},
        )

    if request.param == "single_metric_satisfied_higher_better":
        return ({"accuracy": 0.99 + 1e-10}, {"accuracy": 0.9}, {"accuracy": acc_threshold}, {})

    if request.param == "single_metric_satisfied_lower_better":
        return ({"log_loss": 0.3}, {"log_loss": 0.4}, {"log_loss": log_loss_threshold}, {})

    if request.param == "multiple_metrics_all_satisfied":
        return (
            {"accuracy": 0.9, "f1_score": 0.9, "log_loss": 0.3},
            {"accuracy": 0.7, "f1_score": 0.6, "log_loss": 0.5},
            {
                "accuracy": acc_threshold,
                "f1_score": f1score_threshold,
                "log_loss": log_loss_threshold,
            },
            {},
        )


@pytest.mark.parametrize(
    "min_relative_change_threshold_test_spec",
    [
        ("single_metric_not_satisfied_higher_better"),
        ("multiple_metrics_not_satisfied_higher_better"),
        ("single_metric_not_satisfied_lower_better"),
        ("multiple_metrics_not_satisfied_lower_better"),
        ("missing_baseline_metric"),
    ],
    indirect=["min_relative_change_threshold_test_spec"],
)
def test_validation_model_comparison_relative_threshold_should_fail(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    min_relative_change_threshold_test_spec,
):
    (
        metrics,
        baseline_model_metrics,
        validation_thresholds,
        expected_validation_results,
    ) = min_relative_change_threshold_test_spec

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        evaluator1_config = {}
        evaluator1_return_value = EvaluationResult(
            metrics=metrics, artifacts={}, baseline_model_metrics=baseline_model_metrics
        )
        expected_failure_message = message_separator.join(
            map(str, list(expected_validation_results.values()))
        )
        with mock.patch.object(
            MockEvaluator, "can_evaluate", return_value=True
        ) as _, mock.patch.object(
            MockEvaluator, "evaluate", return_value=evaluator1_return_value
        ) as _:
            with pytest.raises(
                ModelValidationFailedException,
                match=expected_failure_message,
            ):
                evaluate(
                    multiclass_logistic_regressor_model_uri,
                    data=iris_dataset._constructor_args["data"],
                    model_type="classifier",
                    targets=iris_dataset._constructor_args["targets"],
                    dataset_name=iris_dataset.name,
                    evaluators="test_evaluator1",
                    evaluator_config=evaluator1_config,
                    validation_thresholds=validation_thresholds,
                    baseline_model=multiclass_logistic_regressor_model_uri,
                )


@pytest.mark.parametrize(
    "min_relative_change_threshold_test_spec",
    [
        ("single_metric_satisfied_higher_better"),
        ("single_metric_satisfied_lower_better"),
        ("equality_boundary"),
        ("multiple_metrics_all_satisfied"),
    ],
    indirect=["min_relative_change_threshold_test_spec"],
)
def test_validation_model_comparison_relative_threshold_should_pass(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    min_relative_change_threshold_test_spec,
):
    (
        metrics,
        baseline_model_metrics,
        validation_thresholds,
        _,
    ) = min_relative_change_threshold_test_spec
    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        evaluator1_config = {}
        evaluator1_return_value = EvaluationResult(
            metrics=metrics, artifacts={}, baseline_model_metrics=baseline_model_metrics
        )
        with mock.patch.object(
            MockEvaluator, "can_evaluate", return_value=True
        ) as _, mock.patch.object(
            MockEvaluator, "evaluate", return_value=evaluator1_return_value
        ) as _:
            evaluate(
                multiclass_logistic_regressor_model_uri,
                data=iris_dataset._constructor_args["data"],
                model_type="classifier",
                targets=iris_dataset._constructor_args["targets"],
                dataset_name=iris_dataset.name,
                evaluators="test_evaluator1",
                evaluator_config=evaluator1_config,
                validation_thresholds=validation_thresholds,
                baseline_model=multiclass_logistic_regressor_model_uri,
            )


@pytest.fixture
def multi_thresholds_test_spec(request):
    """
    Test specification for multi-thresholds tests:
    :return: (
                metrics: A dictionary mapping scalar metric names to scalar metric values,
                baseline_model_metrics: A dictionary mapping scalar metric names
                    to scalar metric values of baseline_model,
                validation_threhsolds: A dictonary mapping scalar metric names
                    to MetricThreshold(threshold=0.2, higher_is_better=True),
                expected_validation_results: A dictonary mapping scalar metric names
                    to _MetricValidationResult
             )
    """
    acc_threshold = MetricThreshold(
        threshold=0.8, min_absolute_change=0.1, min_relative_change=0.1, higher_is_better=True
    )

    if request.param == "single_metric_all_thresholds_failed":
        acc_validation_result = _MetricValidationResult("accuracy", 0.75, acc_threshold, 0.7)
        acc_validation_result.threshold_failed = True
        acc_validation_result.min_relative_change_failed = True
        acc_validation_result.min_absolute_change_failed = True
        return (
            {"accuracy": 0.75},
            {"accuracy": 0.7},
            {"accuracy": acc_threshold},
            {"accuracy": acc_validation_result},
        )


@pytest.mark.parametrize(
    "multi_thresholds_test_spec",
    [
        ("single_metric_all_thresholds_failed"),
    ],
    indirect=["multi_thresholds_test_spec"],
)
def test_validation_multi_thresholds_should_fail(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    multi_thresholds_test_spec,
):
    (
        metrics,
        baseline_model_metrics,
        validation_thresholds,
        expected_validation_results,
    ) = multi_thresholds_test_spec

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        evaluator1_config = {}
        evaluator1_return_value = EvaluationResult(
            metrics=metrics, artifacts={}, baseline_model_metrics=baseline_model_metrics
        )
        expected_failure_message = message_separator.join(
            map(str, list(expected_validation_results.values()))
        )
        with mock.patch.object(
            MockEvaluator, "can_evaluate", return_value=True
        ) as _, mock.patch.object(
            MockEvaluator, "evaluate", return_value=evaluator1_return_value
        ) as _:
            with pytest.raises(
                ModelValidationFailedException,
                match=expected_failure_message,
            ):
                evaluate(
                    multiclass_logistic_regressor_model_uri,
                    data=iris_dataset._constructor_args["data"],
                    model_type="classifier",
                    targets=iris_dataset._constructor_args["targets"],
                    dataset_name=iris_dataset.name,
                    evaluators="test_evaluator1",
                    evaluator_config=evaluator1_config,
                    validation_thresholds=validation_thresholds,
                    baseline_model=multiclass_logistic_regressor_model_uri,
                )
