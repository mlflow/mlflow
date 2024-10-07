import random
from unittest import mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation import (
    EvaluationResult,
    MetricThreshold,
    ModelEvaluator,
    evaluate,
)
from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry
from mlflow.models.evaluation.validation import (
    MetricThresholdClassException,
    ModelValidationFailedException,
    _MetricValidationResult,
)

from tests.evaluate.test_evaluation import (
    iris_dataset,  # noqa: F401
    multiclass_logistic_regressor_model_uri,  # noqa: F401
)

message_separator = "\n"


@pytest.fixture
def metric_threshold_class_test_spec(request):
    """
    Test specification for MetricThreshold class:

    Returns:
        A tuple containing the following elements:

        - class_params: A dictionary mapping MetricThreshold class parameter names to values.
        - expected_failure_message: Expected failure message.
    """
    class_params = {
        "threshold": 1,
        "min_absolute_change": 1,
        "min_relative_change": 0.1,
        "greater_is_better": True,
    }

    if request.param == "threshold_is_not_number":
        class_params["threshold"] = "string"
        expected_failure_message = "`threshold` parameter must be a number."
    if request.param == "min_absolute_change_is_not_number":
        class_params["min_absolute_change"] = "string"
        expected_failure_message = "`min_absolute_change` parameter must be a positive number."
    elif request.param == "min_absolute_change_is_not_positive":
        class_params["min_absolute_change"] = -1
        expected_failure_message = "`min_absolute_change` parameter must be a positive number."
    elif request.param == "min_relative_change_is_not_float":
        class_params["min_relative_change"] = 2
        expected_failure_message = (
            "`min_relative_change` parameter must be a floating point number."
        )
    elif request.param == "min_relative_change_is_not_between_0_and_1":
        class_params["min_relative_change"] = -0.1
        expected_failure_message = "`min_relative_change` parameter must be between 0 and 1."
    elif request.param == "greater_is_better_is_not_defined":
        class_params["greater_is_better"] = None
        expected_failure_message = "`greater_is_better` parameter must be defined."
    elif request.param == "greater_is_better_is_not_bool":
        class_params["greater_is_better"] = 1
        expected_failure_message = "`greater_is_better` parameter must be a boolean."
    elif request.param == "no_threshold":
        class_params["threshold"] = None
        class_params["min_absolute_change"] = None
        class_params["min_relative_change"] = None
        expected_failure_message = "no threshold was specified."

    return (class_params, expected_failure_message)


@pytest.mark.parametrize(
    "metric_threshold_class_test_spec",
    [
        ("threshold_is_not_number"),
        ("min_absolute_change_is_not_number"),
        ("min_absolute_change_is_not_positive"),
        ("min_relative_change_is_not_float"),
        ("min_relative_change_is_not_between_0_and_1"),
        ("greater_is_better_is_not_defined"),
        ("greater_is_better_is_not_bool"),
        ("no_threshold"),
    ],
    indirect=["metric_threshold_class_test_spec"],
)
def test_metric_threshold_class_should_fail(metric_threshold_class_test_spec):
    class_params, expected_failure_message = metric_threshold_class_test_spec
    with pytest.raises(
        MetricThresholdClassException,
        match=expected_failure_message,
    ):
        MetricThreshold(
            threshold=class_params["threshold"],
            min_absolute_change=class_params["min_absolute_change"],
            min_relative_change=class_params["min_relative_change"],
            greater_is_better=class_params["greater_is_better"],
        )


@pytest.fixture
def faulty_baseline_model_param_test_spec(request):
    """
    Test specification for faulty `baseline_model` parameter tests:

    Returns:
        A dict containing the following elements:

        - validation_thresholds: A dictionary mapping scalar metric names to
          MetricThreshold(threshold=0.2, greater_is_better=True).
        - baseline_model: Value for the `baseline_model` param passed into mlflow.evaluate().
        - expected_failure_message: Expected failure message.
    """
    if request.param == "min_relative_change_present":
        return (
            {"accuracy": MetricThreshold(min_absolute_change=0.1, greater_is_better=True)},
            None,
            "The baseline model must be specified",
        )
    if request.param == "min_absolute_change_present":
        return (
            {"accuracy": MetricThreshold(min_relative_change=0.1, greater_is_better=True)},
            None,
            "The baseline model must be specified",
        )
    if request.param == "both_relative_absolute_change_present":
        return (
            {
                "accuracy": MetricThreshold(
                    min_absolute_change=0.05, min_relative_change=0.1, greater_is_better=True
                )
            },
            None,
            "The baseline model must be specified",
        )
    if request.param == "baseline_model_is_not_string":
        return (
            {
                "accuracy": MetricThreshold(
                    min_absolute_change=0.05, min_relative_change=0.1, greater_is_better=True
                )
            },
            1.0,
            "The baseline model argument must be a string URI",
        )


@pytest.mark.parametrize(
    "validation_thresholds",
    [
        pytest.param(1, id="param_not_dict"),
        pytest.param(
            {1: MetricThreshold(min_absolute_change=0.1, greater_is_better=True)}, id="key_not_str"
        ),
        pytest.param({"accuracy": 1}, id="value_not_metric_threshold"),
    ],
)
def test_validation_faulty_validation_thresholds(validation_thresholds):
    with pytest.raises(MlflowException, match="The validation thresholds argument"):
        mlflow.validate_evaluation_results(
            candidate_result={},
            baseline_result={},
            validation_thresholds=validation_thresholds,
        )


@pytest.fixture
def value_threshold_test_spec(request):
    """
    Test specification for value threshold tests:

    Returns:
        A dict containing the following elements:

        - metrics: A dictionary mapping scalar metric names to scalar metric values.
        - validation_thresholds: A dictionary mapping scalar metric names to
          MetricThreshold(threshold=0.2, greater_is_better=True).
        - expected_validation_results: A dictionary mapping scalar metric names
          to _MetricValidationResult.
    """
    acc_threshold = MetricThreshold(threshold=0.9, greater_is_better=True)
    acc_validation_result = _MetricValidationResult("accuracy", 0.8, acc_threshold, None)
    acc_validation_result.threshold_failed = True

    f1score_threshold = MetricThreshold(threshold=0.8, greater_is_better=True)
    f1score_validation_result = _MetricValidationResult("f1_score", 0.7, f1score_threshold, None)
    f1score_validation_result.threshold_failed = True

    log_loss_threshold = MetricThreshold(threshold=0.5, greater_is_better=False)
    log_loss_validation_result = _MetricValidationResult("log_loss", 0.3, log_loss_threshold, None)

    l1_loss_threshold = MetricThreshold(threshold=0.3, greater_is_better=False)
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

    MockEvaluator = mock.MagicMock(spec=ModelEvaluator)
    MockEvaluator().can_evaluate.return_value = True
    MockEvaluator().evaluate.return_value = EvaluationResult(metrics=metrics, artifacts={})

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        candidate_result = evaluate(
            multiclass_logistic_regressor_model_uri,
            data=iris_dataset._constructor_args["data"],
            model_type="classifier",
            targets=iris_dataset._constructor_args["targets"],
            evaluators="test_evaluator1",
        )

    with pytest.raises(
        ModelValidationFailedException,
        match=message_separator.join(map(str, list(expected_validation_results.values()))),
    ):
        mlflow.validate_evaluation_results(
            candidate_result=candidate_result,
            baseline_result=None,
            validation_thresholds=validation_thresholds,
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

    MockEvaluator = mock.MagicMock(spec=ModelEvaluator)
    MockEvaluator().can_evaluate.return_value = True
    MockEvaluator().evaluate.return_value = EvaluationResult(metrics=metrics, artifacts={})

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        candidate_result = evaluate(
            multiclass_logistic_regressor_model_uri,
            data=iris_dataset._constructor_args["data"],
            model_type="classifier",
            targets=iris_dataset._constructor_args["targets"],
            evaluators="test_evaluator1",
        )

    mlflow.validate_evaluation_results(
        candidate_result=candidate_result,
        baseline_result=None,
        validation_thresholds=validation_thresholds,
    )


@pytest.fixture
def min_absolute_change_threshold_test_spec(request):
    """
    Test specification for min_absolute_change threshold tests:

    Returns:
        A dict containing the following elements:

        - metrics: A dictionary mapping scalar metric names to scalar metric values.
        - baseline_model_metrics: A dictionary mapping scalar metric names
            to scalar metric values of baseline_model.
        - validation_thresholds: A dictonary mapping scalar metric names
            to MetricThreshold(threshold=0.2, greater_is_better=True).
        - expected_validation_results: A dictonary mapping scalar metric names
            to _MetricValidationResult.
    """
    acc_threshold = MetricThreshold(min_absolute_change=0.1, greater_is_better=True)
    f1score_threshold = MetricThreshold(min_absolute_change=0.15, greater_is_better=True)
    log_loss_threshold = MetricThreshold(min_absolute_change=0.1, greater_is_better=False)
    l1_loss_threshold = MetricThreshold(min_absolute_change=0.15, greater_is_better=False)

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

    MockEvaluator = mock.MagicMock(spec=ModelEvaluator)
    MockEvaluator().can_evaluate.return_value = True
    mock_evaluate = MockEvaluator().evaluate

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        common_kwargs = {
            "data": iris_dataset._constructor_args["data"],
            "model_type": "classifier",
            "targets": iris_dataset._constructor_args["targets"],
            "evaluators": "test_evaluator1",
        }

        mock_evaluate.return_value = EvaluationResult(metrics=metrics, artifacts={})
        candidate_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

        if baseline_model_metrics is None:
            baseline_result = None
        else:
            mock_evaluate.return_value = EvaluationResult(
                metrics=baseline_model_metrics, artifacts={}
            )
            baseline_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

    with pytest.raises(
        ModelValidationFailedException,
        match=message_separator.join(map(str, list(expected_validation_results.values()))),
    ):
        mlflow.validate_evaluation_results(
            candidate_result=candidate_result,
            baseline_result=baseline_result,
            validation_thresholds=validation_thresholds,
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

    MockEvaluator = mock.MagicMock(spec=ModelEvaluator)
    MockEvaluator().can_evaluate.return_value = True
    mock_evaluate = MockEvaluator().evaluate

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        common_kwargs = {
            "data": iris_dataset._constructor_args["data"],
            "model_type": "classifier",
            "targets": iris_dataset._constructor_args["targets"],
            "evaluators": "test_evaluator1",
        }

        mock_evaluate.return_value = EvaluationResult(metrics=metrics, artifacts={})
        candidate_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

        mock_evaluate.return_value = EvaluationResult(metrics=baseline_model_metrics, artifacts={})
        baseline_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

    mlflow.validate_evaluation_results(
        candidate_result=candidate_result,
        baseline_result=baseline_result,
        validation_thresholds=validation_thresholds,
    )


@pytest.fixture
def min_relative_change_threshold_test_spec(request):
    """
    Test specification for min_relative_change threshold tests:

    Returns:
        A dict with the following elements:

        - metrics: A dictionary mapping scalar metric names to scalar metric values.
        - baseline_model_metrics: A dictionary mapping scalar metric names
            to scalar metric values of baseline_model.
        - validation_thresholds: A dictionary mapping scalar metric names
            to MetricThreshold(threshold=0.2, greater_is_better=True).
        - expected_validation_results: A dictionary mapping scalar metric names
            to _MetricValidationResult.
    """
    acc_threshold = MetricThreshold(min_relative_change=0.1, greater_is_better=True)
    f1score_threshold = MetricThreshold(min_relative_change=0.15, greater_is_better=True)
    log_loss_threshold = MetricThreshold(min_relative_change=0.15, greater_is_better=False)
    l1_loss_threshold = MetricThreshold(min_relative_change=0.1, greater_is_better=False)

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

    if request.param == "baseline_metric_value_equals_0_succeeds":
        threshold = MetricThreshold(min_relative_change=0.1, greater_is_better=True)
        return (
            {"metric_1": 1e-10},
            {"metric_1": 0},
            {"metric_1": threshold},
            {"metric_1": _MetricValidationResult("metric_1", 0.8, threshold, 0.7)},
        )

    if request.param == "baseline_metric_value_equals_0_fails":
        metric_1_threshold = MetricThreshold(min_relative_change=0.1, greater_is_better=True)
        metric_1_result = _MetricValidationResult("metric_1", 0, metric_1_threshold, 0)
        metric_1_result.min_relative_change_failed = True
        return (
            {"metric_1": 0},
            {"metric_1": 0},
            {"metric_1": metric_1_threshold},
            {"metric_1": metric_1_result},
        )


@pytest.mark.parametrize(
    "min_relative_change_threshold_test_spec",
    [
        ("single_metric_not_satisfied_higher_better"),
        ("multiple_metrics_not_satisfied_higher_better"),
        ("single_metric_not_satisfied_lower_better"),
        ("multiple_metrics_not_satisfied_lower_better"),
        ("missing_baseline_metric"),
        ("baseline_metric_value_equals_0_fails"),
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

    MockEvaluator = mock.MagicMock(spec=ModelEvaluator)
    MockEvaluator().can_evaluate.return_value = True
    mock_evaluate = MockEvaluator().evaluate

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        common_kwargs = {
            "data": iris_dataset._constructor_args["data"],
            "model_type": "classifier",
            "targets": iris_dataset._constructor_args["targets"],
            "evaluators": "test_evaluator1",
        }

        mock_evaluate.return_value = EvaluationResult(metrics=metrics, artifacts={})
        candidate_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

        if baseline_model_metrics is None:
            baseline_result = None
        else:
            mock_evaluate.return_value = EvaluationResult(
                metrics=baseline_model_metrics, artifacts={}
            )
            baseline_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

        with pytest.raises(
            ModelValidationFailedException,
            match=message_separator.join(map(str, list(expected_validation_results.values()))),
        ):
            mlflow.validate_evaluation_results(
                candidate_result=candidate_result,
                baseline_result=baseline_result,
                validation_thresholds=validation_thresholds,
            )


@pytest.mark.parametrize(
    "min_relative_change_threshold_test_spec",
    [
        ("single_metric_satisfied_higher_better"),
        ("single_metric_satisfied_lower_better"),
        ("equality_boundary"),
        ("multiple_metrics_all_satisfied"),
        ("baseline_metric_value_equals_0_succeeds"),
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

    MockEvaluator = mock.MagicMock(spec=ModelEvaluator)
    MockEvaluator().can_evaluate.return_value = True
    mock_evaluate = MockEvaluator().evaluate

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        common_kwargs = {
            "data": iris_dataset._constructor_args["data"],
            "model_type": "classifier",
            "targets": iris_dataset._constructor_args["targets"],
            "evaluators": "test_evaluator1",
        }

        mock_evaluate.return_value = EvaluationResult(metrics=metrics, artifacts={})
        candidate_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

        mock_evaluate.return_value = EvaluationResult(metrics=baseline_model_metrics, artifacts={})
        baseline_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

    mlflow.validate_evaluation_results(
        candidate_result=candidate_result,
        baseline_result=baseline_result,
        validation_thresholds=validation_thresholds,
    )


@pytest.fixture
def multi_thresholds_test_spec(request):
    """
    Test specification for multi-thresholds tests:

    Returns:
        A dict with the following elements:

        - metrics: A dictionary mapping scalar metric names to scalar metric values.
        - baseline_model_metrics: A dictionary mapping scalar metric names
            to scalar metric values of baseline_model.
        - validation_thresholds: A dictonary mapping scalar metric names
            to MetricThreshold(threshold=0.2, greater_is_better=True).
        - expected_validation_results: A dictonary mapping scalar metric names
            to _MetricValidationResult.
    """
    acc_threshold = MetricThreshold(
        threshold=0.8, min_absolute_change=0.1, min_relative_change=0.1, greater_is_better=True
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

    MockEvaluator = mock.MagicMock(spec=ModelEvaluator)
    MockEvaluator().can_evaluate.return_value = True
    mock_evaluate = MockEvaluator().evaluate

    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": MockEvaluator}
    ):
        common_kwargs = {
            "data": iris_dataset._constructor_args["data"],
            "model_type": "classifier",
            "targets": iris_dataset._constructor_args["targets"],
            "evaluators": "test_evaluator1",
        }

        mock_evaluate.return_value = EvaluationResult(metrics=metrics, artifacts={})
        candidate_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

        mock_evaluate.return_value = EvaluationResult(metrics=baseline_model_metrics, artifacts={})
        baseline_result = evaluate(multiclass_logistic_regressor_model_uri, **common_kwargs)

    with pytest.raises(
        ModelValidationFailedException,
        match=message_separator.join(map(str, list(expected_validation_results.values()))),
    ):
        mlflow.validate_evaluation_results(
            candidate_result=candidate_result,
            baseline_result=baseline_result,
            validation_thresholds=validation_thresholds,
        )


def test_validation_thresholds_no_mock():
    targets = [0, 1, 1, 1]
    data = [[random.random()] for _ in targets]

    class BaseModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return len(model_input) * [0]

    class CandidateModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return len(model_input) * [1]

    with mlflow.start_run():
        base = mlflow.pyfunc.log_model("base", python_model=BaseModel())
        candidate = mlflow.pyfunc.log_model("candidate", python_model=CandidateModel())

        candidate_result = evaluate(
            candidate.model_uri,
            data=data,
            model_type="classifier",
            targets=targets,
        )

        baseline_result = evaluate(
            base.model_uri,
            data=data,
            model_type="classifier",
            targets=targets,
        )

    mlflow.validate_evaluation_results(
        candidate_result=candidate_result,
        baseline_result=baseline_result,
        validation_thresholds={
            "recall_score": MetricThreshold(
                threshold=0.9,
                min_absolute_change=0.1,
                greater_is_better=True,
            ),
        },
    )

    with pytest.raises(
        ModelValidationFailedException,
        match="recall_score value threshold check failed",
    ):
        mlflow.validate_evaluation_results(
            candidate_result=baseline_result,
            baseline_result=candidate_result,
            validation_thresholds={
                "recall_score": MetricThreshold(
                    threshold=0.9,
                    min_absolute_change=0.1,
                    greater_is_better=True,
                ),
            },
        )


def test_legacy_validation_within_evaluate():
    # Test legacy validation within mlflow.evaluate(). This is deprecated
    # in favor of the new mlflow.mlflow.validate_evaluation_results API but we
    # keep backward compatibility until it is entirely removed.
    targets = [0, 1, 1, 1]
    data = [[random.random()] for _ in targets]

    class BaseModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return len(model_input) * [0]

    class CandidateModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return len(model_input) * [1]

    with mlflow.start_run():
        base = mlflow.pyfunc.log_model("base", python_model=BaseModel())
        candidate = mlflow.pyfunc.log_model("candidate", python_model=CandidateModel())

    with mlflow.start_run():
        evaluate(
            candidate.model_uri,
            data=data,
            model_type="classifier",
            targets=targets,
            validation_thresholds={
                "recall_score": MetricThreshold(
                    threshold=0.9,
                    min_absolute_change=0.1,
                    greater_is_better=True,
                ),
            },
            baseline_model=base.model_uri,
        )

    with pytest.raises(
        ModelValidationFailedException,
        match="recall_score value threshold check failed",
    ):
        with mlflow.start_run():
            evaluate(
                base.model_uri,
                data=data,
                model_type="classifier",
                targets=targets,
                validation_thresholds={
                    "recall_score": MetricThreshold(
                        threshold=0.9,
                        min_absolute_change=0.1,
                        greater_is_better=True,
                    ),
                },
                baseline_model=candidate.model_uri,
            )
