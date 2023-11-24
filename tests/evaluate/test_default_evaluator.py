from __future__ import annotations

import io
import json
from os.path import join as path_join
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageChops
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
    MetricValue,
    make_metric,
)
from mlflow.metrics.genai import model_utils
from mlflow.models import Model
from mlflow.models.evaluation.artifacts import (
    CsvEvaluationArtifact,
    ImageEvaluationArtifact,
    JsonEvaluationArtifact,
    NumpyEvaluationArtifact,
    ParquetEvaluationArtifact,
    PickleEvaluationArtifact,
    TextEvaluationArtifact,
)
from mlflow.models.evaluation.base import evaluate
from mlflow.models.evaluation.default_evaluator import (
    _compute_df_mode_or_mean,
    _CustomArtifact,
    _CustomMetric,
    _evaluate_custom_artifacts,
    _evaluate_extra_metric,
    _extract_output_and_other_columns,
    _extract_predict_fn,
    _extract_raw_model,
    _gen_classifier_curve,
    _get_aggregate_metrics_values,
    _get_binary_classifier_metrics,
    _get_binary_sum_up_label_pred_prob,
    _get_multiclass_classifier_metrics,
    _get_regressor_metrics,
    _infer_model_type_by_labels,
)

from tests.evaluate.test_evaluation import (
    baseline_model_uri,  # noqa: F401
    binary_logistic_regressor_model_uri,  # noqa: F401
    breast_cancer_dataset,  # noqa: F401
    diabetes_dataset,  # noqa: F401
    diabetes_spark_dataset,  # noqa: F401
    get_pipeline_model_dataset,
    get_run_data,
    iris_dataset,  # noqa: F401
    iris_pandas_df_dataset,  # noqa: F401
    iris_pandas_df_num_cols_dataset,  # noqa: F401
    linear_regressor_model_uri,  # noqa: F401
    multiclass_logistic_regressor_model_uri,  # noqa: F401
    pipeline_model_uri,  # noqa: F401
    spark_linear_regressor_model_uri,  # noqa: F401
    svm_model_uri,  # noqa: F401
)


def assert_dict_equal(d1, d2, rtol):
    for k in d1:
        assert k in d2
        assert np.isclose(d1[k], d2[k], rtol=rtol)


def evaluate_model_helper(
    model,
    baseline_model,
    data,
    targets,
    model_type: str,
    evaluators=None,
    evaluator_config=None,
    eval_baseline_model_only=False,
):
    """
    Helper function for testing mlflow.evaluate
    To test if evaluation for baseline model does not log metrics and artifacts;
    we set "disable_candidate_model" to true for the evaluator_config so that the
    DefaultEvaluator will evaluate only the baseline_model with logging
    disabled. This code path is only for testing purposes.
    """
    if eval_baseline_model_only:
        if not evaluator_config:
            evaluator_config = {"_disable_candidate_model": True}
        elif not evaluators or evaluators == "default":
            evaluator_config.update({"_disable_candidate_model": True})
        else:
            for config in evaluator_config.values():
                config.update({"_disable_candidate_model": True})

    return evaluate(
        model=model,
        data=data,
        model_type=model_type,
        targets=targets,
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        baseline_model=baseline_model,
    )


def check_metrics_not_logged_for_baseline_model_evaluation(
    logged_metrics, result_metrics, expected_metrics
):
    """
    Helper function for checking metrics of evaluation of baseline_model
     - Metrics should not be logged
     - Metrics should be returned in EvaluationResult as expected
    """
    assert logged_metrics == {}
    for metric_key in expected_metrics:
        assert np.isclose(expected_metrics[metric_key], result_metrics[metric_key], rtol=1e-3)


def check_artifacts_are_not_generated_for_baseline_model_evaluation(
    logged_artifacts, result_artifacts
):
    """
    Helper function for unit tests for checking artifacts of evaluation of baseline model
        - No Artifact is returned nor logged
    """
    assert logged_artifacts == []
    assert result_artifacts == {}


@pytest.mark.parametrize(
    ("baseline_model_uri", "use_sample_weights"),
    [
        ("None", False),
        ("None", True),
        ("linear_regressor_model_uri", False),
    ],
    indirect=["baseline_model_uri"],
)
def test_regressor_evaluation(
    linear_regressor_model_uri,
    diabetes_dataset,
    baseline_model_uri,
    use_sample_weights,
):
    sample_weights = (
        np.random.rand(len(diabetes_dataset.labels_data)) if use_sample_weights else None
    )

    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            linear_regressor_model_uri,
            baseline_model_uri,
            diabetes_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=False,
            evaluator_config={
                "sample_weights": sample_weights,
            },
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(linear_regressor_model_uri)

    y = diabetes_dataset.labels_data
    y_pred = model.predict(diabetes_dataset.features_data)

    expected_metrics = _get_regressor_metrics(y, y_pred, sample_weights=sample_weights)
    expected_metrics["score"] = model._model_impl.score(
        diabetes_dataset.features_data, diabetes_dataset.labels_data, sample_weight=sample_weights
    )

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    for metric_key, expected_metric_val in expected_metrics.items():
        assert np.isclose(
            expected_metric_val,
            metrics[metric_key],
            rtol=1e-3,
        )
        assert np.isclose(expected_metric_val, result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_beeswarm_plot.png",
        "shap_feature_importance_plot.png",
        "shap_summary_plot.png",
    }

    assert result.artifacts.keys() == {
        "shap_beeswarm_plot",
        "shap_feature_importance_plot",
        "shap_summary_plot",
    }


def test_regressor_evaluation_disable_logging_metrics_and_artifacts(
    linear_regressor_model_uri,
    diabetes_dataset,
):
    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            linear_regressor_model_uri,
            linear_regressor_model_uri,
            diabetes_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=True,
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(linear_regressor_model_uri)

    y = diabetes_dataset.labels_data
    y_pred = model.predict(diabetes_dataset.features_data)

    expected_metrics = _get_regressor_metrics(y, y_pred, sample_weights=None)
    expected_metrics["score"] = model._model_impl.score(
        diabetes_dataset.features_data, diabetes_dataset.labels_data
    )

    check_metrics_not_logged_for_baseline_model_evaluation(
        expected_metrics=expected_metrics,
        result_metrics=result.baseline_model_metrics,
        logged_metrics=logged_metrics,
    )

    assert "mlflow.datassets" not in tags

    check_artifacts_are_not_generated_for_baseline_model_evaluation(
        logged_artifacts=artifacts,
        result_artifacts=result.artifacts,
    )


def test_regressor_evaluation_with_int_targets(
    linear_regressor_model_uri, diabetes_dataset, tmp_path
):
    with mlflow.start_run():
        result = evaluate(
            linear_regressor_model_uri,
            diabetes_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_dataset._constructor_args["targets"].astype(np.int64),
            evaluators="default",
        )
        result.save(tmp_path)


@pytest.mark.parametrize(
    ("baseline_model_uri", "use_sample_weights"),
    [
        ("None", False),
        ("None", True),
        ("multiclass_logistic_regressor_baseline_model_uri_4", False),
    ],
    indirect=["baseline_model_uri"],
)
def test_multi_classifier_evaluation(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    baseline_model_uri,
    use_sample_weights,
):
    sample_weights = np.random.rand(len(iris_dataset.labels_data)) if use_sample_weights else None

    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            multiclass_logistic_regressor_model_uri,
            baseline_model_uri,
            iris_dataset._constructor_args["data"],
            model_type="classifier",
            targets=iris_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=False,
            evaluator_config={
                "sample_weights": sample_weights,
            },
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(multiclass_logistic_regressor_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, predict_proba_fn = _extract_predict_fn(model, raw_model)
    y = iris_dataset.labels_data
    y_pred = predict_fn(iris_dataset.features_data)
    y_probs = predict_proba_fn(iris_dataset.features_data)

    expected_metrics = _get_multiclass_classifier_metrics(
        y_true=y, y_pred=y_pred, y_proba=y_probs, sample_weights=sample_weights
    )
    expected_metrics["score"] = model._model_impl.score(
        iris_dataset.features_data, iris_dataset.labels_data, sample_weight=sample_weights
    )

    for metric_key, expected_metric_val in expected_metrics.items():
        assert np.isclose(expected_metric_val, metrics[metric_key], rtol=1e-3)
        assert np.isclose(expected_metric_val, result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**iris_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_beeswarm_plot.png",
        "per_class_metrics.csv",
        "roc_curve_plot.png",
        "precision_recall_curve_plot.png",
        "shap_feature_importance_plot.png",
        "explainer",
        "confusion_matrix.png",
        "shap_summary_plot.png",
    }
    assert result.artifacts.keys() == {
        "per_class_metrics",
        "roc_curve_plot",
        "precision_recall_curve_plot",
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_multi_classifier_evaluation_disable_logging_metrics_and_artifacts(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
):
    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            multiclass_logistic_regressor_model_uri,
            multiclass_logistic_regressor_model_uri,
            iris_dataset._constructor_args["data"],
            model_type="classifier",
            targets=iris_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=True,
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(multiclass_logistic_regressor_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, predict_proba_fn = _extract_predict_fn(model, raw_model)
    y = iris_dataset.labels_data
    y_pred = predict_fn(iris_dataset.features_data)
    y_probs = predict_proba_fn(iris_dataset.features_data)

    expected_metrics = _get_multiclass_classifier_metrics(
        y_true=y, y_pred=y_pred, y_proba=y_probs, sample_weights=None
    )
    expected_metrics["score"] = model._model_impl.score(
        iris_dataset.features_data, iris_dataset.labels_data
    )

    check_metrics_not_logged_for_baseline_model_evaluation(
        expected_metrics=expected_metrics,
        result_metrics=result.baseline_model_metrics,
        logged_metrics=logged_metrics,
    )

    assert "mlflow.datassets" not in tags

    check_artifacts_are_not_generated_for_baseline_model_evaluation(
        logged_artifacts=artifacts,
        result_artifacts=result.artifacts,
    )


@pytest.mark.parametrize(
    ("baseline_model_uri", "use_sample_weights"),
    [
        ("None", False),
        ("binary_logistic_regressor_model_uri", False),
        ("binary_logistic_regressor_model_uri", True),
    ],
    indirect=["baseline_model_uri"],
)
def test_bin_classifier_evaluation(
    binary_logistic_regressor_model_uri,
    breast_cancer_dataset,
    baseline_model_uri,
    use_sample_weights,
):
    sample_weights = (
        np.random.rand(len(breast_cancer_dataset.labels_data)) if use_sample_weights else None
    )

    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            binary_logistic_regressor_model_uri,
            baseline_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=False,
            evaluator_config={
                "sample_weights": sample_weights,
            },
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, predict_proba_fn = _extract_predict_fn(model, raw_model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)
    y_probs = predict_proba_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(
        y_true=y, y_pred=y_pred, y_proba=y_probs, sample_weights=sample_weights
    )
    expected_metrics["score"] = model._model_impl.score(
        breast_cancer_dataset.features_data,
        breast_cancer_dataset.labels_data,
        sample_weight=sample_weights,
    )

    for metric_key, expected_metric_val in expected_metrics.items():
        assert np.isclose(
            expected_metric_val,
            metrics[metric_key],
            rtol=1e-3,
        )
        assert np.isclose(expected_metric_val, result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**breast_cancer_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_feature_importance_plot.png",
        "lift_curve_plot.png",
        "shap_beeswarm_plot.png",
        "precision_recall_curve_plot.png",
        "confusion_matrix.png",
        "shap_summary_plot.png",
        "roc_curve_plot.png",
    }
    assert result.artifacts.keys() == {
        "roc_curve_plot",
        "precision_recall_curve_plot",
        "lift_curve_plot",
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_bin_classifier_evaluation_disable_logging_metrics_and_artifacts(
    binary_logistic_regressor_model_uri,
    breast_cancer_dataset,
):
    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            binary_logistic_regressor_model_uri,
            binary_logistic_regressor_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=True,
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, predict_proba_fn = _extract_predict_fn(model, raw_model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)
    y_probs = predict_proba_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(
        y_true=y, y_pred=y_pred, y_proba=y_probs, sample_weights=None
    )
    expected_metrics["score"] = model._model_impl.score(
        breast_cancer_dataset.features_data, breast_cancer_dataset.labels_data
    )

    check_metrics_not_logged_for_baseline_model_evaluation(
        expected_metrics=expected_metrics,
        result_metrics=result.baseline_model_metrics,
        logged_metrics=logged_metrics,
    )

    assert "mlflow.datassets" not in tags

    check_artifacts_are_not_generated_for_baseline_model_evaluation(
        logged_artifacts=artifacts,
        result_artifacts=result.artifacts,
    )


@pytest.mark.parametrize(
    "baseline_model_uri",
    [
        ("None"),
        ("spark_linear_regressor_model_uri"),
    ],
    indirect=["baseline_model_uri"],
)
def test_spark_regressor_model_evaluation(
    spark_linear_regressor_model_uri,
    diabetes_spark_dataset,
    baseline_model_uri,
):
    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            spark_linear_regressor_model_uri,
            baseline_model_uri,
            diabetes_spark_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_spark_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=False,
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    X = diabetes_spark_dataset.features_data
    y = diabetes_spark_dataset.labels_data
    y_pred = model.predict(X)

    expected_metrics = _get_regressor_metrics(y, y_pred, sample_weights=None)

    for metric_key, expected_metric_val in expected_metrics.items():
        assert np.isclose(
            expected_metric_val,
            metrics[metric_key],
            rtol=1e-3,
        )
        assert np.isclose(expected_metric_val, result.metrics[metric_key], rtol=1e-3)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_spark_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == set()
    assert result.artifacts == {}


def test_spark_regressor_model_evaluation_disable_logging_metrics_and_artifacts(
    spark_linear_regressor_model_uri,
    diabetes_spark_dataset,
):
    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            spark_linear_regressor_model_uri,
            spark_linear_regressor_model_uri,
            diabetes_spark_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_spark_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=True,
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    X = diabetes_spark_dataset.features_data
    y = diabetes_spark_dataset.labels_data
    y_pred = model.predict(X)

    expected_metrics = _get_regressor_metrics(y, y_pred, sample_weights=None)

    check_metrics_not_logged_for_baseline_model_evaluation(
        expected_metrics=expected_metrics,
        result_metrics=result.baseline_model_metrics,
        logged_metrics=logged_metrics,
    )

    assert "mlflow.datassets" not in tags

    check_artifacts_are_not_generated_for_baseline_model_evaluation(
        logged_artifacts=artifacts,
        result_artifacts=result.artifacts,
    )


@pytest.mark.parametrize(
    "baseline_model_uri",
    [
        ("None"),
        ("svm_model_uri"),
    ],
    indirect=["baseline_model_uri"],
)
def test_svm_classifier_evaluation(svm_model_uri, breast_cancer_dataset, baseline_model_uri):
    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            svm_model_uri,
            baseline_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=False,
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(svm_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, _ = _extract_predict_fn(model, raw_model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(y_true=y, y_pred=y_pred, sample_weights=None)
    expected_metrics["score"] = model._model_impl.score(
        breast_cancer_dataset.features_data, breast_cancer_dataset.labels_data
    )

    for metric_key, expected_metric_val in expected_metrics.items():
        assert np.isclose(
            expected_metric_val,
            metrics[metric_key],
            rtol=1e-3,
        )
        assert np.isclose(expected_metric_val, result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**breast_cancer_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "confusion_matrix.png",
        "shap_feature_importance_plot.png",
        "shap_beeswarm_plot.png",
        "shap_summary_plot.png",
    }
    assert result.artifacts.keys() == {
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def _evaluate_explainer_with_exceptions(model_uri, dataset):
    with mlflow.start_run():
        evaluate(
            model_uri,
            dataset._constructor_args["data"],
            model_type="classifier",
            targets=dataset._constructor_args["targets"],
            evaluators="default",
            evaluator_config={
                "ignore_exceptions": False,
            },
        )


def test_default_explainer_pandas_df_str_cols(
    multiclass_logistic_regressor_model_uri, iris_pandas_df_dataset
):
    _evaluate_explainer_with_exceptions(
        multiclass_logistic_regressor_model_uri, iris_pandas_df_dataset
    )


def test_default_explainer_pandas_df_num_cols(
    multiclass_logistic_regressor_model_uri, iris_pandas_df_num_cols_dataset
):
    _evaluate_explainer_with_exceptions(
        multiclass_logistic_regressor_model_uri, iris_pandas_df_num_cols_dataset
    )


def test_svm_classifier_evaluation_disable_logging_metrics_and_artifacts(
    svm_model_uri, breast_cancer_dataset
):
    with mlflow.start_run() as run:
        result = evaluate_model_helper(
            svm_model_uri,
            svm_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=True,
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(svm_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, _ = _extract_predict_fn(model, raw_model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(y_true=y, y_pred=y_pred, sample_weights=None)
    expected_metrics["score"] = model._model_impl.score(
        breast_cancer_dataset.features_data, breast_cancer_dataset.labels_data
    )

    check_metrics_not_logged_for_baseline_model_evaluation(
        expected_metrics=expected_metrics,
        result_metrics=result.baseline_model_metrics,
        logged_metrics=logged_metrics,
    )

    assert "mlflow.datassets" not in tags

    check_artifacts_are_not_generated_for_baseline_model_evaluation(
        logged_artifacts=artifacts,
        result_artifacts=result.artifacts,
    )


@pytest.mark.parametrize(
    "baseline_model_uri",
    [
        ("None"),
        ("pipeline_model_uri"),
    ],
    indirect=["baseline_model_uri"],
)
def test_pipeline_model_kernel_explainer_on_categorical_features(
    pipeline_model_uri, baseline_model_uri
):
    from mlflow.models.evaluation._shap_patch import _PatchedKernelExplainer

    data, target_col = get_pipeline_model_dataset()
    with mlflow.start_run() as run:
        evaluate_model_helper(
            pipeline_model_uri,
            baseline_model_uri,
            data[0::3],
            model_type="classifier",
            targets=target_col,
            evaluators="default",
            evaluator_config={"explainability_algorithm": "kernel"},
            eval_baseline_model_only=False,
        )
    run_data = get_run_data(run.info.run_id)
    assert {
        "shap_beeswarm_plot.png",
        "shap_feature_importance_plot.png",
        "shap_summary_plot.png",
        "explainer",
    }.issubset(run_data.artifacts)

    explainer = mlflow.shap.load_explainer(f"runs:/{run.info.run_id}/explainer")
    assert isinstance(explainer, _PatchedKernelExplainer)


def test_compute_df_mode_or_mean():
    df = pd.DataFrame(
        {
            "a": [2.0, 2.0, 5.0],
            "b": [3, 3, 5],
            "c": [2.0, 2.0, 6.5],
            "d": [True, False, True],
            "e": ["abc", "b", "abc"],
            "f": [1.5, 2.5, np.nan],
            "g": ["ab", "ab", None],
            "h": pd.Series([2.0, 2.0, 6.5], dtype="category"),
        }
    )
    result = _compute_df_mode_or_mean(df)
    assert result == {
        "a": 2,
        "b": 3,
        "c": 3.5,
        "d": True,
        "e": "abc",
        "f": 2.0,
        "g": "ab",
        "h": 2.0,
    }

    # Test on dataframe that all columns are continuous.
    df2 = pd.DataFrame(
        {
            "c": [2.0, 2.0, 6.5],
            "f": [1.5, 2.5, np.nan],
        }
    )
    assert _compute_df_mode_or_mean(df2) == {"c": 3.5, "f": 2.0}

    # Test on dataframe that all columns are not continuous.
    df2 = pd.DataFrame(
        {
            "d": [True, False, True],
            "g": ["ab", "ab", None],
        }
    )
    assert _compute_df_mode_or_mean(df2) == {"d": True, "g": "ab"}


def test_infer_model_type_by_labels():
    assert _infer_model_type_by_labels(["a", "b"]) == "classifier"
    assert _infer_model_type_by_labels([True, False]) == "classifier"
    assert _infer_model_type_by_labels([1, 2.5]) == "regressor"
    assert _infer_model_type_by_labels(pd.Series(["a", "b"], dtype="category")) == "classifier"
    assert _infer_model_type_by_labels(pd.Series([1.5, 2.5], dtype="category")) == "classifier"
    assert _infer_model_type_by_labels([1, 2, 3]) is None


def test_extract_raw_model_and_predict_fn(
    binary_logistic_regressor_model_uri, breast_cancer_dataset
):
    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    model_loader_module, raw_model = _extract_raw_model(model)
    predict_fn, predict_proba_fn = _extract_predict_fn(model, raw_model)

    assert model_loader_module == "mlflow.sklearn"
    assert isinstance(raw_model, LogisticRegression)
    np.testing.assert_allclose(
        predict_fn(breast_cancer_dataset.features_data),
        raw_model.predict(breast_cancer_dataset.features_data),
    )
    np.testing.assert_allclose(
        predict_proba_fn(breast_cancer_dataset.features_data),
        raw_model.predict_proba(breast_cancer_dataset.features_data),
    )


@pytest.mark.parametrize("use_sample_weights", [True, False])
def test_get_regressor_metrics(use_sample_weights):
    y = [1.1, 2.1, -3.5]
    y_pred = [1.5, 2.0, -3.0]
    sample_weights = [1, 2, 3] if use_sample_weights else None

    metrics = _get_regressor_metrics(y, y_pred, sample_weights)

    if use_sample_weights:
        expected_metrics = {
            "example_count": 3,
            "mean_absolute_error": 0.35000000000000003,
            "mean_squared_error": 0.155,
            "root_mean_squared_error": 0.39370039370059057,
            "sum_on_target": -5.199999999999999,
            "mean_on_target": -1.7333333333333332,
            "r2_score": 0.9780003154076644,
            "max_error": 0.5,
            "mean_absolute_percentage_error": 0.1479076479076479,
        }
    else:
        expected_metrics = {
            "example_count": 3,
            "mean_absolute_error": 0.3333333333333333,
            "mean_squared_error": 0.13999999999999999,
            "root_mean_squared_error": 0.3741657386773941,
            "sum_on_target": -0.2999999999999998,
            "mean_on_target": -0.09999999999999994,
            "r2_score": 0.976457399103139,
            "max_error": 0.5,
            "mean_absolute_percentage_error": 0.18470418470418468,
        }

    assert_dict_equal(metrics, expected_metrics, rtol=1e-3)


def test_get_binary_sum_up_label_pred_prob():
    y = [0, 1, 2]
    y_pred = [0, 2, 1]
    y_probs = [[0.7, 0.1, 0.2], [0.2, 0.3, 0.5], [0.25, 0.4, 0.35]]

    results = []
    for idx, label in enumerate([0, 1, 2]):
        y_bin, y_pred_bin, y_prob_bin = _get_binary_sum_up_label_pred_prob(
            idx, label, y, y_pred, y_probs
        )
        results.append((list(y_bin), list(y_pred_bin), list(y_prob_bin)))

    assert results == [
        ([1, 0, 0], [1, 0, 0], [0.7, 0.2, 0.25]),
        ([0, 1, 0], [0, 0, 1], [0.1, 0.3, 0.4]),
        ([0, 0, 1], [0, 1, 0], [0.2, 0.5, 0.35]),
    ]


@pytest.mark.parametrize("use_sample_weights", [True, False])
def test_get_binary_classifier_metrics(use_sample_weights):
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    sample_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1] if use_sample_weights else None

    if use_sample_weights:
        expected_metrics = {
            "example_count": 10,
            "true_negatives": 3,
            "true_positives": 4,
            "false_negatives": 1,
            "false_positives": 2,
            "accuracy_score": 0.9347826086956524,
            "f1_score": 0.9361702127659577,
            "precision_score": 0.9166666666666667,
            "recall_score": 0.9565217391304349,
        }
    else:
        expected_metrics = {
            "example_count": 10,
            "true_negatives": 3,
            "true_positives": 4,
            "false_negatives": 1,
            "false_positives": 2,
            "accuracy_score": 0.7,
            "f1_score": 0.7272727272727272,
            "precision_score": 0.6666666666666666,
            "recall_score": 0.8,
        }

    metrics = _get_binary_classifier_metrics(y_true=y, y_pred=y_pred, sample_weights=sample_weights)
    assert_dict_equal(metrics, expected_metrics, rtol=1e-3)


@pytest.mark.parametrize("use_sample_weights", [True, False])
def test_get_multiclass_classifier_metrics(use_sample_weights):
    y = [0, 1, 2, 1, 2]
    y_pred = [0, 2, 1, 1, 0]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]
    sample_weights = [1, 0.1, 0.1, 1, 0.1] if use_sample_weights else None

    if use_sample_weights:
        expected_metrics = {
            "example_count": 5,
            "accuracy_score": 0.8695652173913042,
            "f1_score": 0.8488612836438922,
            "log_loss": 0.7515668165194579,
            "precision_score": 0.8300395256916996,
            "recall_score": 0.8695652173913042,
            "roc_auc": 0.8992673992673993,
        }
    else:
        expected_metrics = {
            "example_count": 5,
            "accuracy_score": 0.4,
            "f1_score": 0.3333333333333333,
            "log_loss": 1.1658691395263094,
            "precision_score": 0.3,
            "recall_score": 0.4,
            "roc_auc": 0.5833333333333334,
        }

    metrics = _get_multiclass_classifier_metrics(
        y_true=y, y_pred=y_pred, y_proba=y_probs, labels=[0, 1, 2], sample_weights=sample_weights
    )
    assert_dict_equal(metrics, expected_metrics, 1e-3)


def test_gen_binary_precision_recall_curve_no_sample_weights():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]

    results = _gen_classifier_curve(
        is_binomial=True,
        y=y,
        y_probs=y_prob,
        labels=[0, 1],
        pos_label=1,
        curve_type="pr",
        sample_weights=None,
    )
    np.testing.assert_allclose(
        results.plot_fn_args["data_series"][0][1],
        np.array([1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.4, 0.4, 0.2, 0.0]),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        results.plot_fn_args["data_series"][0][2],
        np.array([0.5, 0.55555556, 0.5, 0.57142857, 0.66666667, 0.6, 0.5, 0.66666667, 1.0, 1.0]),
        rtol=1e-3,
    )
    assert results.plot_fn_args["xlabel"] == "Recall (Positive label: 1)"
    assert results.plot_fn_args["ylabel"] == "Precision (Positive label: 1)"
    assert results.plot_fn_args["title"] == "Precision recall curve"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}
    assert np.isclose(results.auc, 0.69777777, rtol=1e-3)


def test_gen_binary_precision_recall_curve_with_sample_weights():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]
    sample_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 0.1, 0.1]

    results = _gen_classifier_curve(
        is_binomial=True,
        y=y,
        y_probs=y_prob,
        labels=[0, 1],
        pos_label=1,
        curve_type="pr",
        sample_weights=sample_weights,
    )
    np.testing.assert_allclose(
        results.plot_fn_args["data_series"][0][1],
        np.array(
            [
                1.0,
                1.0,
                0.83870968,
                0.83870968,
                0.83870968,
                0.51612903,
                0.48387097,
                0.48387097,
                0.16129032,
                0.0,
            ]
        ),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        results.plot_fn_args["data_series"][0][2],
        np.array(
            [
                0.54386,
                0.59615385,
                0.55319149,
                0.7027027,
                0.72222222,
                0.61538462,
                0.6,
                0.75,
                1.0,
                1.0,
            ]
        ),
        rtol=1e-3,
    )
    assert results.plot_fn_args["xlabel"] == "Recall (Positive label: 1)"
    assert results.plot_fn_args["ylabel"] == "Precision (Positive label: 1)"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}
    assert np.isclose(results.auc, 0.7522056796250345, rtol=1e-3)


def test_gen_binary_roc_curve_no_sample_weights():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]

    results = _gen_classifier_curve(
        is_binomial=True,
        y=y,
        y_probs=y_prob,
        labels=[0, 1],
        pos_label=1,
        curve_type="roc",
        sample_weights=None,
    )
    np.testing.assert_allclose(
        results.plot_fn_args["data_series"][0][1],
        np.array([0.0, 0.0, 0.2, 0.4, 0.4, 0.8, 0.8, 1.0]),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        results.plot_fn_args["data_series"][0][2],
        np.array([0.0, 0.2, 0.4, 0.4, 0.8, 0.8, 1.0, 1.0]),
        rtol=1e-3,
    )
    assert results.plot_fn_args["xlabel"] == "False Positive Rate (Positive label: 1)"
    assert results.plot_fn_args["ylabel"] == "True Positive Rate (Positive label: 1)"
    assert results.plot_fn_args["title"] == "ROC curve"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}
    assert np.isclose(results.auc, 0.66, rtol=1e-3)


def test_gen_binary_roc_curve_with_sample_weights():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]
    sample_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 0.1, 0.1]

    results = _gen_classifier_curve(
        is_binomial=True,
        y=y,
        y_probs=y_prob,
        labels=[0, 1],
        pos_label=1,
        curve_type="roc",
        sample_weights=sample_weights,
    )
    np.testing.assert_allclose(
        results.plot_fn_args["data_series"][0][1],
        np.array(
            [
                0.0,
                0.0,
                0.19230769,
                0.38461538,
                0.38461538,
                0.38461538,
                0.42307692,
                0.80769231,
                0.80769231,
                1.0,
            ]
        ),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        results.plot_fn_args["data_series"][0][2],
        np.array(
            [
                0.0,
                0.16129032,
                0.48387097,
                0.48387097,
                0.51612903,
                0.83870968,
                0.83870968,
                0.83870968,
                1.0,
                1.0,
            ]
        ),
        rtol=1e-3,
    )
    assert results.plot_fn_args["xlabel"] == "False Positive Rate (Positive label: 1)"
    assert results.plot_fn_args["ylabel"] == "True Positive Rate (Positive label: 1)"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}
    assert np.isclose(results.auc, 0.702, rtol=1e-3)


def test_gen_multiclass_precision_recall_curve_no_sample_weights():
    y = [0, 1, 2, 1, 2]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]

    results = _gen_classifier_curve(
        is_binomial=False,
        y=y,
        y_probs=y_probs,
        labels=[0, 1, 2],
        pos_label=None,
        curve_type="pr",
        sample_weights=None,
    )
    expected_x_data_list = [
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.5, 0.0],
        [1.0, 0.5, 0.5, 0.5, 0.0, 0.0],
    ]
    expected_y_data_list = [
        [0.2, 0.25, 0.333333, 0.5, 0.0, 1.0],
        [0.4, 0.66666667, 0.5, 1.0],
        [0.4, 0.25, 0.33333333, 0.5, 0.0, 1.0],
    ]
    line_labels = ["label=0,AP=0.500", "label=1,AP=0.583", "label=2,AP=0.450"]
    for index, (name, x_data, y_data) in enumerate(results.plot_fn_args["data_series"]):
        assert name == line_labels[index]
        np.testing.assert_allclose(x_data, expected_x_data_list[index], rtol=1e-3)
        np.testing.assert_allclose(y_data, expected_y_data_list[index], rtol=1e-3)

    assert results.plot_fn_args["xlabel"] == "Recall"
    assert results.plot_fn_args["ylabel"] == "Precision"
    assert results.plot_fn_args["title"] == "Precision recall curve"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}

    expected_auc = [0.5, 0.583333, 0.45]
    np.testing.assert_allclose(results.auc, expected_auc, rtol=1e-3)


def test_gen_multiclass_precision_recall_curve_with_sample_weights():
    y = [0, 1, 2, 1, 2]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]
    sample_weights = [0.5, 0.5, 0.5, 0.25, 0.75]

    results = _gen_classifier_curve(
        is_binomial=False,
        y=y,
        y_probs=y_probs,
        labels=[0, 1, 2],
        pos_label=None,
        curve_type="pr",
        sample_weights=sample_weights,
    )
    expected_x_data_list = [
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.333333, 0.0],
        [1.0, 0.4, 0.4, 0.4, 0.0, 0.0],
    ]
    expected_y_data_list = [
        [0.2, 0.25, 0.333333, 0.4, 0.0, 1.0],
        [0.3, 0.6, 0.333333, 1.0],
        [0.5, 0.285714, 0.4, 0.5, 0.0, 1.0],
    ]
    line_labels = ["label=0,AP=0.400", "label=1,AP=0.511", "label=2,AP=0.500"]
    for index, (name, x_data, y_data) in enumerate(results.plot_fn_args["data_series"]):
        assert name == line_labels[index]
        np.testing.assert_allclose(x_data, expected_x_data_list[index], rtol=1e-3)
        np.testing.assert_allclose(y_data, expected_y_data_list[index], rtol=1e-3)

    assert results.plot_fn_args["xlabel"] == "Recall"
    assert results.plot_fn_args["ylabel"] == "Precision"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}

    expected_auc = [0.4, 0.511111, 0.5]
    np.testing.assert_allclose(results.auc, expected_auc, rtol=1e-3)


def test_gen_multiclass_roc_curve_no_sample_weights():
    y = [0, 1, 2, 1, 2]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]

    results = _gen_classifier_curve(
        is_binomial=False,
        y=y,
        y_probs=y_probs,
        labels=[0, 1, 2],
        pos_label=None,
        curve_type="roc",
        sample_weights=None,
    )

    expected_x_data_list = [
        [0.0, 0.25, 0.25, 1.0],
        [0.0, 0.33333333, 0.33333333, 1.0],
        [0.0, 0.33333333, 0.33333333, 1.0, 1.0],
    ]
    expected_y_data_list = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5, 1.0]]
    line_labels = ["label=0,AUC=0.750", "label=1,AUC=0.750", "label=2,AUC=0.333"]
    for index, (name, x_data, y_data) in enumerate(results.plot_fn_args["data_series"]):
        assert name == line_labels[index]
        np.testing.assert_allclose(x_data, expected_x_data_list[index], rtol=1e-3)
        np.testing.assert_allclose(y_data, expected_y_data_list[index], rtol=1e-3)

    assert results.plot_fn_args["xlabel"] == "False Positive Rate"
    assert results.plot_fn_args["ylabel"] == "True Positive Rate"
    assert results.plot_fn_args["title"] == "ROC curve"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}

    expected_auc = [0.75, 0.75, 0.3333]
    np.testing.assert_allclose(results.auc, expected_auc, rtol=1e-3)


def test_gen_multiclass_roc_curve_with_sample_weights():
    y = [0, 1, 2, 1, 2]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]
    sample_weights = [0.5, 0.5, 0.5, 0.25, 0.75]

    results = _gen_classifier_curve(
        is_binomial=False,
        y=y,
        y_probs=y_probs,
        labels=[0, 1, 2],
        pos_label=None,
        curve_type="roc",
        sample_weights=sample_weights,
    )

    expected_x_data_list = [
        [0.0, 0.375, 0.375, 0.5, 1.0],
        [0.0, 0.285714, 0.285714, 1.0],
        [0.0, 0.4, 0.4, 0.6, 1.0, 1.0],
    ]
    expected_y_data_list = [
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.333333, 1.0, 1.0],
        [0.0, 0.0, 0.4, 0.4, 0.4, 1.0],
    ]
    line_labels = ["label=0,AUC=0.625", "label=1,AUC=0.762", "label=2,AUC=0.240"]
    for index, (name, x_data, y_data) in enumerate(results.plot_fn_args["data_series"]):
        assert name == line_labels[index]
        np.testing.assert_allclose(x_data, expected_x_data_list[index], rtol=1e-3)
        np.testing.assert_allclose(y_data, expected_y_data_list[index], rtol=1e-3)

    assert results.plot_fn_args["xlabel"] == "False Positive Rate"
    assert results.plot_fn_args["ylabel"] == "True Positive Rate"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}

    expected_auc = [0.625, 0.761905, 0.24]
    np.testing.assert_allclose(results.auc, expected_auc, rtol=1e-3)


def test_evaluate_extra_metric_backwards_compatible():
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    builtin_metrics = _get_regressor_metrics(
        eval_df["target"], eval_df["prediction"], sample_weights=None
    )
    metrics = _get_aggregate_metrics_values(builtin_metrics)

    def old_fn(eval_df, builtin_metrics):
        return builtin_metrics["mean_absolute_error"] * 1.5

    eval_fn_args = [eval_df, builtin_metrics]
    res_metric = _evaluate_extra_metric(_CustomMetric(old_fn, "old_fn", 0), eval_fn_args)
    assert res_metric.scores is None
    assert res_metric.justifications is None
    assert res_metric.aggregate_results["old_fn"] == builtin_metrics["mean_absolute_error"] * 1.5

    new_eval_fn_args = [eval_df, None, metrics]

    def new_fn(predictions, targets=None, metrics=None):
        return metrics["mean_absolute_error"].aggregate_results["mean_absolute_error"] * 1.5

    res_metric = _evaluate_extra_metric(_CustomMetric(new_fn, "new_fn", 0), new_eval_fn_args)
    assert res_metric.scores is None
    assert res_metric.justifications is None
    assert res_metric.aggregate_results["new_fn"] == builtin_metrics["mean_absolute_error"] * 1.5


def test_evaluate_custom_metric_incorrect_return_formats():
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    builtin_metrics = _get_regressor_metrics(
        eval_df["target"], eval_df["prediction"], sample_weights=None
    )
    eval_fn_args = [eval_df, builtin_metrics]

    def dummy_fn(*_):
        pass

    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(_CustomMetric(dummy_fn, "dummy_fn", 0), eval_fn_args)
        mock_warning.assert_called_once_with(
            "Did not log metric 'dummy_fn' at index 0 in the `extra_metrics` parameter"
            " because it returned None."
        )

    def incorrect_return_type(*_):
        return ["stuff"], 3

    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(
            _CustomMetric(incorrect_return_type, incorrect_return_type.__name__, 0), eval_fn_args
        )
        mock_warning.assert_called_once_with(
            f"Did not log metric '{incorrect_return_type.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it did not return a MetricValue."
        )

    def non_list_scores(*_):
        return MetricValue(scores=5)

    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(
            _CustomMetric(non_list_scores, non_list_scores.__name__, 0), eval_fn_args
        )
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_list_scores.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with scores as a list."
        )

    def non_numeric_scores(*_):
        return MetricValue(scores=[{"val": "string"}])

    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(
            _CustomMetric(non_numeric_scores, non_numeric_scores.__name__, 0), eval_fn_args
        )
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_numeric_scores.__name__}' at index 0 in the `extra_metrics`"
            " parameter because it must return MetricValue with numeric or string scores."
        )

    def non_list_justifications(*_):
        return MetricValue(justifications="string")

    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(
            _CustomMetric(non_list_justifications, non_list_justifications.__name__, 0),
            eval_fn_args,
        )
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_list_justifications.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with justifications "
            "as a list."
        )

    def non_str_justifications(*_):
        return MetricValue(justifications=[3, 4])

    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(
            _CustomMetric(non_str_justifications, non_str_justifications.__name__, 0), eval_fn_args
        )
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_str_justifications.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with string "
            "justifications."
        )

    def non_dict_aggregates(*_):
        return MetricValue(aggregate_results=[5.0, 4.0])

    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(
            _CustomMetric(non_dict_aggregates, non_dict_aggregates.__name__, 0), eval_fn_args
        )
        mock_warning.assert_called_once_with(
            f"Did not log metric '{non_dict_aggregates.__name__}' at index 0 in the "
            "`extra_metrics` parameter because it must return MetricValue with aggregate_results "
            "as a dict."
        )

    def wrong_type_aggregates(*_):
        return MetricValue(aggregate_results={"toxicity": 0.0, "hi": "hi"})

    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(
            _CustomMetric(wrong_type_aggregates, wrong_type_aggregates.__name__, 0), eval_fn_args
        )
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
    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
        _evaluate_extra_metric(_CustomMetric(fn, "<lambda>", 0), eval_fn_args)
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
    res_metric = _evaluate_extra_metric(
        _CustomMetric(example_count_times_1_point_5, "", 0), eval_fn_args
    )
    assert (
        res_metric.aggregate_results["example_count_times_1_point_5"]
        == builtin_metrics["example_count"] * 1.5
    )
    assert res_metric.scores == [score * 1.5 for score in eval_df["prediction"].tolist()]
    assert res_metric.justifications == ["justification"] * len(eval_df["prediction"])


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


def _get_results_for_custom_metrics_tests(
    model_uri, dataset, *, extra_metrics=None, custom_artifacts=None
):
    with mlflow.start_run() as run:
        result = evaluate(
            model_uri,
            dataset._constructor_args["data"],
            model_type="classifier",
            targets=dataset._constructor_args["targets"],
            evaluators="default",
            extra_metrics=extra_metrics,
            custom_artifacts=custom_artifacts,
        )
    _, metrics, _, artifacts = get_run_data(run.info.run_id)
    return result, metrics, artifacts


def test_custom_metric_produced_multiple_artifacts_with_same_name_throw_exception(
    binary_logistic_regressor_model_uri, breast_cancer_dataset
):
    def example_custom_artifact_1(_, __, ___):
        return {"test_json_artifact": {"a": 2, "b": [1, 2]}}

    def example_custom_artifact_2(_, __, ___):
        return {"test_json_artifact": {"a": 3, "b": [1, 2]}}

    with pytest.raises(
        MlflowException,
        match="cannot be logged because there already exists an artifact with the same name",
    ):
        _get_results_for_custom_metrics_tests(
            binary_logistic_regressor_model_uri,
            breast_cancer_dataset,
            custom_artifacts=[
                example_custom_artifact_1,
                example_custom_artifact_2,
            ],
        )


def test_custom_metric_mixed(binary_logistic_regressor_model_uri, breast_cancer_dataset):
    def true_count(predictions, targets=None, metrics=None):
        true_negatives = metrics["true_negatives"].aggregate_results["true_negatives"]
        true_positives = metrics["true_positives"].aggregate_results["true_positives"]
        return MetricValue(aggregate_results={"true_count": true_negatives + true_positives})

    def positive_count(eval_df, _metrics):
        return MetricValue(aggregate_results={"positive_count": np.sum(eval_df["prediction"])})

    def example_custom_artifact(_eval_df, _given_metrics, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_csv(path_join(tmp_path, "user_logged_df.csv"), index=False)
        np_array = np.array([1, 2, 3, 4, 5])
        np.save(path_join(tmp_path, "arr.npy"), np_array)
        return {
            "test_json_artifact": {"a": 3, "b": [1, 2]},
            "test_npy_artifact": path_join(tmp_path, "arr.npy"),
        }

    result, metrics, artifacts = _get_results_for_custom_metrics_tests(
        binary_logistic_regressor_model_uri,
        breast_cancer_dataset,
        extra_metrics=[
            make_metric(eval_fn=true_count, greater_is_better=True),
            make_metric(eval_fn=positive_count, greater_is_better=True),
        ],
        custom_artifacts=[example_custom_artifact],
    )

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, _ = _extract_predict_fn(model, raw_model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(y_true=y, y_pred=y_pred, sample_weights=None)

    assert "true_count" in metrics
    assert np.isclose(
        metrics["true_count"],
        expected_metrics["true_negatives"] + expected_metrics["true_positives"],
        rtol=1e-3,
    )
    assert "true_count" in result.metrics
    assert np.isclose(
        result.metrics["true_count"],
        expected_metrics["true_negatives"] + expected_metrics["true_positives"],
        rtol=1e-3,
    )

    assert "positive_count" in metrics
    assert np.isclose(metrics["positive_count"], np.sum(y_pred), rtol=1e-3)
    assert "positive_count" in result.metrics
    assert np.isclose(result.metrics["positive_count"], np.sum(y_pred), rtol=1e-3)

    assert "test_json_artifact" in result.artifacts
    assert "test_json_artifact.json" in artifacts
    assert isinstance(result.artifacts["test_json_artifact"], JsonEvaluationArtifact)
    assert result.artifacts["test_json_artifact"].content == {"a": 3, "b": [1, 2]}

    assert "test_npy_artifact" in result.artifacts
    assert "test_npy_artifact.npy" in artifacts
    assert isinstance(result.artifacts["test_npy_artifact"], NumpyEvaluationArtifact)
    np.testing.assert_array_equal(
        result.artifacts["test_npy_artifact"].content, np.array([1, 2, 3, 4, 5])
    )


def test_custom_metric_logs_artifacts_from_paths(
    binary_logistic_regressor_model_uri, breast_cancer_dataset
):
    fig_x = 8.0
    fig_y = 5.0
    fig_dpi = 100.0
    img_formats = ("png", "jpeg", "jpg")

    def example_custom_artifact(_, __, tmp_path):
        example_artifacts = {}

        # images
        for ext in img_formats:
            fig = plt.figure(figsize=(fig_x, fig_y), dpi=fig_dpi)
            plt.plot([1, 2, 3])
            fig.savefig(path_join(tmp_path, f"test.{ext}"), format=ext)
            plt.clf()
            example_artifacts[f"test_{ext}_artifact"] = path_join(tmp_path, f"test.{ext}")

        # json
        with open(path_join(tmp_path, "test.json"), "w") as f:
            json.dump([1, 2, 3], f)
        example_artifacts["test_json_artifact"] = path_join(tmp_path, "test.json")

        # numpy
        np_array = np.array([1, 2, 3, 4, 5])
        np.save(path_join(tmp_path, "test.npy"), np_array)
        example_artifacts["test_npy_artifact"] = path_join(tmp_path, "test.npy")

        # csv
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_csv(path_join(tmp_path, "test.csv"), index=False)
        example_artifacts["test_csv_artifact"] = path_join(tmp_path, "test.csv")

        # parquet
        df = pd.DataFrame({"test": [1, 2, 3]})
        df.to_parquet(path_join(tmp_path, "test.parquet"))
        example_artifacts["test_parquet_artifact"] = path_join(tmp_path, "test.parquet")

        # text
        with open(path_join(tmp_path, "test.txt"), "w") as f:
            f.write("hello world")
        example_artifacts["test_text_artifact"] = path_join(tmp_path, "test.txt")

        return example_artifacts

    result, _, artifacts = _get_results_for_custom_metrics_tests(
        binary_logistic_regressor_model_uri,
        breast_cancer_dataset,
        custom_artifacts=[example_custom_artifact],
    )

    with TemporaryDirectory() as tmp_dir:
        for img_ext in img_formats:
            assert f"test_{img_ext}_artifact" in result.artifacts
            assert f"test_{img_ext}_artifact.{img_ext}" in artifacts
            assert isinstance(result.artifacts[f"test_{img_ext}_artifact"], ImageEvaluationArtifact)

            fig = plt.figure(figsize=(fig_x, fig_y), dpi=fig_dpi)
            plt.plot([1, 2, 3])
            fig.savefig(path_join(tmp_dir, f"test.{img_ext}"), format=img_ext)
            plt.clf()

            saved_img = Image.open(path_join(tmp_dir, f"test.{img_ext}"))
            result_img = result.artifacts[f"test_{img_ext}_artifact"].content

            for img in (saved_img, result_img):
                img_ext_qualified = "jpeg" if img_ext == "jpg" else img_ext
                assert img.format.lower() == img_ext_qualified
                assert img.size == (fig_x * fig_dpi, fig_y * fig_dpi)
                assert pytest.approx(img.info.get("dpi"), 0.001) == (fig_dpi, fig_dpi)

    assert "test_json_artifact" in result.artifacts
    assert "test_json_artifact.json" in artifacts
    assert isinstance(result.artifacts["test_json_artifact"], JsonEvaluationArtifact)
    assert result.artifacts["test_json_artifact"].content == [1, 2, 3]

    assert "test_npy_artifact" in result.artifacts
    assert "test_npy_artifact.npy" in artifacts
    assert isinstance(result.artifacts["test_npy_artifact"], NumpyEvaluationArtifact)
    np.testing.assert_array_equal(
        result.artifacts["test_npy_artifact"].content, np.array([1, 2, 3, 4, 5])
    )

    assert "test_csv_artifact" in result.artifacts
    assert "test_csv_artifact.csv" in artifacts
    assert isinstance(result.artifacts["test_csv_artifact"], CsvEvaluationArtifact)
    pd.testing.assert_frame_equal(
        result.artifacts["test_csv_artifact"].content, pd.DataFrame({"a": [1, 2, 3]})
    )

    assert "test_parquet_artifact" in result.artifacts
    assert "test_parquet_artifact.parquet" in artifacts
    assert isinstance(result.artifacts["test_parquet_artifact"], ParquetEvaluationArtifact)
    pd.testing.assert_frame_equal(
        result.artifacts["test_parquet_artifact"].content, pd.DataFrame({"test": [1, 2, 3]})
    )

    assert "test_text_artifact" in result.artifacts
    assert "test_text_artifact.txt" in artifacts
    assert isinstance(result.artifacts["test_text_artifact"], TextEvaluationArtifact)
    assert result.artifacts["test_text_artifact"].content == "hello world"


class _ExampleToBePickledObject:
    def __init__(self):
        self.a = [1, 2, 3]
        self.b = "hello"

    def __eq__(self, o: object) -> bool:
        return self.a == o.a and self.b == self.b


def test_custom_metric_logs_artifacts_from_objects(
    binary_logistic_regressor_model_uri, breast_cancer_dataset
):
    fig = plt.figure()
    plt.plot([1, 2, 3])
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    def example_custom_artifacts(_, __, ___):
        return {
            "test_image_artifact": fig,
            "test_json_artifact": {
                "list": [1, 2, 3],
                "numpy_int": np.int64(0),
                "numpy_float": np.float64(0.5),
            },
            "test_npy_artifact": np.array([1, 2, 3, 4, 5]),
            "test_csv_artifact": pd.DataFrame({"a": [1, 2, 3]}),
            "test_json_text_artifact": '{"a": [1, 2, 3], "c": 3.4}',
            "test_pickled_artifact": _ExampleToBePickledObject(),
        }

    result, _, artifacts = _get_results_for_custom_metrics_tests(
        binary_logistic_regressor_model_uri,
        breast_cancer_dataset,
        custom_artifacts=[example_custom_artifacts],
    )

    assert "test_image_artifact" in result.artifacts
    assert "test_image_artifact.png" in artifacts
    assert isinstance(result.artifacts["test_image_artifact"], ImageEvaluationArtifact)
    img_diff = ImageChops.difference(result.artifacts["test_image_artifact"].content, img).getbbox()
    assert img_diff is None

    assert "test_json_artifact" in result.artifacts
    assert "test_json_artifact.json" in artifacts
    assert isinstance(result.artifacts["test_json_artifact"], JsonEvaluationArtifact)
    assert result.artifacts["test_json_artifact"].content == {
        "list": [1, 2, 3],
        "numpy_int": 0,
        "numpy_float": 0.5,
    }

    assert "test_npy_artifact" in result.artifacts
    assert "test_npy_artifact.npy" in artifacts
    assert isinstance(result.artifacts["test_npy_artifact"], NumpyEvaluationArtifact)
    np.testing.assert_array_equal(
        result.artifacts["test_npy_artifact"].content, np.array([1, 2, 3, 4, 5])
    )

    assert "test_csv_artifact" in result.artifacts
    assert "test_csv_artifact.csv" in artifacts
    assert isinstance(result.artifacts["test_csv_artifact"], CsvEvaluationArtifact)
    pd.testing.assert_frame_equal(
        result.artifacts["test_csv_artifact"].content, pd.DataFrame({"a": [1, 2, 3]})
    )

    assert "test_json_text_artifact" in result.artifacts
    assert "test_json_text_artifact.json" in artifacts
    assert isinstance(result.artifacts["test_json_text_artifact"], JsonEvaluationArtifact)
    assert result.artifacts["test_json_text_artifact"].content == {"a": [1, 2, 3], "c": 3.4}

    assert "test_pickled_artifact" in result.artifacts
    assert "test_pickled_artifact.pickle" in artifacts
    assert isinstance(result.artifacts["test_pickled_artifact"], PickleEvaluationArtifact)
    assert result.artifacts["test_pickled_artifact"].content == _ExampleToBePickledObject()


def test_evaluate_sklearn_model_score_skip_when_not_scorable(
    linear_regressor_model_uri, diabetes_dataset
):
    with mock.patch(
        "sklearn.linear_model.LinearRegression.score",
        side_effect=RuntimeError("LinearRegression.score failed"),
    ) as mock_score:
        with mlflow.start_run():
            result = evaluate(
                linear_regressor_model_uri,
                diabetes_dataset._constructor_args["data"],
                model_type="regressor",
                targets=diabetes_dataset._constructor_args["targets"],
                evaluators="default",
            )
        mock_score.assert_called_once()
        assert "score" not in result.metrics


@pytest.mark.parametrize(
    "model",
    [LogisticRegression(), LinearRegression()],
)
def test_autologging_is_disabled_during_evaluate(model):
    mlflow.sklearn.autolog()
    try:
        X, y = load_iris(as_frame=True, return_X_y=True)
        with mlflow.start_run() as run:
            model.fit(X, y)
            model_info = mlflow.sklearn.log_model(model, "model")
            result = evaluate(
                model_info.model_uri,
                X.assign(target=y),
                model_type="classifier" if isinstance(model, LogisticRegression) else "regressor",
                targets="target",
                evaluators="default",
            )

        run_data = get_run_data(run.info.run_id)
        duplicate_metrics = []
        for evaluate_metric_key in result.metrics.keys():
            matched_keys = [k for k in run_data.metrics.keys() if k.startswith(evaluate_metric_key)]
            if len(matched_keys) > 1:
                duplicate_metrics += matched_keys
        assert duplicate_metrics == []
    finally:
        mlflow.sklearn.autolog(disable=True)


def test_evaluation_works_with_model_pipelines_that_modify_input_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=["0", "1", "2", "3"])
    y = pd.Series(iris.target)

    def add_feature(df):
        df["newfeature"] = 1
        return df

    # Define a transformer that modifies input data by adding an extra feature column
    add_feature_transformer = FunctionTransformer(add_feature, validate=False)
    model_pipeline = Pipeline(
        steps=[("add_feature", add_feature_transformer), ("predict", LogisticRegression())]
    )
    model_pipeline.fit(X, y)

    with mlflow.start_run() as run:
        pipeline_model_uri = mlflow.sklearn.log_model(model_pipeline, "model").model_uri

        evaluation_data = pd.DataFrame(load_iris().data, columns=["0", "1", "2", "3"])
        evaluation_data["labels"] = load_iris().target

        evaluate(
            pipeline_model_uri,
            evaluation_data,
            model_type="regressor",
            targets="labels",
            evaluators="default",
            evaluator_config={
                "log_model_explainability": True,
                # Use the kernel explainability algorithm, which fails if there is a mismatch
                # between the number of features in the input dataset and the number of features
                # expected by the model
                "explainability_algorithm": "kernel",
            },
        )

        _, _, _, artifacts = get_run_data(run.info.run_id)
        assert set(artifacts) >= {
            "shap_beeswarm_plot.png",
            "shap_feature_importance_plot.png",
            "shap_summary_plot.png",
        }


@pytest.mark.parametrize("prefix", ["train_", None])
def test_evaluation_metric_name_configs(prefix):
    X, y = load_iris(as_frame=True, return_X_y=True)
    with mlflow.start_run() as run:
        model = LogisticRegression()
        model.fit(X, y)
        model_info = mlflow.sklearn.log_model(model, "model")
        result = evaluate(
            model_info.model_uri,
            X.assign(target=y),
            model_type="classifier" if isinstance(model, LogisticRegression) else "regressor",
            targets="target",
            evaluators="default",
            evaluator_config={"metric_prefix": prefix},
        )

    _, metrics, _, _ = get_run_data(run.info.run_id)
    assert len(metrics) > 0

    if prefix is not None:
        assert f"{prefix}accuracy_score" in metrics
        assert f"{prefix}log_loss" in metrics
        assert f"{prefix}score" in metrics

        assert f"{prefix}accuracy_score" in result.metrics
        assert f"{prefix}log_loss" in result.metrics
        assert f"{prefix}score" in result.metrics


@pytest.mark.parametrize(
    "env_manager",
    ["virtualenv", "conda"],
)
def test_evaluation_with_env_restoration(
    multiclass_logistic_regressor_model_uri, iris_dataset, env_manager
):
    with mlflow.start_run() as run:
        result = evaluate(
            model=multiclass_logistic_regressor_model_uri,
            data=iris_dataset._constructor_args["data"],
            model_type="classifier",
            targets=iris_dataset._constructor_args["targets"],
            evaluators="default",
            env_manager=env_manager,
        )

    _, metrics, _, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(multiclass_logistic_regressor_model_uri)
    y = iris_dataset.labels_data
    y_pred = model.predict(iris_dataset.features_data)

    expected_metrics = _get_multiclass_classifier_metrics(y_true=y, y_pred=y_pred, y_proba=None)

    for metric_key, expected_metric_val in expected_metrics.items():
        assert np.isclose(expected_metric_val, metrics[metric_key], rtol=1e-3)
        assert np.isclose(expected_metric_val, result.metrics[metric_key], rtol=1e-3)

    assert set(artifacts) == {
        "per_class_metrics.csv",
        "confusion_matrix.png",
    }
    assert result.artifacts.keys() == {
        "per_class_metrics",
        "confusion_matrix",
    }


@pytest.mark.parametrize("pos_label", [None, 0, 1, 2])
def test_evaluation_binary_classification_with_pos_label(pos_label):
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    X = X.iloc[:, :4].head(100)
    y = y.head(len(X))
    if pos_label == 2:
        y = [2 if trg == 1 else trg for trg in y]
    elif pos_label is None:
        # Use a different positive class other than the 1 to verify
        # that an unspecified `pos_label` doesn't cause problems
        # for binary classification tasks with nonstandard labels
        y = [10 if trg == 1 else trg for trg in y]
    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X, y)
        model_info = mlflow.sklearn.log_model(model, "model")
        result = evaluate(
            model_info.model_uri,
            X.assign(target=y),
            model_type="classifier",
            targets="target",
            evaluators="default",
            evaluator_config=None if pos_label is None else {"pos_label": pos_label},
        )
        y_pred = model.predict(X)
        pl = 10 if pos_label is None else pos_label
        precision = precision_score(y, y_pred, pos_label=pl)
        recall = recall_score(y, y_pred, pos_label=pl)
        f1 = f1_score(y, y_pred, pos_label=pl)
        np.testing.assert_allclose(result.metrics["precision_score"], precision)
        np.testing.assert_allclose(result.metrics["recall_score"], recall)
        np.testing.assert_allclose(result.metrics["f1_score"], f1)


@pytest.mark.parametrize("average", [None, "weighted", "macro", "micro"])
def test_evaluation_multiclass_classification_with_average(average):
    X, y = load_iris(as_frame=True, return_X_y=True)
    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X, y)
        model_info = mlflow.sklearn.log_model(model, "model")
        result = evaluate(
            model_info.model_uri,
            X.assign(target=y),
            model_type="classifier",
            targets="target",
            evaluators="default",
            evaluator_config=None if average is None else {"average": average},
        )
        y_pred = model.predict(X)
        avg = average or "weighted"
        precision = precision_score(y, y_pred, average=avg)
        recall = recall_score(y, y_pred, average=avg)
        f1 = f1_score(y, y_pred, average=avg)
        np.testing.assert_allclose(result.metrics["precision_score"], precision)
        np.testing.assert_allclose(result.metrics["recall_score"], recall)
        np.testing.assert_allclose(result.metrics["f1_score"], f1)


def test_custom_metrics():
    X, y = load_iris(as_frame=True, return_X_y=True)
    with mlflow.start_run():
        model = LogisticRegression().fit(X, y)
        model_info = mlflow.sklearn.log_model(model, "model")
        result = evaluate(
            model_info.model_uri,
            X.assign(target=y),
            model_type="classifier",
            targets="target",
            evaluators="default",
            extra_metrics=[
                make_metric(
                    eval_fn=lambda _eval_df, _builtin_metrics: MetricValue(
                        aggregate_results={"cm": 1.0}
                    ),
                    name="cm",
                    greater_is_better=True,
                    long_name="custom_metric",
                )
            ],
            evaluator_config={"log_model_explainability": False},  # For faster evaluation
        )
        np.testing.assert_allclose(result.metrics["cm"], 1.0)


def test_custom_artifacts():
    X, y = load_iris(as_frame=True, return_X_y=True)
    with mlflow.start_run():
        model = LogisticRegression().fit(X, y)
        model_info = mlflow.sklearn.log_model(model, "model")
        result = evaluate(
            model_info.model_uri,
            X.assign(target=y),
            model_type="classifier",
            targets="target",
            evaluators="default",
            custom_artifacts=[
                lambda *_args, **_kwargs: {"custom_artifact": {"k": "v"}},
            ],
            evaluator_config={"log_model_explainability": False},  # For faster evaluation
        )
        custom_artifact = result.artifacts["custom_artifact"]
        assert json.loads(Path(custom_artifact.uri).read_text()) == {"k": "v"}


def test_make_metric_name_inference():
    def metric(_df, _metrics):
        return 1

    metric = make_metric(eval_fn=metric, greater_is_better=True)
    assert metric.name == "metric"

    metric = make_metric(eval_fn=metric, greater_is_better=True, name="my_metric")
    assert metric.name == "my_metric"

    metric = make_metric(eval_fn=lambda _df, _metrics: 0, greater_is_better=True, name="metric")
    assert metric.name == "metric"

    with pytest.raises(
        MlflowException, match="`name` must be specified if `eval_fn` is a lambda function."
    ):
        make_metric(eval_fn=lambda _df, _metrics: 0, greater_is_better=True)

    class Callable:
        def __call__(self, _df, _metrics):
            return 1

    with pytest.raises(
        MlflowException,
        match="`name` must be specified if `eval_fn` does not have a `__name__` attribute.",
    ):
        make_metric(eval_fn=Callable(), greater_is_better=True)


def language_model(inputs: list[str]) -> list[str]:
    return inputs


def validate_question_answering_logged_data(
    logged_data, with_targets=True, predictions_name="outputs"
):
    columns = {
        "question",
        predictions_name,
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "token_count",
    }
    if with_targets:
        columns.update({"answer"})

    assert set(logged_data.columns.tolist()) == columns

    assert logged_data["question"].tolist() == ["words random", "This is a sentence."]
    assert logged_data[predictions_name].tolist() == ["words random", "This is a sentence."]
    assert logged_data["toxicity/v1/score"][0] < 0.5
    assert logged_data["toxicity/v1/score"][1] < 0.5
    assert all(
        isinstance(grade, float) for grade in logged_data["flesch_kincaid_grade_level/v1/score"]
    )
    assert all(isinstance(grade, float) for grade in logged_data["ari_grade_level/v1/score"])
    assert all(isinstance(grade, int) for grade in logged_data["token_count"])

    if with_targets:
        assert logged_data["answer"].tolist() == ["words random", "This is a sentence."]


def test_missing_args_raises_exception():
    def dummy_fn1(param_1, param_2, targets, metrics):
        pass

    def dummy_fn2(param_3, param_4, builtin_metrics):
        pass

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"question": ["a", "b"], "answer": ["a", "b"]})

    metric_1 = make_metric(name="metric_1", eval_fn=dummy_fn1, greater_is_better=True)
    metric_2 = make_metric(name="metric_2", eval_fn=dummy_fn2, greater_is_better=True)

    error_message = (
        r"Error: Metric calculation failed for the following metrics:\n"
        r"Metric 'metric_1' requires the columns \['param_1', 'param_2'\]\n"
        r"Metric 'metric_2' requires the columns \['param_3', 'builtin_metrics'\]\n\n"
        r"Below are the existing column names for the input/output data:\n"
        r"Input Columns: \['question', 'answer'\]\n"
        r"Output Columns: \['predictions'\]\n"
        r"To resolve this issue, you may want to map the missing column to an existing column\n"
        r"using the following configuration:\n"
        r"evaluator_config=\{'col_mapping': \{<missing column name>: <existing column name>\}\}"
    )

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                model_info.model_uri,
                data,
                targets="answer",
                evaluators="default",
                model_type="question-answering",
                extra_metrics=[metric_1, metric_2],
                evaluator_config={"col_mapping": {"param_4": "question"}},
            )


def test_custom_metrics_deprecated(
    binary_logistic_regressor_model_uri,
    breast_cancer_dataset,
):
    def dummy_fn(eval_df, metrics):
        pass

    with pytest.raises(
        MlflowException,
        match="The 'custom_metrics' parameter in mlflow.evaluate is deprecated. Please update "
        "your code to only use the 'extra_metrics' parameter instead.",
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                binary_logistic_regressor_model_uri,
                breast_cancer_dataset._constructor_args["data"],
                targets=breast_cancer_dataset._constructor_args["targets"],
                evaluators="default",
                model_type="classifier",
                custom_metrics=[make_metric(eval_fn=dummy_fn, greater_is_better=True)],
                extra_metrics=[make_metric(eval_fn=dummy_fn, greater_is_better=True)],
            )

    message = "The 'custom_metrics' parameter in mlflow.evaluate is deprecated. Please update your "
    "code to use the 'extra_metrics' parameter instead."
    with pytest.warns(FutureWarning, match=message):
        with mlflow.start_run():
            mlflow.evaluate(
                binary_logistic_regressor_model_uri,
                breast_cancer_dataset._constructor_args["data"],
                targets=breast_cancer_dataset._constructor_args["targets"],
                evaluators="default",
                model_type="classifier",
                custom_metrics=[make_metric(eval_fn=dummy_fn, greater_is_better=True)],
            )


def test_evaluate_question_answering_with_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "answer": ["words random", "This is a sentence."],
            }
        )
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="answer",
            model_type="question-answering",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_question_answering_logged_data(logged_data)
    assert set(results.metrics.keys()) == set(
        get_question_answering_metrics_keys(with_targets=True)
    )
    assert results.metrics["exact_match/v1"] == 1.0


def test_evaluate_question_answering_on_static_dataset_with_targets():
    with mlflow.start_run() as run:
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "answer": ["words random", "This is a sentence."],
                "pred": ["words random", "This is a sentence."],
            }
        )
        results = mlflow.evaluate(
            data=data,
            targets="answer",
            predictions="pred",
            model_type="question-answering",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_question_answering_logged_data(logged_data, predictions_name="pred")
    assert set(results.metrics.keys()) == {
        "toxicity/v1/variance",
        "toxicity/v1/ratio",
        "toxicity/v1/mean",
        "flesch_kincaid_grade_level/v1/variance",
        "ari_grade_level/v1/p90",
        "flesch_kincaid_grade_level/v1/p90",
        "flesch_kincaid_grade_level/v1/mean",
        "ari_grade_level/v1/mean",
        "ari_grade_level/v1/variance",
        "exact_match/v1",
        "toxicity/v1/p90",
    }
    assert results.metrics["exact_match/v1"] == 1.0
    assert results.metrics["toxicity/v1/ratio"] == 0.0


def question_classifier(inputs):
    return inputs["question"].map({"a": 0, "b": 1})


def test_evaluate_question_answering_with_numerical_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=question_classifier, input_example=[0, 1]
        )
        data = pd.DataFrame({"question": ["a", "b"], "answer": [0, 1]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="answer",
            model_type="question-answering",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    pd.testing.assert_frame_equal(
        logged_data.drop("token_count", axis=1),
        data.assign(outputs=[0, 1]),
    )
    assert results.metrics == {"exact_match/v1": 1.0}


def test_evaluate_question_answering_without_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"question": ["words random", "This is a sentence."]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="question-answering",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_question_answering_logged_data(logged_data, False)
    assert set(results.metrics.keys()) == set(
        get_question_answering_metrics_keys(with_targets=False)
    )


def validate_text_summarization_logged_data(logged_data, with_targets=True):
    columns = {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "token_count",
    }
    if with_targets:
        columns.update(
            {
                "summary",
                "rouge1/v1/score",
                "rouge2/v1/score",
                "rougeL/v1/score",
                "rougeLsum/v1/score",
            }
        )

    assert set(logged_data.columns.tolist()) == columns

    assert logged_data["text"].tolist() == ["a", "b"]
    assert logged_data["outputs"].tolist() == ["a", "b"]
    assert logged_data["toxicity/v1/score"][0] < 0.5
    assert logged_data["toxicity/v1/score"][1] < 0.5
    assert all(
        isinstance(grade, float) for grade in logged_data["flesch_kincaid_grade_level/v1/score"]
    )
    assert all(isinstance(grade, float) for grade in logged_data["ari_grade_level/v1/score"])
    assert all(isinstance(grade, int) for grade in logged_data["token_count"])

    if with_targets:
        assert logged_data["summary"].tolist() == ["a", "b"]
        assert logged_data["rouge1/v1/score"].tolist() == [1.0, 1.0]
        assert logged_data["rouge2/v1/score"].tolist() == [0.0, 0.0]
        assert logged_data["rougeL/v1/score"].tolist() == [1.0, 1.0]
        assert logged_data["rougeLsum/v1/score"].tolist() == [1.0, 1.0]


def get_text_metrics_keys():
    metric_names = ["toxicity", "flesch_kincaid_grade_level", "ari_grade_level"]
    standard_aggregations = ["mean", "variance", "p90"]
    version = "v1"

    metrics_keys = [
        f"{metric}/{version}/{agg}" for metric in metric_names for agg in standard_aggregations
    ]
    additional_aggregations = ["toxicity/v1/ratio"]
    return metrics_keys + additional_aggregations


def get_text_summarization_metrics_keys(with_targets=False):
    if with_targets:
        metric_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        standard_aggregations = ["mean", "variance", "p90"]
        version = "v1"

        metrics_keys = [
            f"{metric}/{version}/{agg}" for metric in metric_names for agg in standard_aggregations
        ]
    else:
        metrics_keys = []
    return get_text_metrics_keys() + metrics_keys


def get_question_answering_metrics_keys(with_targets=False):
    metrics_keys = ["exact_match/v1"] if with_targets else []
    return get_text_metrics_keys() + metrics_keys


def test_evaluate_text_summarization_with_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"], "summary": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="summary",
            model_type="text-summarization",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_text_summarization_logged_data(logged_data)

    metrics = results.metrics
    assert set(metrics.keys()) == set(get_text_summarization_metrics_keys(with_targets=True))


def test_evaluate_text_summarization_with_targets_no_type_hints():
    def another_language_model(x):
        x.rename(columns={"text": "outputs"}, inplace=True)
        return x

    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=another_language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"], "summary": ["a", "b"]})
        results = evaluate(
            model_info.model_uri,
            data,
            targets="summary",
            model_type="text-summarization",
            evaluators="default",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_text_summarization_logged_data(logged_data)

    metrics = results.metrics
    assert set(metrics.keys()) == set(get_text_summarization_metrics_keys(with_targets=True))


def test_evaluate_text_summarization_without_targets():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text-summarization",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_text_summarization_logged_data(logged_data, with_targets=False)

    assert set(results.metrics.keys()) == set(
        get_text_summarization_metrics_keys(with_targets=False)
    )


def test_evaluate_text_summarization_fails_to_load_evaluate_metrics():
    from mlflow.metrics.metric_definitions import _cached_evaluate_load

    _cached_evaluate_load.cache_clear()

    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )

        data = pd.DataFrame({"text": ["a", "b"], "summary": ["a", "b"]})
        with mock.patch(
            "mlflow.metrics.metric_definitions._cached_evaluate_load",
            side_effect=ImportError("mocked error"),
        ) as mock_load:
            results = mlflow.evaluate(
                model_info.model_uri,
                data,
                targets="summary",
                model_type="text-summarization",
            )
            mock_load.assert_any_call("rouge")
            mock_load.assert_any_call("toxicity", module_type="measurement")

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "summary",
        "outputs",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "token_count",
    }
    assert logged_data["text"].tolist() == ["a", "b"]
    assert logged_data["summary"].tolist() == ["a", "b"]
    assert logged_data["outputs"].tolist() == ["a", "b"]


def test_evaluate_text_and_text_metrics():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["sentence not", "All women are bad."]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "token_count",
    }
    assert logged_data["text"].tolist() == ["sentence not", "All women are bad."]
    assert logged_data["outputs"].tolist() == ["sentence not", "All women are bad."]
    # Hateful sentiments should be marked as toxic
    assert logged_data["toxicity/v1/score"][0] < 0.5
    assert logged_data["toxicity/v1/score"][1] > 0.5
    # Simple sentences should have a low grade level.
    assert logged_data["flesch_kincaid_grade_level/v1/score"][1] < 4
    assert logged_data["ari_grade_level/v1/score"][1] < 4
    assert set(results.metrics.keys()) == set(get_text_metrics_keys())


def very_toxic(predictions, targets=None, metrics=None):
    new_scores = [1.0 if score > 0.9 else 0.0 for score in metrics["toxicity/v1"].scores]
    return MetricValue(
        scores=new_scores,
        justifications=["toxic" if score == 1.0 else "not toxic" for score in new_scores],
        aggregate_results={"ratio": sum(new_scores) / len(new_scores)},
    )


def per_row_metric(predictions, targets=None, metrics=None):
    return MetricValue(scores=[1] * len(predictions))


def test_evaluate_text_custom_metrics():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"], "target": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="target",
            model_type="text",
            extra_metrics=[
                make_metric(eval_fn=very_toxic, greater_is_better=True, version="v2"),
                make_metric(eval_fn=per_row_metric, greater_is_better=False, name="no_version"),
            ],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)

    assert "very_toxic/v2/score" in logged_data.columns.tolist()
    assert "very_toxic/v2/justification" in logged_data.columns.tolist()
    assert all(isinstance(score, float) for score in logged_data["very_toxic/v2/score"])
    assert all(
        isinstance(justification, str)
        for justification in logged_data["very_toxic/v2/justification"]
    )
    assert "very_toxic/v2/ratio" in set(results.metrics.keys())
    assert "no_version/score" in logged_data.columns.tolist()


@pytest.mark.parametrize("metric_prefix", ["train_", None])
def test_eval_results_table_json_can_be_prefixed_with_metric_prefix(metric_prefix):
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["a", "b"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text",
            evaluators="default",
            evaluator_config={
                "metric_prefix": metric_prefix,
            },
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]

    if metric_prefix is None:
        metric_prefix = ""

    assert f"{metric_prefix}eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        f"{metric_prefix}toxicity/v1/score",
        f"{metric_prefix}flesch_kincaid_grade_level/v1/score",
        f"{metric_prefix}ari_grade_level/v1/score",
        f"{metric_prefix}token_count",
    }


@pytest.mark.parametrize(
    "baseline_model_uri",
    [("svm_model_uri")],
    indirect=["baseline_model_uri"],
)
def test_default_evaluator_for_pyfunc_model(baseline_model_uri, breast_cancer_dataset):
    data = load_breast_cancer()
    raw_model = LinearSVC()
    raw_model.fit(data.data, data.target)

    mlflow_model = Model()
    mlflow.pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = mlflow.pyfunc.PyFuncModel(model_meta=mlflow_model, model_impl=raw_model)

    with mlflow.start_run() as run:
        evaluate_model_helper(
            pyfunc_model,
            baseline_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
            eval_baseline_model_only=False,
        )
    run_data = get_run_data(run.info.run_id)
    assert set(run_data.artifacts) == {
        "confusion_matrix.png",
        "shap_feature_importance_plot.png",
        "shap_beeswarm_plot.png",
        "shap_summary_plot.png",
    }


def test_extracting_output_and_other_columns():
    data_dict = {
        "text": ["text_a", "text_b"],
        "target": ["target_a", "target_b"],
        "other": ["other_a", "other_b"],
    }
    data_df = pd.DataFrame(data_dict)
    data_list_dict = [
        {
            "text": "text_a",
            "target": "target_a",
            "other": "other_a",
        },
        {
            "text": "text_b",
            "target": "target_b",
            "other": "other_b",
        },
    ]
    data_list = (["data_a", "data_b"],)
    data_dict_text = {
        "text": ["text_a", "text_b"],
    }

    output1, other1, prediction_col1 = _extract_output_and_other_columns(data_dict, "target")
    output2, other2, prediction_col2 = _extract_output_and_other_columns(data_df, "target")
    output3, other3, prediction_col3 = _extract_output_and_other_columns(data_list_dict, "target")
    output4, other4, prediction_col4 = _extract_output_and_other_columns(data_list, None)
    output5, other5, prediction_col5 = _extract_output_and_other_columns(pd.Series(data_list), None)
    output6, other6, prediction_col6 = _extract_output_and_other_columns(data_dict_text, None)
    output7, other7, prediction_col7 = _extract_output_and_other_columns(
        pd.DataFrame(data_dict_text), None
    )

    assert output1.equals(data_df["target"])
    assert other1.equals(data_df.drop(columns=["target"]))
    assert prediction_col1 == "target"
    assert output2.equals(data_df["target"])
    assert other2.equals(data_df.drop(columns=["target"]))
    assert prediction_col2 == "target"
    assert output3.equals(data_df["target"])
    assert other3.equals(data_df.drop(columns=["target"]))
    assert prediction_col3 == "target"
    assert output4 == data_list
    assert other4 is None
    assert prediction_col4 is None
    assert output5.equals(pd.Series(data_list))
    assert other5 is None
    assert prediction_col5 is None
    assert output6.equals(pd.Series(data_dict_text["text"]))
    assert other6 is None
    assert prediction_col6 == "text"
    assert output7.equals(pd.Series(data_dict_text["text"]))
    assert other7 is None
    assert prediction_col7 == "text"


def language_model_with_context(inputs: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "context": f"context_{input}",
            "output": input,
        }
        for input in inputs
    ]


def test_constructing_eval_df_for_custom_metrics():
    test_eval_df_value = pd.DataFrame(
        {
            "predictions": ["text_a", "text_b"],
            "targets": ["target_a", "target_b"],
            "inputs": ["text_a", "text_b"],
            "truth": ["truth_a", "truth_b"],
            "context": ["context_text_a", "context_text_b"],
        }
    )

    def example_custom_artifact(_, __, ___):
        return {"test_json_artifact": {"a": 2, "b": [1, 2]}}

    def test_eval_df(predictions, targets, metrics, inputs, truth, context):
        global eval_df_value
        eval_df_value = pd.DataFrame(
            {
                "predictions": predictions,
                "targets": targets,
                "inputs": inputs,
                "truth": truth,
                "context": context,
            }
        )
        return predictions.eq(targets).mean()

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=language_model_with_context,
            input_example=["a", "b"],
        )
        data = pd.DataFrame(
            {
                "text": ["text_a", "text_b"],
                "truth": ["truth_a", "truth_b"],
                "targets": ["target_a", "target_b"],
            }
        )
        eval_results = mlflow.evaluate(
            model_info.model_uri,
            data,
            targets="targets",
            predictions="output",
            model_type="text",
            extra_metrics=[make_metric(eval_fn=test_eval_df, greater_is_better=True)],
            custom_artifacts=[example_custom_artifact],
            evaluators="default",
            evaluator_config={"col_mapping": {"inputs": "text"}},
        )

    assert eval_df_value.equals(test_eval_df_value)
    assert len(eval_results.artifacts) == 2
    assert len(eval_results.tables) == 1
    assert eval_results.tables["eval_results_table"].columns.tolist() == [
        "text",
        "truth",
        "targets",
        "output",
        "context",
        "token_count",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
    ]


def test_evaluate_no_model_type():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        with pytest.raises(
            MlflowException,
            match="The extra_metrics argument must be specified model_type is None.",
        ):
            mlflow.evaluate(
                model_info.model_uri,
                data,
            )


def test_evaluate_no_model_type_with_builtin_metric():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            extra_metrics=[mlflow.metrics.toxicity()],
        )
        assert results.metrics.keys() == {
            "toxicity/v1/mean",
            "toxicity/v1/variance",
            "toxicity/v1/p90",
            "toxicity/v1/ratio",
        }
        assert len(results.tables) == 1
        assert results.tables["eval_results_table"].columns.tolist() == [
            "text",
            "outputs",
            "toxicity/v1/score",
        ]


def test_evaluate_no_model_type_with_custom_metric():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        from mlflow.metrics import make_metric
        from mlflow.metrics.base import standard_aggregations

        def word_count_eval(predictions, targets=None, metrics=None):
            scores = []
            for prediction in predictions:
                scores.append(len(prediction.split(" ")))
            return MetricValue(
                scores=scores,
                aggregate_results=standard_aggregations(scores),
            )

        word_count = make_metric(eval_fn=word_count_eval, greater_is_better=True, name="word_count")

        results = mlflow.evaluate(model_info.model_uri, data, extra_metrics=[word_count])
        assert results.metrics.keys() == {
            "word_count/mean",
            "word_count/variance",
            "word_count/p90",
        }
        assert results.metrics["word_count/mean"] == 3.0
        assert len(results.tables) == 1
        assert results.tables["eval_results_table"].columns.tolist() == [
            "text",
            "outputs",
            "word_count/score",
        ]


def multi_output_model(inputs):
    return pd.DataFrame(
        {
            "answer": ["words random", "This is a sentence."],
            "source": ["words random", "This is a sentence."],
        }
    )


def test_default_metrics_as_custom_metrics():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=multi_output_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "truth": ["words random", "This is a sentence."],
            }
        )
        results = evaluate(
            model_info.model_uri,
            data,
            targets="truth",
            predictions="answer",
            model_type="question-answering",
            custom_metrics=[
                mlflow.metrics.exact_match(),
            ],
            evaluators="default",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    assert "exact_match/v1" in results.metrics.keys()


def test_default_metrics_as_custom_metrics_static_dataset():
    with mlflow.start_run() as run:
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "truth": ["words random", "This is a sentence."],
                "answer": ["words random", "This is a sentence."],
                "source": ["words random", "This is a sentence."],
            }
        )
        results = evaluate(
            data=data,
            targets="truth",
            predictions="answer",
            model_type="question-answering",
            custom_metrics=[
                mlflow.metrics.flesch_kincaid_grade_level(),
                mlflow.metrics.ari_grade_level(),
                mlflow.metrics.toxicity(),
                mlflow.metrics.exact_match(),
            ],
            evaluators="default",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    for metric in ["toxicity", "ari_grade_level", "flesch_kincaid_grade_level"]:
        for measure in ["mean", "p90", "variance"]:
            assert f"{metric}/v1/{measure}" in results.metrics.keys()
    assert "exact_match/v1" in results.metrics.keys()


def test_multi_output_model_error_handling():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=multi_output_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "truth": ["words random", "This is a sentence."],
            }
        )
        with pytest.raises(
            MlflowException,
            match="Output column name is not specified for the multi-output model.",
        ):
            evaluate(
                model_info.model_uri,
                data,
                targets="truth",
                model_type="question-answering",
                custom_metrics=[
                    mlflow.metrics.flesch_kincaid_grade_level(),
                    mlflow.metrics.ari_grade_level(),
                    mlflow.metrics.toxicity(),
                    mlflow.metrics.exact_match(),
                ],
                evaluators="default",
            )


def test_invalid_extra_metrics():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        with pytest.raises(
            MlflowException,
            match="Please ensure that all extra metrics are instances of "
            "mlflow.metrics.EvaluationMetric.",
        ):
            mlflow.evaluate(
                model_info.model_uri,
                data,
                model_type="text",
                evaluators="default",
                extra_metrics=[mlflow.metrics.latency],
            )


def test_evaluate_with_latency():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["sentence not", "Hello world."]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            model_type="text",
            evaluators="default",
            extra_metrics=[mlflow.metrics.latency()],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "latency",
        "token_count",
    }
    assert all(isinstance(grade, float) for grade in logged_data["latency"])


def test_evaluate_with_latency_static_dataset():
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame(
            {
                "text": ["foo", "bar"],
                "model_output": ["FOO", "BAR"],
            }
        )
        results = mlflow.evaluate(
            data=data,
            model_type="text",
            evaluators="default",
            predictions="model_output",
            extra_metrics=[mlflow.metrics.latency()],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "text",
        "outputs",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "latency",
        "token_count",
    }
    assert all(isinstance(grade, float) for grade in logged_data["latency"])
    assert all(grade == 0.0 for grade in logged_data["latency"])


properly_formatted_openai_response1 = (
    '{\n  "score": 3,\n  "justification": "' "justification" '"\n}'
)


def test_evaluate_with_correctness():
    metric = mlflow.metrics.genai.make_genai_metric(
        name="correctness",
        definition=(
            "Correctness refers to how well the generated output matches "
            "or aligns with the reference or ground truth text that is considered "
            "accurate and appropriate for the given input. The ground truth serves as "
            "a benchmark against which the provided output is compared to determine the "
            "level of accuracy and fidelity."
        ),
        grading_prompt=(
            "Correctness: If the answer correctly answer the question, below "
            "are the details for different scores: "
            "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
            "the question or is completely contrary to the correct answer. "
            "- Score 1: the answer provides some relevance to the question and answer "
            "one aspect of the question correctly. "
            "- Score 2: the answer mostly answer the question but is missing or hallucinating "
            "on one critical aspect. "
            "- Score 4: the answer correctly answer the question and not missing any "
            "major aspect"
        ),
        examples=[],
        version="v1",
        model="openai:/gpt-3.5-turbo-16k",
        grading_context_columns=["ground_truth"],
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ):
        with mlflow.start_run():
            eval_df = pd.DataFrame(
                {
                    "inputs": [
                        "What is MLflow?",
                        "What is Spark?",
                        "What is Python?",
                    ],
                    "ground_truth": [
                        "MLflow is an open-source platform",
                        "Apache Spark is an open-source, distributed computing system",
                        "Python is a high-level programming language",
                    ],
                    "prediction": [
                        "MLflow is an open-source platform",
                        "Apache Spark is an open-source, distributed computing system",
                        "Python is a high-level programming language",
                    ],
                }
            )
            results = mlflow.evaluate(
                data=eval_df,
                evaluators="default",
                targets="ground_truth",
                predictions="prediction",
                extra_metrics=[metric],
            )

            assert results.metrics == {
                "correctness/v1/mean": 3.0,
                "correctness/v1/variance": 0.0,
                "correctness/v1/p90": 3.0,
            }


def test_evaluate_custom_metrics_string_values():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            extra_metrics=[
                make_metric(
                    eval_fn=lambda predictions, metrics, eval_config: MetricValue(
                        aggregate_results={"eval_config_value_average": eval_config}
                    ),
                    name="cm",
                    greater_is_better=True,
                    long_name="custom_metric",
                )
            ],
            evaluators="default",
            evaluator_config={"eval_config": 3},
        )
        assert results.metrics["cm/eval_config_value_average"] == 3


def validate_retriever_logged_data(logged_data, k=3):
    columns = {
        "question",
        "retrieved_context",
        f"precision_at_{k}/score",
        f"recall_at_{k}/score",
        f"ndcg_at_{k}/score",
        "ground_truth",
    }

    assert set(logged_data.columns.tolist()) == columns

    assert logged_data["question"].tolist() == ["q1?", "q1?", "q1?"]
    assert logged_data["retrieved_context"].tolist() == [["doc1", "doc3", "doc2"]] * 3
    assert (logged_data[f"precision_at_{k}/score"] <= 1).all()
    assert (logged_data[f"recall_at_{k}/score"] <= 1).all()
    assert (logged_data[f"ndcg_at_{k}/score"] <= 1).all()
    assert logged_data["ground_truth"].tolist() == [["doc1", "doc2"]] * 3


def test_evaluate_retriever():
    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [["doc1", "doc2"]] * 3})

    def fn(X):
        return pd.DataFrame({"retrieved_context": [["doc1", "doc3", "doc2"]] * len(X)})

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluators="default",
            evaluator_config={
                "k": 3,
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 2 / 3,
        "precision_at_3/variance": 0,
        "precision_at_3/p90": 2 / 3,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_3/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_3/variance": 0.0,
    }
    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_retriever_logged_data(logged_data)

    # test with a big k to ensure we use min(k, len(retrieved_chunks))
    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluators="default",
            evaluator_config={
                "retriever_k": 6,
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_6/mean": 2 / 3,
        "precision_at_6/variance": 0,
        "precision_at_6/p90": 2 / 3,
        "recall_at_6/mean": 1.0,
        "recall_at_6/p90": 1.0,
        "recall_at_6/variance": 0.0,
        "ndcg_at_6/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_6/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_6/variance": 0.0,
    }

    # test with default k
    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            model_type="retriever",
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 2 / 3,
        "precision_at_3/variance": 0,
        "precision_at_3/p90": 2 / 3,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_3/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_3/variance": 0.0,
    }

    # test with multiple chunks from same doc
    def fn2(X):
        return pd.DataFrame({"retrieved_context": [["doc1", "doc1", "doc3"]] * len(X)})

    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [["doc1", "doc3"]] * 3})

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn2,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluator_config={
                "default": {
                    "retriever_k": 3,
                }
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 1,
        "precision_at_3/p90": 1,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": 1.0,
        "ndcg_at_3/p90": 1.0,
        "ndcg_at_3/variance": 0.0,
    }

    # test with empty retrieved doc
    def fn3(X):
        return pd.DataFrame({"output": [[]] * len(X)})

    with mlflow.start_run() as run:
        mlflow.evaluate(
            model=fn3,
            data=X,
            targets="ground_truth",
            model_type="retriever",
            evaluator_config={
                "default": {
                    "retriever_k": 4,
                }
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_4/mean": 0,
        "precision_at_4/p90": 0,
        "precision_at_4/variance": 0,
        "recall_at_4/mean": 0,
        "recall_at_4/p90": 0,
        "recall_at_4/variance": 0,
        "ndcg_at_4/mean": 0.0,
        "ndcg_at_4/p90": 0.0,
        "ndcg_at_4/variance": 0.0,
    }

    # test with a static dataset
    X_1 = pd.DataFrame(
        {
            "question": [["q1?"]] * 3,
            "targets_param": [["doc1", "doc2"]] * 3,
            "predictions_param": [["doc1", "doc4", "doc5"]] * 3,
        }
    )
    with mlflow.start_run() as run:
        mlflow.evaluate(
            data=X_1,
            predictions="predictions_param",
            targets="targets_param",
            model_type="retriever",
            extra_metrics=[mlflow.metrics.precision_at_k(4), mlflow.metrics.recall_at_k(4)],
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 1 / 3,
        "precision_at_3/p90": 1 / 3,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 0.5,
        "recall_at_3/p90": 0.5,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.6131471927654585),
        "ndcg_at_3/p90": pytest.approx(0.6131471927654585),
        "ndcg_at_3/variance": 0.0,
        "precision_at_4/mean": 1 / 3,
        "precision_at_4/p90": 1 / 3,
        "precision_at_4/variance": 0.0,
        "recall_at_4/mean": 0.5,
        "recall_at_4/p90": 0.5,
        "recall_at_4/variance": 0.0,
    }

    # test to make sure it silently fails with invalid k
    with mlflow.start_run() as run:
        mlflow.evaluate(
            data=X_1,
            predictions="predictions_param",
            targets="targets_param",
            model_type="retriever",
            extra_metrics=[mlflow.metrics.precision_at_k(-1)],
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {
        "precision_at_3/mean": 1 / 3,
        "precision_at_3/p90": 1 / 3,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 0.5,
        "recall_at_3/p90": 0.5,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.6131471927654585),
        "ndcg_at_3/p90": pytest.approx(0.6131471927654585),
        "ndcg_at_3/variance": 0.0,
    }

    # silent failure with evaluator_config method too!
    with mlflow.start_run() as run:
        mlflow.evaluate(
            data=X_1,
            predictions="predictions_param",
            targets="targets_param",
            model_type="retriever",
            evaluators="default",
            evaluator_config={
                "retriever_k": -1,
            },
        )
    run = mlflow.get_run(run.info.run_id)
    assert run.data.metrics == {}


def test_evaluate_retriever_builtin_metrics_no_model_type():
    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [["doc1", "doc2"]] * 3})

    def fn(X):
        return {"retrieved_context": [["doc1", "doc3", "doc2"]] * len(X)}

    with mlflow.start_run() as run:
        results = mlflow.evaluate(
            model=fn,
            data=X,
            targets="ground_truth",
            extra_metrics=[
                mlflow.metrics.precision_at_k(4),
                mlflow.metrics.recall_at_k(4),
                mlflow.metrics.ndcg_at_k(4),
            ],
        )
    run = mlflow.get_run(run.info.run_id)
    assert (
        run.data.metrics
        == results.metrics
        == {
            "precision_at_4/mean": 2 / 3,
            "precision_at_4/p90": 2 / 3,
            "precision_at_4/variance": 0.0,
            "recall_at_4/mean": 1.0,
            "recall_at_4/p90": 1.0,
            "recall_at_4/variance": 0.0,
            "ndcg_at_4/mean": pytest.approx(0.9197207891481877),
            "ndcg_at_4/p90": pytest.approx(0.9197207891481877),
            "ndcg_at_4/variance": 0.0,
        }
    )
    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    validate_retriever_logged_data(logged_data, 4)


def test_evaluate_with_numpy_array():
    data = [
        ["What is MLflow?"],
    ]
    ground_truth = [
        "MLflow is an open-source platform for managing the end-to-end machine learning",
    ]

    with mlflow.start_run():
        logged_model = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        results = mlflow.evaluate(
            logged_model.model_uri,
            data,
            targets=ground_truth,
            extra_metrics=[mlflow.metrics.toxicity()],
        )

        assert results.metrics.keys() == {
            "toxicity/v1/mean",
            "toxicity/v1/variance",
            "toxicity/v1/p90",
            "toxicity/v1/ratio",
        }
        assert len(results.tables) == 1
        assert results.tables["eval_results_table"].columns.tolist() == [
            "feature_1",
            "target",
            "outputs",
            "toxicity/v1/score",
        ]


def test_target_prediction_col_mapping():
    metric = mlflow.metrics.genai.make_genai_metric(
        name="correctness",
        definition=(
            "Correctness refers to how well the generated output matches "
            "or aligns with the reference or ground truth text that is considered "
            "accurate and appropriate for the given input. The ground truth serves as "
            "a benchmark against which the provided output is compared to determine the "
            "level of accuracy and fidelity."
        ),
        grading_prompt=(
            "Correctness: If the answer correctly answer the question, below "
            "are the details for different scores: "
            "- Score 0: the answer is completely incorrect, doesn't mention anything about "
            "the question or is completely contrary to the correct answer. "
            "- Score 1: the answer provides some relevance to the question and answer "
            "one aspect of the question correctly. "
            "- Score 2: the answer mostly answer the question but is missing or hallucinating "
            "on one critical aspect. "
            "- Score 3: the answer correctly answer the question and not missing any "
            "major aspect"
        ),
        examples=[],
        version="v1",
        model="openai:/gpt-4",
        grading_context_columns=["renamed_ground_truth"],
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ):
        with mlflow.start_run():
            eval_df = pd.DataFrame(
                {
                    "inputs": [
                        "What is MLflow?",
                        "What is Spark?",
                        "What is Python?",
                    ],
                    "ground_truth": [
                        "MLflow is an open-source platform",
                        "Apache Spark is an open-source, distributed computing system",
                        "Python is a high-level programming language",
                    ],
                    "prediction": [
                        "MLflow is an open-source platform",
                        "Apache Spark is an open-source, distributed computing system",
                        "Python is a high-level programming language",
                    ],
                }
            )
            results = mlflow.evaluate(
                data=eval_df,
                evaluators="default",
                targets="renamed_ground_truth",
                predictions="prediction",
                extra_metrics=[metric],
                evaluator_config={"col_mapping": {"renamed_ground_truth": "ground_truth"}},
            )

            assert results.metrics == {
                "correctness/v1/mean": 3.0,
                "correctness/v1/variance": 0.0,
                "correctness/v1/p90": 3.0,
            }


def test_evaluate_custom_metric_with_string_type():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        from mlflow.metrics import make_metric

        def word_count_eval(predictions):
            scores = []
            avg = 0
            aggregate_results = {}
            for prediction in predictions:
                scores.append(prediction)
                avg += len(prediction.split(" "))

            avg /= len(predictions)
            aggregate_results["avg_len"] = avg

            return MetricValue(
                scores=scores,
                aggregate_results=aggregate_results,
            )

        word_count = make_metric(eval_fn=word_count_eval, greater_is_better=True, name="word_count")

        results = mlflow.evaluate(model_info.model_uri, data, extra_metrics=[word_count])
        assert results.metrics["word_count/avg_len"] == 3.0
        assert len(results.tables) == 1
        assert results.tables["eval_results_table"].columns.tolist() == [
            "text",
            "outputs",
            "word_count/score",
        ]
        pd.testing.assert_series_equal(
            results.tables["eval_results_table"]["word_count/score"],
            data["text"],
            check_names=False,
        )
