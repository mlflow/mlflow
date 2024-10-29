from __future__ import annotations

import io
import json
import os
import re
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
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
    MetricValue,
    flesch_kincaid_grade_level,
    make_metric,
    toxicity,
)
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.genai_metric import (
    _GENAI_CUSTOM_METRICS_FILE_NAME,
    make_genai_metric_from_prompt,
    retrieve_custom_metrics,
)
from mlflow.metrics.genai.metric_definitions import answer_similarity
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
    _CustomArtifact,
    _evaluate_custom_artifacts,
    _extract_predict_fn,
    _extract_raw_model,
    _get_aggregate_metrics_values,
)
from mlflow.models.evaluation.evaluators.classifier import (
    _extract_predict_fn_and_prodict_proba_fn,
    _gen_classifier_curve,
    _get_binary_classifier_metrics,
    _get_binary_sum_up_label_pred_prob,
    _get_multiclass_classifier_metrics,
    _infer_model_type_by_labels,
)
from mlflow.models.evaluation.evaluators.default import _extract_output_and_other_columns
from mlflow.models.evaluation.evaluators.regressor import _get_regressor_metrics
from mlflow.models.evaluation.evaluators.shap import _compute_df_mode_or_mean
from mlflow.models.evaluation.utils.metric import MetricDefinition

from tests.evaluate.test_evaluation import (
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


@pytest.fixture(autouse=True)
def suppress_dummy_evaluator():
    """
    Dummy evaluator is registered by the test plugin and used in
    test_evaluation.py, but we don't want it to be used in this test.

    This fixture suppress dummy evaluator for the duration of each test.
    """
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    dummy_evaluator = _model_evaluation_registry._registry.pop("dummy_evaluator")

    yield

    _model_evaluation_registry._registry["dummy_evaluator"] = dummy_evaluator


def assert_dict_equal(d1, d2, rtol):
    for k in d1:
        assert k in d2
        assert np.isclose(d1[k], d2[k], rtol=rtol)


def assert_metrics_equal(actual, expected):
    for metric_key in expected:
        assert np.isclose(expected[metric_key], actual[metric_key], rtol=1e-3)


@pytest.mark.parametrize("use_sample_weights", [False, True])
@pytest.mark.parametrize("evaluators", ["default", ["regressor", "shap"], None])
def test_regressor_evaluation(
    linear_regressor_model_uri,
    diabetes_dataset,
    use_sample_weights,
    evaluators,
):
    sample_weights = (
        np.random.rand(len(diabetes_dataset.labels_data)) if use_sample_weights else None
    )

    evaluator_config = {"sample_weights": sample_weights} if use_sample_weights else {}

    if isinstance(evaluators, list):
        evaluator_config = {evaluator: evaluator_config for evaluator in evaluators}

    with mlflow.start_run() as run:
        result = evaluate(
            linear_regressor_model_uri,
            diabetes_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_dataset._constructor_args["targets"],
            evaluators=evaluators,
            evaluator_config=evaluator_config,
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
        "explainer",
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
        result = evaluate(
            linear_regressor_model_uri,
            diabetes_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_dataset._constructor_args["targets"],
            evaluators="default",
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(linear_regressor_model_uri)

    y = diabetes_dataset.labels_data
    y_pred = model.predict(diabetes_dataset.features_data)

    expected_metrics = _get_regressor_metrics(y, y_pred, sample_weights=None)
    expected_metrics["score"] = model._model_impl.score(
        diabetes_dataset.features_data, diabetes_dataset.labels_data
    )

    assert_metrics_equal(result.metrics, expected_metrics)
    assert "mlflow.datassets" not in tags


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


@pytest.mark.parametrize("use_sample_weights", [True, False])
@pytest.mark.parametrize("evaluators", ["default", ["classifier", "shap"], None])
def test_multi_classifier_evaluation(
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    use_sample_weights,
    evaluators,
):
    sample_weights = np.random.rand(len(iris_dataset.labels_data)) if use_sample_weights else None
    evaluator_config = {"sample_weights": sample_weights} if use_sample_weights else {}

    with mlflow.start_run() as run:
        result = evaluate(
            multiclass_logistic_regressor_model_uri,
            iris_dataset._constructor_args["data"],
            model_type="classifier",
            targets=iris_dataset._constructor_args["targets"],
            evaluators="default",
            evaluator_config=evaluator_config,
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(multiclass_logistic_regressor_model_uri)

    predict_fn, predict_proba_fn = _extract_predict_fn_and_prodict_proba_fn(model)
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
        result = evaluate(
            multiclass_logistic_regressor_model_uri,
            iris_dataset._constructor_args["data"],
            model_type="classifier",
            targets=iris_dataset._constructor_args["targets"],
            evaluators="default",
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(multiclass_logistic_regressor_model_uri)

    predict_fn, predict_proba_fn = _extract_predict_fn_and_prodict_proba_fn(model)
    y = iris_dataset.labels_data
    y_pred = predict_fn(iris_dataset.features_data)
    y_probs = predict_proba_fn(iris_dataset.features_data)

    expected_metrics = _get_multiclass_classifier_metrics(
        y_true=y, y_pred=y_pred, y_proba=y_probs, sample_weights=None
    )
    expected_metrics["score"] = model._model_impl.score(
        iris_dataset.features_data, iris_dataset.labels_data
    )

    assert_metrics_equal(result.metrics, expected_metrics)
    assert "mlflow.datassets" not in tags


def test_bin_classifier_evaluation(
    binary_logistic_regressor_model_uri,
    breast_cancer_dataset,
):
    with mlflow.start_run() as run:
        result = evaluate(
            binary_logistic_regressor_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
            evaluator_config={"sample_weights": None},
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    predict_fn, predict_proba_fn = _extract_predict_fn_and_prodict_proba_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)
    y_probs = predict_proba_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(y_true=y, y_pred=y_pred, y_proba=y_probs)
    expected_metrics["score"] = model._model_impl.score(
        breast_cancer_dataset.features_data,
        breast_cancer_dataset.labels_data,
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
        result = evaluate(
            binary_logistic_regressor_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, predict_proba_fn = _extract_predict_fn_and_prodict_proba_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)
    y_probs = predict_proba_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(
        y_true=y, y_pred=y_pred, y_proba=y_probs, sample_weights=None
    )
    expected_metrics["score"] = model._model_impl.score(
        breast_cancer_dataset.features_data, breast_cancer_dataset.labels_data
    )

    assert_metrics_equal(result.metrics, expected_metrics)
    assert "mlflow.datassets" not in tags


def test_spark_regressor_model_evaluation(
    spark_linear_regressor_model_uri,
    diabetes_spark_dataset,
):
    with mlflow.start_run() as run:
        result = evaluate(
            spark_linear_regressor_model_uri,
            diabetes_spark_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_spark_dataset._constructor_args["targets"],
            evaluators="default",
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
        result = evaluate(
            spark_linear_regressor_model_uri,
            diabetes_spark_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_spark_dataset._constructor_args["targets"],
            evaluators="default",
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    X = diabetes_spark_dataset.features_data
    y = diabetes_spark_dataset.labels_data
    y_pred = model.predict(X)

    expected_metrics = _get_regressor_metrics(y, y_pred, sample_weights=None)
    assert_metrics_equal(result.metrics, expected_metrics)


def test_static_spark_dataset_evaluation():
    data = load_diabetes()
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    rows = [
        (Vectors.dense(features), float(label), float(label))
        for features, label in zip(data.data, data.target)
    ]
    spark_dataframe = spark.createDataFrame(
        spark.sparkContext.parallelize(rows, 1), ["features", "label", "model_output"]
    )
    with mlflow.start_run():
        mlflow.evaluate(
            data=spark_dataframe,
            targets="label",
            predictions="model_output",
            model_type="regressor",
        )
        run_id = mlflow.active_run().info.run_id

    computed_eval_metrics = mlflow.get_run(run_id).data.metrics
    assert "mean_squared_error" in computed_eval_metrics


def test_svm_classifier_evaluation(svm_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            svm_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(svm_model_uri)

    predict_fn, _ = _extract_predict_fn_and_prodict_proba_fn(model)
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
        result = evaluate(
            svm_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(svm_model_uri)

    _, raw_model = _extract_raw_model(model)
    predict_fn, _ = _extract_predict_fn_and_prodict_proba_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(y_true=y, y_pred=y_pred, sample_weights=None)
    expected_metrics["score"] = model._model_impl.score(
        breast_cancer_dataset.features_data, breast_cancer_dataset.labels_data
    )

    assert_metrics_equal(result.metrics, expected_metrics)
    assert "mlflow.datassets" not in tags


def test_pipeline_model_kernel_explainer_on_categorical_features(pipeline_model_uri):
    from mlflow.models.evaluation._shap_patch import _PatchedKernelExplainer

    data, target_col = get_pipeline_model_dataset()
    with mlflow.start_run() as run:
        evaluate(
            pipeline_model_uri,
            data[0::3],
            model_type="classifier",
            targets=target_col,
            evaluators="default",
            evaluator_config={"explainability_algorithm": "kernel"},
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
    predict_fn, predict_proba_fn = _extract_predict_fn_and_prodict_proba_fn(model)

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
    with mock.patch("mlflow.models.evaluation.default_evaluator._logger.warning") as mock_warning:
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

    predict_fn = _extract_predict_fn(model)
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

    eval_metric = make_metric(eval_fn=metric, greater_is_better=True)
    assert eval_metric.name == "metric"

    eval_metric = make_metric(eval_fn=metric, greater_is_better=True, name="my_metric")
    assert eval_metric.name == "my_metric"

    eval_metric = make_metric(
        eval_fn=lambda _df, _metrics: 0, greater_is_better=True, name="metric"
    )
    assert eval_metric.name == "metric"

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
            "model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"question": ["a", "b"], "answer": ["a", "b"]})

    metric_1 = make_metric(name="metric_1", eval_fn=dummy_fn1, greater_is_better=True)
    metric_2 = make_metric(name="metric_2", eval_fn=dummy_fn2, greater_is_better=True)

    error_message = (
        r"Error: Metric calculation failed for the following metrics:\n"
        r"Metric 'metric_1' requires the following:\n"
        r"- the 'targets' parameter needs to be specified\n"
        r"- missing columns \['param_1', 'param_2'\] need to be defined or mapped\n"
        r"Metric 'metric_2' requires the following:\n"
        r"- missing columns \['param_3', 'builtin_metrics'\] need to be defined or mapped\n\n"
        r"Below are the existing column names for the input/output data:\n"
        r"Input Columns: \['question', 'answer'\]\n"
        r"Output Columns: \['predictions'\]\n\n"
    )

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                model_info.model_uri,
                data,
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model",
            python_model=question_classifier,
            input_example=pd.DataFrame({"question": ["a", "b"]}),
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model",
            python_model=another_language_model,
            input_example=pd.DataFrame({"text": ["a", "b"]}),
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model", python_model=language_model, input_example=["a", "b"]
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


def test_default_evaluator_for_pyfunc_model(breast_cancer_dataset):
    data = load_breast_cancer()
    raw_model = LinearSVC()
    raw_model.fit(data.data, data.target)

    mlflow_model = Model()
    mlflow.pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = mlflow.pyfunc.PyFuncModel(model_meta=mlflow_model, model_impl=raw_model)

    with mlflow.start_run() as run:
        evaluate(
            pyfunc_model,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
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
            "model",
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


def test_evaluate_no_model_or_predictions_specified():
    data = pd.DataFrame(
        {
            "question": ["words random", "This is a sentence."],
            "truth": ["words random", "This is a sentence."],
        }
    )

    with pytest.raises(
        MlflowException,
        match=(
            "Either a model or set of predictions must be specified in order to use the"
            " default evaluator"
        ),
    ):
        mlflow.evaluate(
            data=data,
            targets="truth",
            model_type="regressor",
            evaluators="default",
        )


def test_evaluate_no_model_and_predictions_specified_with_unsupported_data_type():
    X = np.random.random((5, 5))
    y = np.random.random(5)

    with pytest.raises(
        MlflowException,
        match="If predictions is specified, data must be one of the following types",
    ):
        mlflow.evaluate(
            data=X,
            targets=y,
            predictions="model_output",
            model_type="regressor",
            evaluators="default",
        )


def test_evaluate_no_model_type():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model", python_model=language_model, input_example=["a", "b"]
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


def test_default_metrics_as_extra_metrics():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=multi_output_model, input_example=["a"]
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
            extra_metrics=[
                mlflow.metrics.exact_match(),
            ],
            evaluators="default",
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "eval_results_table.json" in artifacts
    assert "exact_match/v1" in results.metrics.keys()


def test_default_metrics_as_extra_metrics_static_dataset():
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
            extra_metrics=[
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


def test_derived_metrics_basic_dependency_graph():
    def metric_1(predictions, targets, metrics):
        return MetricValue(
            scores=[0, 1],
            justifications=["first justification", "second justification"],
            aggregate_results={"aggregate": 0.5},
        )

    def metric_2(predictions, targets, metrics, metric_1):
        return MetricValue(
            scores=[score * 5 for score in metric_1.scores],
            justifications=[
                "metric_2: " + justification for justification in metric_1.justifications
            ],
            aggregate_results={
                **metric_1.aggregate_results,
                **metrics["toxicity/v1"].aggregate_results,
            },
        )

    def metric_3(predictions, targets, metric_1, metric_2):
        return MetricValue(
            scores=[score - 1 for score in metric_2.scores],
            justifications=metric_1.justifications,
            aggregate_results=metric_2.aggregate_results,
        )

    with mlflow.start_run():
        data = pd.DataFrame(
            {
                "question": ["words random", "This is a sentence."],
                "truth": ["words random", "This is a sentence."],
                "answer": ["words random", "This is a sentence."],
            }
        )
        results = evaluate(
            data=data,
            targets="truth",
            predictions="answer",
            model_type="text",
            extra_metrics=[
                make_metric(eval_fn=metric_1, greater_is_better=True, version="v1"),
                make_metric(eval_fn=metric_2, greater_is_better=True, version="v2"),
                make_metric(eval_fn=metric_3, greater_is_better=True),
            ],
            evaluators="default",
        )

    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "question",
        "truth",
        "answer",
        "token_count",
        "toxicity/v1/score",
        "flesch_kincaid_grade_level/v1/score",
        "ari_grade_level/v1/score",
        "metric_1/v1/score",
        "metric_2/v2/score",
        "metric_3/score",
        "metric_1/v1/justification",
        "metric_2/v2/justification",
        "metric_3/justification",
    }

    assert logged_data["metric_1/v1/score"].tolist() == [0, 1]
    assert logged_data["metric_2/v2/score"].tolist() == [0, 5]
    assert logged_data["metric_3/score"].tolist() == [-1, 4]
    assert logged_data["metric_1/v1/justification"].tolist() == [
        "first justification",
        "second justification",
    ]
    assert logged_data["metric_2/v2/justification"].tolist() == [
        "metric_2: first justification",
        "metric_2: second justification",
    ]
    assert logged_data["metric_3/justification"].tolist() == [
        "first justification",
        "second justification",
    ]

    assert results.metrics["metric_1/v1/aggregate"] == 0.5
    assert results.metrics["metric_2/v2/aggregate"] == 0.5
    assert results.metrics["metric_3/aggregate"] == 0.5
    assert "metric_2/v2/mean" in results.metrics.keys()
    assert "metric_2/v2/variance" in results.metrics.keys()
    assert "metric_2/v2/p90" in results.metrics.keys()
    assert "metric_3/mean" in results.metrics.keys()
    assert "metric_3/variance" in results.metrics.keys()
    assert "metric_3/p90" in results.metrics.keys()


def test_derived_metrics_complicated_dependency_graph():
    def metric_1(predictions, targets, metric_2, metric_3, metric_6):
        assert metric_2.scores == [2, 3]
        assert metric_3.scores == [3, 4]
        assert metric_6.scores == [6, 7]
        return MetricValue(scores=[1, 2])

    def metric_2(predictions, targets):
        return MetricValue(scores=[2, 3])

    def metric_3(predictions, targets, metric_4, metric_5):
        assert metric_4.scores == [4, 5]
        assert metric_5.scores == [5, 6]
        return MetricValue(scores=[3, 4])

    def metric_4(predictions, targets, metric_6):
        assert metric_6.scores == [6, 7]
        return MetricValue(scores=[4, 5])

    def metric_5(predictions, targets, metric_4, metric_6):
        assert metric_4.scores == [4, 5]
        assert metric_6.scores == [6, 7]
        return MetricValue(scores=[5, 6])

    def metric_6(predictions, targets, metric_2):
        assert metric_2.scores == [2, 3]
        return MetricValue(scores=[6, 7])

    data = pd.DataFrame(
        {
            "question": ["words random", "This is a sentence."],
            "truth": ["words random", "This is a sentence."],
            "answer": ["words random", "This is a sentence."],
        }
    )

    with mlflow.start_run():
        results = evaluate(
            data=data,
            predictions="answer",
            targets="truth",
            extra_metrics=[
                make_metric(eval_fn=metric_1, greater_is_better=True, version="v1"),
                make_metric(eval_fn=metric_2, greater_is_better=True, version="v2"),
                make_metric(eval_fn=metric_3, greater_is_better=True),
                make_metric(eval_fn=metric_4, greater_is_better=True),
                make_metric(eval_fn=metric_5, greater_is_better=True, version="v1"),
                make_metric(eval_fn=metric_6, greater_is_better=True, version="v3"),
            ],
            evaluators="default",
        )

    logged_data = pd.DataFrame(**results.artifacts["eval_results_table"].content)
    assert set(logged_data.columns.tolist()) == {
        "question",
        "truth",
        "answer",
        "metric_1/v1/score",
        "metric_2/v2/score",
        "metric_3/score",
        "metric_4/score",
        "metric_5/v1/score",
        "metric_6/v3/score",
    }

    assert logged_data["metric_1/v1/score"].tolist() == [1, 2]
    assert logged_data["metric_2/v2/score"].tolist() == [2, 3]
    assert logged_data["metric_3/score"].tolist() == [3, 4]
    assert logged_data["metric_4/score"].tolist() == [4, 5]
    assert logged_data["metric_5/v1/score"].tolist() == [5, 6]
    assert logged_data["metric_6/v3/score"].tolist() == [6, 7]

    def metric_7(predictions, targets, metric_8, metric_9):
        return MetricValue(scores=[7, 8])

    def metric_8(predictions, targets, metric_11):
        return MetricValue(scores=[8, 9])

    def metric_9(predictions, targets):
        return MetricValue(scores=[9, 10])

    def metric_10(predictions, targets, metric_9):
        return MetricValue(scores=[10, 11])

    def metric_11(predictions, targets, metric_7, metric_10):
        return MetricValue(scores=[11, 12])

    error_message = r"Error: Metric calculation failed for the following metrics:\n"

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                data=data,
                predictions="answer",
                targets="truth",
                model_type="text",
                extra_metrics=[
                    make_metric(eval_fn=metric_7, greater_is_better=True),
                    make_metric(eval_fn=metric_8, greater_is_better=True),
                    make_metric(eval_fn=metric_9, greater_is_better=True),
                    make_metric(eval_fn=metric_10, greater_is_better=True),
                    make_metric(eval_fn=metric_11, greater_is_better=True),
                ],
                evaluators="default",
            )


def test_derived_metrics_circular_dependencies_raises_exception():
    def metric_1(predictions, targets, metric_2):
        return 0

    def metric_2(predictions, targets, metric_3):
        return 0

    def metric_3(predictions, targets, metric_1):
        return 0

    error_message = r"Error: Metric calculation failed for the following metrics:\n"

    data = pd.DataFrame(
        {
            "question": ["words random", "This is a sentence."],
            "answer": ["words random", "This is a sentence."],
        }
    )

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                data=data,
                predictions="answer",
                model_type="text",
                extra_metrics=[
                    make_metric(eval_fn=metric_1, greater_is_better=True),
                    make_metric(eval_fn=metric_2, greater_is_better=True),
                    make_metric(eval_fn=metric_3, greater_is_better=True),
                ],
                evaluators="default",
            )


def test_derived_metrics_missing_dependencies_raises_exception():
    def metric_1(predictions, targets, metric_2):
        return 0

    def metric_2(predictions, targets, metric_3):
        return 0

    error_message = r"Error: Metric calculation failed for the following metrics:\n"

    data = pd.DataFrame(
        {
            "question": ["words random", "This is a sentence."],
            "answer": ["words random", "This is a sentence."],
        }
    )

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        with mlflow.start_run():
            mlflow.evaluate(
                data=data,
                predictions="answer",
                model_type="text",
                extra_metrics=[
                    make_metric(eval_fn=metric_1, greater_is_better=True),
                    make_metric(eval_fn=metric_2, greater_is_better=True),
                ],
                evaluators="default",
            )


def test_custom_metric_bad_names():
    def metric_fn(predictions, targets):
        return 0

    error_message = re.escape(
        "Invalid metric name 'metric/with/slash'. Metric names cannot include "
        "forward slashes ('/')."
    )
    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        make_metric(eval_fn=metric_fn, name="metric/with/slash", greater_is_better=True)

    with mock.patch("mlflow.models.evaluation.base._logger.warning") as mock_warning:
        make_metric(eval_fn=metric_fn, name="bad-metric-name", greater_is_better=True)
        mock_warning.assert_called_once_with(
            "The metric name 'bad-metric-name' provided is not a valid Python identifier, which "
            "will prevent its use as a base metric for derived metrics. Please use a valid "
            "identifier to enable creation of derived metrics that use the given metric."
        )

    with mock.patch("mlflow.models.evaluation.base._logger.warning") as mock_warning:
        make_metric(eval_fn=metric_fn, name="None", greater_is_better=True)
        mock_warning.assert_called_once_with(
            "The metric name 'None' is a reserved Python keyword, which will "
            "prevent its use as a base metric for derived metrics. Please use a valid identifier "
            "to enable creation of derived metrics that use the given metric."
        )

    with mock.patch("mlflow.models.evaluation.base._logger.warning") as mock_warning:
        make_metric(eval_fn=metric_fn, name="predictions", greater_is_better=True)
        mock_warning.assert_called_once_with(
            "The metric name 'predictions' is used as a special parameter in MLflow metrics, which "
            "will prevent its use as a base metric for derived metrics. Please use a different "
            "name to enable creation of derived metrics that use the given metric."
        )


def test_multi_output_model_error_handling():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=multi_output_model, input_example=["a"]
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
                extra_metrics=[
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
            "model", python_model=language_model, input_example=["a", "b"]
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
            "model", python_model=language_model, input_example=["a", "b"]
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


def test_evaluate_with_latency_and_pd_series():
    with mlflow.start_run() as run:

        def pd_series_model(inputs: list[str]) -> pd.Series:
            return pd.Series(inputs)

        model_info = mlflow.pyfunc.log_model(
            "model", python_model=pd_series_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["input text", "random text"]})
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


def test_evaluate_with_latency_static_dataset():
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=language_model, input_example=["a", "b"])
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


properly_formatted_openai_response1 = """\
{
  "score": 3,
  "justification": "justification"
}"""


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
        model="openai:/gpt-4o-mini",
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
            "model", python_model=language_model, input_example=["a", "b"]
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


def test_evaluate_retriever_with_numpy_array_values():
    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [np.array(["doc1", "doc2"])] * 3})

    def fn(X):
        return pd.DataFrame({"retrieved_context": [np.array(["doc1", "doc3", "doc2"])] * len(X)})

    with mlflow.start_run():
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
    assert results.metrics == {
        "precision_at_3/mean": 2 / 3,
        "precision_at_3/p90": 2 / 3,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_3/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_3/variance": 0.0,
    }


def test_evaluate_retriever_with_ints():
    X = pd.DataFrame({"question": ["q1?"] * 3, "ground_truth": [[1, 2]] * 3})

    def fn(X):
        return pd.DataFrame({"retrieved_context": [np.array([1, 3, 2])] * len(X)})

    with mlflow.start_run():
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
    assert results.metrics == {
        "precision_at_3/mean": 2 / 3,
        "precision_at_3/p90": 2 / 3,
        "precision_at_3/variance": 0.0,
        "recall_at_3/mean": 1.0,
        "recall_at_3/p90": 1.0,
        "recall_at_3/variance": 0.0,
        "ndcg_at_3/mean": pytest.approx(0.9197207891481877),
        "ndcg_at_3/p90": pytest.approx(0.9197207891481877),
        "ndcg_at_3/variance": 0.0,
    }


def test_evaluate_with_numpy_array():
    data = [
        ["What is MLflow?"],
    ]
    ground_truth = [
        "MLflow is an open-source platform for managing the end-to-end machine learning",
    ]

    with mlflow.start_run():
        logged_model = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
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


def test_precanned_metrics_work():
    metric = mlflow.metrics.rouge1()
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
            predictions="prediction",
            extra_metrics=[metric],
            evaluator_config={
                "col_mapping": {
                    "targets": "ground_truth",
                }
            },
        )

        assert results.metrics == {
            "rouge1/v1/mean": 1.0,
            "rouge1/v1/variance": 0.0,
            "rouge1/v1/p90": 1.0,
        }


def test_evaluate_custom_metric_with_string_type():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a", "b"]
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


def test_do_not_log_built_in_metrics_as_artifacts():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "inputs": ["words random", "This is a sentence."],
                "ground_truth": ["words random", "This is a sentence."],
            }
        )
        evaluate(
            model_info.model_uri,
            data,
            targets="ground_truth",
            predictions="answer",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                toxicity(),
                flesch_kincaid_grade_level(),
            ],
        )
        client = mlflow.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
        assert _GENAI_CUSTOM_METRICS_FILE_NAME not in artifacts

        results = retrieve_custom_metrics(run_id=run.info.run_id)
        assert len(results) == 0


def test_log_genai_custom_metrics_as_artifacts():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "inputs": ["words random", "This is a sentence."],
                "ground_truth": ["words random", "This is a sentence."],
            }
        )
        example = EvaluationExample(
            input="What is MLflow?",
            output="MLflow is an open-source platform for managing machine learning workflows.",
            score=4,
            justification="test",
            grading_context={"targets": "test"},
        )
        # This simulates the code path for metrics created from make_genai_metric
        answer_similarity_metric = answer_similarity(
            model="gateway:/gpt-4o-mini", examples=[example]
        )
        another_custom_metric = make_genai_metric_from_prompt(
            name="another custom llm judge",
            judge_prompt="This is another custom judge prompt.",
            greater_is_better=False,
            parameters={"temperature": 0.0},
        )
        result = evaluate(
            model_info.model_uri,
            data,
            targets="ground_truth",
            predictions="answer",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                answer_similarity_metric,
                another_custom_metric,
            ],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert _GENAI_CUSTOM_METRICS_FILE_NAME in artifacts

    table = result.tables[os.path.splitext(_GENAI_CUSTOM_METRICS_FILE_NAME)[0]]
    assert table.loc[0, "name"] == "answer_similarity"
    assert table.loc[0, "version"] == "v1"
    assert table.loc[1, "name"] == "another custom llm judge"
    assert table.loc[1, "version"] == ""
    assert table["version"].dtype == "object"

    results = retrieve_custom_metrics(run.info.run_id)
    assert len(results) == 2
    assert [r.name for r in results] == ["answer_similarity", "another custom llm judge"]

    results = retrieve_custom_metrics(run_id=run.info.run_id, name="another custom llm judge")
    assert len(results) == 1
    assert results[0].name == "another custom llm judge"

    results = retrieve_custom_metrics(run_id=run.info.run_id, version="v1")
    assert len(results) == 1
    assert results[0].name == "answer_similarity"

    results = retrieve_custom_metrics(
        run_id=run.info.run_id, name="answer_similarity", version="v1"
    )
    assert len(results) == 1
    assert results[0].name == "answer_similarity"

    results = retrieve_custom_metrics(run_id=run.info.run_id, name="do not match")
    assert len(results) == 0

    results = retrieve_custom_metrics(run_id=run.info.run_id, version="do not match")
    assert len(results) == 0


def test_all_genai_custom_metrics_are_from_user_prompt():
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=language_model, input_example=["a"]
        )
        data = pd.DataFrame(
            {
                "inputs": ["words random", "This is a sentence."],
                "ground_truth": ["words random", "This is a sentence."],
            }
        )
        custom_metric = make_genai_metric_from_prompt(
            name="custom llm judge",
            judge_prompt="This is a custom judge prompt.",
            greater_is_better=False,
            parameters={"temperature": 0.0},
        )
        another_custom_metric = make_genai_metric_from_prompt(
            name="another custom llm judge",
            judge_prompt="This is another custom judge prompt.",
            greater_is_better=False,
            parameters={"temperature": 0.7},
        )
        result = evaluate(
            model_info.model_uri,
            data,
            targets="ground_truth",
            predictions="answer",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                custom_metric,
                another_custom_metric,
            ],
        )

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert _GENAI_CUSTOM_METRICS_FILE_NAME in artifacts

    table = result.tables[os.path.splitext(_GENAI_CUSTOM_METRICS_FILE_NAME)[0]]
    assert table.loc[0, "name"] == "custom llm judge"
    assert table.loc[1, "name"] == "another custom llm judge"
    assert table.loc[0, "version"] == ""
    assert table.loc[1, "version"] == ""
    assert table["version"].dtype == "object"


def test_xgboost_model_evaluate_work_with_shap_explainer():
    import shap
    import xgboost
    from sklearn.model_selection import train_test_split

    mlflow.xgboost.autolog(log_input_examples=True)
    X, y = shap.datasets.adult()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    xgb_model = xgboost.XGBClassifier()
    with mlflow.start_run():
        xgb_model.fit(X_train, y_train)

        eval_data = X_test
        eval_data["label"] = y_test

        model_uri = mlflow.get_artifact_uri("model")
        with mock.patch("mlflow.models.evaluation.evaluators.shap._logger.warning") as mock_warning:
            mlflow.evaluate(
                model_uri,
                eval_data,
                targets="label",
                model_type="classifier",
                evaluators=["default"],
            )
            assert not any(
                "Shap evaluation failed." in call_arg[0]
                for call_arg in mock_warning.call_args or []
                if isinstance(call_arg, tuple)
            )


@pytest.mark.parametrize(
    "evaluator_config",
    [
        None,
        {"default": {"pos_label": 1}},
        {"default": {"label_list": [0, 1]}},
        {"default": {"label_list": [0, 1], "pos_label": 1}},
    ],
)
def test_evaluate_binary_classifier_calculate_label_list_correctly(evaluator_config):
    data = pd.DataFrame({"target": [0, 0, 1, 0], "prediction": [0, 1, 0, 0]})

    result = mlflow.evaluate(
        data=data,
        model_type="classifier",
        targets="target",
        predictions="prediction",
        evaluator_config=evaluator_config,
    )
    metrics_set = {
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "example_count",
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
    }
    assert metrics_set.issubset(result.metrics)


@pytest.mark.parametrize(
    ("evaluator_config", "data"),
    [
        (None, {"target": [1, 0, 1, 1], "prediction": [1, 2, 0, 0]}),
        (
            {"default": {"label_list": [0, 1, 2]}},
            {"target": [1, 0, 1, 1], "prediction": [1, 2, 0, 0]},
        ),
        (
            {"default": {"label_list": [0, 1, 2], "pos_label": 1}},
            {"target": [0, 0, 0, 0], "prediction": [0, 0, 0, 0]},
        ),
    ],
)
def test_evaluate_multi_classifier_calculate_label_list_correctly(
    evaluator_config, data, monkeypatch
):
    monkeypatch.setenv("_MLFLOW_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS", "true")
    result = mlflow.evaluate(
        data=pd.DataFrame(data),
        model_type="classifier",
        targets="target",
        predictions="prediction",
        evaluator_config=evaluator_config,
    )
    metrics_set = {
        "example_count",
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
    }
    assert metrics_set.issubset(result.metrics)
    assert {"true_negatives", "false_positives", "false_negatives", "true_positives"}.isdisjoint(
        result.metrics
    )


def test_evaluate_errors_invalid_pos_label():
    data = pd.DataFrame({"target": [0, 0, 1, 0], "prediction": [0, 1, 0, 0]})
    with pytest.raises(MlflowException, match=r"'pos_label' 1 must exist in 'label_list'"):
        mlflow.evaluate(
            data=data,
            model_type="classifier",
            targets="target",
            predictions="prediction",
            evaluator_config={"default": {"pos_label": 1, "label_list": [0]}},
        )
