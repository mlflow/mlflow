from __future__ import annotations

import io
import json
from os.path import join as path_join
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageChops
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.metrics import MetricValue, make_metric
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
from mlflow.models.evaluation.evaluators.base import _extract_raw_model
from mlflow.models.evaluation.evaluators.classifier import (
    _extract_predict_fn_and_prodict_proba_fn,
    _gen_classifier_curve,
    _get_binary_classifier_metrics,
    _get_binary_sum_up_label_pred_prob,
    _get_multiclass_classifier_metrics,
    _infer_model_type_by_labels,
)

from tests.evaluate.evaluators.conftest import assert_dict_equal, assert_metrics_equal
from tests.evaluate.test_evaluation import (
    binary_logistic_regressor_model_uri,  # noqa: F401
    breast_cancer_dataset,  # noqa: F401
    get_run_data,
    iris_dataset,  # noqa: F401
    multiclass_logistic_regressor_model_uri,  # noqa: F401
    svm_model_uri,  # noqa: F401
)


def test_extract_predict_proba_fn(binary_logistic_regressor_model_uri, breast_cancer_dataset):
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
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
            binary_logistic_regressor_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

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


def test_svm_classifier_evaluation(svm_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = mlflow.evaluate(
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


def test_svm_classifier_evaluation_disable_logging_metrics_and_artifacts(
    svm_model_uri, breast_cancer_dataset
):
    with mlflow.start_run() as run:
        result = mlflow.evaluate(
            svm_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            evaluators="default",
        )

    _, logged_metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(svm_model_uri)

    predict_fn, _ = _extract_predict_fn_and_prodict_proba_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_binary_classifier_metrics(y_true=y, y_pred=y_pred, sample_weights=None)
    expected_metrics["score"] = model._model_impl.score(
        breast_cancer_dataset.features_data, breast_cancer_dataset.labels_data
    )

    assert_metrics_equal(result.metrics, expected_metrics)
    assert "mlflow.datassets" not in tags


def test_infer_model_type_by_labels():
    assert _infer_model_type_by_labels(["a", "b"]) == "classifier"
    assert _infer_model_type_by_labels([True, False]) == "classifier"
    assert _infer_model_type_by_labels([1, 2.5]) == "regressor"
    assert _infer_model_type_by_labels(pd.Series(["a", "b"], dtype="category")) == "classifier"
    assert _infer_model_type_by_labels(pd.Series([1.5, 2.5], dtype="category")) == "classifier"
    assert _infer_model_type_by_labels([1, 2, 3]) is None


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

    predict_fn, _ = _extract_predict_fn_and_prodict_proba_fn(model)
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
            result = mlflow.evaluate(
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


@pytest.mark.parametrize(
    "env_manager",
    ["virtualenv", "conda"],
)
def test_evaluation_with_env_restoration(
    multiclass_logistic_regressor_model_uri, iris_dataset, env_manager
):
    with mlflow.start_run() as run:
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
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


def _get_results_for_custom_metrics_tests(
    model_uri, dataset, *, extra_metrics=None, custom_artifacts=None
):
    with mlflow.start_run() as run:
        result = mlflow.evaluate(
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


def test_default_evaluator_for_pyfunc_model(breast_cancer_dataset):
    data = load_breast_cancer()
    raw_model = LinearSVC()
    raw_model.fit(data.data, data.target)

    mlflow_model = Model()
    mlflow.pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = mlflow.pyfunc.PyFuncModel(model_meta=mlflow_model, model_impl=raw_model)

    with mlflow.start_run() as run:
        mlflow.evaluate(
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
