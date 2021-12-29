import numpy as np
import json
import math
import sklearn.metrics

from mlflow.models.evaluation import evaluate, EvaluationDataset
from mlflow.models.evaluation.default_evaluator import (
    _get_regressor_metrics,
    _get_classifier_global_metrics,
    _get_classifier_per_class_metrics,
    _extract_raw_model_and_predict_fn,
)
import mlflow
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from tests.models.test_evaluation import (
    get_run_data,
    linear_regressor_model_uri,
    diabetes_dataset,
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    binary_logistic_regressor_model_uri,
    breast_cancer_dataset,
    spark_linear_regressor_model_uri,
    diabetes_spark_dataset,
    svm_model_uri,
    breast_cancer_dataset,
)


def test_regressor_evaluation(linear_regressor_model_uri, diabetes_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            linear_regressor_model_uri,
            model_type="regressor",
            dataset=diabetes_dataset,
            evaluators="default",
        )
        print(f"regressor evaluation run: {run.info.run_id}")

    params, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(linear_regressor_model_uri)

    y = diabetes_dataset.labels
    y_pred = model.predict(diabetes_dataset.data)

    expected_metrics = _get_regressor_metrics(y, y_pred)
    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_diabetes_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_beeswarm_plot_on_data_diabetes_dataset.png",
        "shap_feature_importance_plot_on_data_diabetes_dataset.png",
        "shap_summary_plot_on_data_diabetes_dataset.png",
    }
    assert result.artifacts.keys() == {
        "shap_beeswarm_plot",
        "shap_feature_importance_plot",
        "shap_summary_plot",
    }


def test_multi_classifier_evaluation(multiclass_logistic_regressor_model_uri, iris_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            multiclass_logistic_regressor_model_uri,
            model_type="classifier",
            dataset=iris_dataset,
            evaluators="default",
        )
        print(f"multi-classifier evaluation run: {run.info.run_id}")

    params, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(multiclass_logistic_regressor_model_uri)

    _, raw_model, predict_fn, predict_proba_fn = _extract_raw_model_and_predict_fn(model)
    y = iris_dataset.labels
    y_pred = predict_fn(iris_dataset.data)
    y_probs = predict_proba_fn(iris_dataset.data)

    expected_metrics = _get_classifier_global_metrics(False, y, y_pred, y_probs)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key], metrics[metric_key + "_on_data_iris_dataset"], rtol=1e-3
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**iris_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_beeswarm_plot_on_data_iris_dataset.png",
        "per_class_metrics_data_on_data_iris_dataset.csv",
        "roc_curve_plot_on_data_iris_dataset.png",
        "precision_recall_curve_plot_on_data_iris_dataset.png",
        "shap_feature_importance_plot_on_data_iris_dataset.png",
        "explainer_on_data_iris_dataset",
        "per_class_roc_curve_data_on_data_iris_dataset.csv",
        "confusion_matrix_on_data_iris_dataset.png",
        "shap_summary_plot_on_data_iris_dataset.png",
        "per_class_precision_recall_curve_data_on_data_iris_dataset.csv",
    }
    assert result.artifacts.keys() == {
        "per_class_metrics_data",
        "per_class_roc_curve_data",
        "per_class_precision_recall_curve_data",
        "roc_curve_plot",
        "precision_recall_curve_plot",
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_bin_classifier_evaluation(binary_logistic_regressor_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            binary_logistic_regressor_model_uri,
            model_type="classifier",
            dataset=breast_cancer_dataset,
            evaluators="default",
        )
        print(f"bin-classifier evaluation run: {run.info.run_id}")

    params, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    _, raw_model, predict_fn, predict_proba_fn = _extract_raw_model_and_predict_fn(model)
    y = breast_cancer_dataset.labels
    y_pred = predict_fn(breast_cancer_dataset.data)
    y_probs = predict_proba_fn(breast_cancer_dataset.data)

    expected_metrics = _get_classifier_global_metrics(True, y, y_pred, y_probs)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_breast_cancer_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**breast_cancer_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_feature_importance_plot_on_data_breast_cancer_dataset.png",
        "lift_curve_plot_on_data_breast_cancer_dataset.png",
        "shap_beeswarm_plot_on_data_breast_cancer_dataset.png",
        "precision_recall_curve_plot_on_data_breast_cancer_dataset.png",
        "roc_curve_data_on_data_breast_cancer_dataset.csv",
        "precision_recall_curve_data_on_data_breast_cancer_dataset.csv",
        "confusion_matrix_on_data_breast_cancer_dataset.png",
        "shap_summary_plot_on_data_breast_cancer_dataset.png",
        "roc_curve_plot_on_data_breast_cancer_dataset.png",
    }
    assert result.artifacts.keys() == {
        "roc_curve_data",
        "roc_curve_plot",
        "precision_recall_curve_data",
        "precision_recall_curve_plot",
        "lift_curve_plot",
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_spark_regressor_model_evaluation(spark_linear_regressor_model_uri, diabetes_spark_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            spark_linear_regressor_model_uri,
            model_type="regressor",
            dataset=diabetes_spark_dataset,
            evaluators="default",
            evaluator_config={"log_model_explainability": True},
        )
        print(f"spark model evaluation run: {run.info.run_id}")

    params, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    X, y = diabetes_spark_dataset._extract_features_and_labels()
    y_pred = model.predict(X)

    expected_metrics = _get_regressor_metrics(y, y_pred)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_diabetes_spark_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_spark_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == set()
    assert result.artifacts == {}


def test_svm_classifier_evaluation(svm_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            svm_model_uri,
            model_type="classifier",
            dataset=breast_cancer_dataset,
            evaluators="default",
        )
        print(f"svm evaluation run: {run.info.run_id}")

    params, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(svm_model_uri)

    _, raw_model, predict_fn, predict_proba_fn = _extract_raw_model_and_predict_fn(model)
    y = breast_cancer_dataset.labels
    y_pred = predict_fn(breast_cancer_dataset.data)

    expected_metrics = _get_classifier_global_metrics(True, y, y_pred, None)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_breast_cancer_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**breast_cancer_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "confusion_matrix_on_data_breast_cancer_dataset.png",
        "shap_feature_importance_plot_on_data_breast_cancer_dataset.png",
        "shap_beeswarm_plot_on_data_breast_cancer_dataset.png",
        "shap_summary_plot_on_data_breast_cancer_dataset.png",
    }
    assert result.artifacts.keys() == {
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }
