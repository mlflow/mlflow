from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import mlflow
from mlflow.models.evaluation.evaluators.regressor import _get_regressor_metrics

from tests.evaluate.evaluators.conftest import assert_dict_equal, assert_metrics_equal
from tests.evaluate.test_evaluation import (
    diabetes_dataset,  # noqa: F401
    diabetes_spark_dataset,  # noqa: F401
    get_run_data,
    linear_regressor_model_uri,  # noqa: F401
    pipeline_model_uri,  # noqa: F401
    spark_linear_regressor_model_uri,  # noqa: F401
)


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
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
            linear_regressor_model_uri,
            diabetes_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_dataset._constructor_args["targets"].astype(np.int64),
            evaluators="default",
        )
        result.save(tmp_path)


def test_spark_regressor_model_evaluation(
    spark_linear_regressor_model_uri,
    diabetes_spark_dataset,
):
    with mlflow.start_run() as run:
        result = mlflow.evaluate(
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
        result = mlflow.evaluate(
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


def test_evaluate_sklearn_model_score_skip_when_not_scorable(
    linear_regressor_model_uri, diabetes_dataset
):
    with mock.patch(
        "sklearn.linear_model.LinearRegression.score",
        side_effect=RuntimeError("LinearRegression.score failed"),
    ) as mock_score:
        with mlflow.start_run():
            result = mlflow.evaluate(
                linear_regressor_model_uri,
                diabetes_dataset._constructor_args["data"],
                model_type="regressor",
                targets=diabetes_dataset._constructor_args["targets"],
                evaluators="default",
            )
        mock_score.assert_called_once()
        assert "score" not in result.metrics


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

        mlflow.evaluate(
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
        result = mlflow.evaluate(
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
