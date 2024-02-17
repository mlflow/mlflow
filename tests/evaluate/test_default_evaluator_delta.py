import tempfile
from typing import List

import pandas as pd
import pytest
from pyspark.sql import SparkSession

import mlflow
from mlflow.exceptions import MlflowException


def language_model(inputs: List[str]) -> List[str]:
    return inputs


def test_write_to_delta_fails_without_spark():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        with pytest.raises(
            MlflowException,
            match="eval_results_path is only supported in Spark environment",
        ):
            mlflow.evaluate(
                model_info.model_uri,
                data,
                extra_metrics=[mlflow.metrics.latency()],
                evaluators="default",
                evaluator_config={
                    "eval_results_path": "my_path",
                    "eval_results_mode": "overwrite",
                },
            )


@pytest.fixture
def spark_session_with_delta():
    with tempfile.TemporaryDirectory() as tmpdir:
        with SparkSession.builder.master("local[*]").config(
            "spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0"
        ).config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension").config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        ).config("spark.sql.warehouse.dir", tmpdir).getOrCreate() as spark:
            yield spark, tmpdir


def test_write_to_delta_fails_with_invalid_mode(spark_session_with_delta):
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        with pytest.raises(
            MlflowException,
            match="eval_results_mode can only be 'overwrite' or 'append'",
        ):
            mlflow.evaluate(
                model_info.model_uri,
                data,
                extra_metrics=[mlflow.metrics.latency()],
                evaluators="default",
                evaluator_config={
                    "eval_results_path": "my_path",
                    "eval_results_mode": "invalid_mode",
                },
            )


def test_write_eval_table_to_delta(spark_session_with_delta):
    spark_session, tmpdir = spark_session_with_delta
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        results = mlflow.evaluate(
            model_info.model_uri,
            data,
            extra_metrics=[mlflow.metrics.latency()],
            evaluators="default",
            evaluator_config={
                "eval_results_path": "my_path",
                "eval_results_mode": "overwrite",
            },
        )

        eval_table = results.tables["eval_results_table"].sort_values("text").reset_index(drop=True)

        eval_table_from_delta = (
            spark_session.read.format("delta")
            .load(f"{tmpdir}/my_path")
            .toPandas()
            .sort_values("text")
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(eval_table_from_delta, eval_table)


def test_write_eval_table_to_delta_append(spark_session_with_delta):
    spark_session, tmpdir = spark_session_with_delta
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model", python_model=language_model, input_example=["a", "b"]
        )
        data = pd.DataFrame({"text": ["Hello world", "My name is MLflow"]})
        mlflow.evaluate(
            model_info.model_uri,
            data,
            extra_metrics=[mlflow.metrics.latency()],
            evaluators="default",
            evaluator_config={
                "eval_results_path": "my_path",
                "eval_results_mode": "overwrite",
            },
        )

        mlflow.evaluate(
            model_info.model_uri,
            data,
            extra_metrics=[mlflow.metrics.latency()],
            evaluators="default",
            evaluator_config={
                "eval_results_path": "my_path",
                "eval_results_mode": "append",
            },
        )

        eval_table_from_delta = spark_session.read.format("delta").load(f"{tmpdir}/my_path")

        assert eval_table_from_delta.count() == 4
