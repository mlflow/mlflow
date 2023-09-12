import json
import os

import numpy as np
import pandas as pd
import pyspark
import pytest
from packaging.version import Version
from pyspark.ml.connect.classification import (
    LogisticRegression as LORV2,
)
from pyspark.ml.connect.feature import StandardScaler
from pyspark.ml.connect.pipeline import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as spark_f
from pyspark.sql.types import LongType
from sklearn import datasets

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.pyfunc import spark_udf

from tests.helper_functions import (
    pyfunc_serve_and_score_model,
)
from tests.pyfunc.test_spark import score_model_as_udf
from tests.spark.test_spark_model_export import (
    SparkModelWithData,
)

if Version(pyspark.__version__) < Version("3.5"):
    pytest.skip("pyspark ML connect Model is available in pyspark >= 3.5")


def _get_spark_connect_session():
    return (
        SparkSession.builder.remote("local[2]")
        .config("spark.connect.copyFromLocalToFs.allowDestLocal", "true")
        .getOrCreate()
    )


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


def score_model_as_udf(model_uri, pandas_df, result_type):
    spark = SparkSession.getActiveSession()
    spark_df = spark.createDataFrame(pandas_df).coalesce(1)
    pyfunc_udf = spark_udf(
        spark=spark, model_uri=model_uri, result_type=result_type, env_manager="local"
    )
    new_df = spark_df.withColumn("prediction", pyfunc_udf(spark_f.struct(spark_f.col("features"))))
    return new_df.toPandas()["prediction"]


@pytest.fixture(scope="module")
def spark():
    spark = _get_spark_connect_session()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def iris_df(spark):
    X, y = datasets.load_iris(return_X_y=True)
    spark_df = spark.createDataFrame(list(zip(X, y)), schema="features: array<double>, label: long")
    return spark_df.toPandas(), spark_df


@pytest.fixture(scope="module")
def spark_model_iris(iris_df):
    iris_pandas_df, iris_spark_df = iris_df
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    lr = LORV2(maxIter=50, numTrainWorkers=2, learningRate=0.001)
    pipeline = Pipeline(stages=[scaler, lr])
    # Fit the model
    model = pipeline.fit(iris_spark_df)
    preds_pandas_df = model.transform(iris_pandas_df.copy(deep=False))
    return SparkModelWithData(
        model=model,
        spark_df=iris_spark_df,
        pandas_df=iris_pandas_df,
        predictions=preds_pandas_df,
    )


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


def test_model_export(spark_model_iris, model_path):
    mlflow.spark.save_model(spark_model_iris.model, path=model_path)
    # 1. score and compare reloaded sparkml model
    reloaded_model = mlflow.spark.load_model(model_uri=model_path)
    preds_df = reloaded_model.transform(spark_model_iris.pandas_df.copy(deep=False))
    pd.testing.assert_frame_equal(spark_model_iris.predictions, preds_df, check_dtype=False)

    m = pyfunc.load_model(model_path)
    # 2. score and compare reloaded pyfunc
    preds2 = m.predict(spark_model_iris.pandas_df.copy(deep=False))
    pd.testing.assert_series_equal(
        spark_model_iris.predictions["prediction"], preds2, check_dtype=False
    )

    # 3. score and compare reloaded pyfunc Spark udf
    preds3 = score_model_as_udf(
        model_uri=model_path, pandas_df=spark_model_iris.pandas_df, result_type=LongType()
    )
    pd.testing.assert_series_equal(
        spark_model_iris.predictions["prediction"], preds3, check_dtype=False
    )


def test_sparkml_model_log(tmp_path, spark_model_iris):
    old_tracking_uri = mlflow.get_tracking_uri()

    try:
        tracking_dir = tmp_path.joinpath("mlruns")
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        with mlflow.start_run():
            model_info = mlflow.spark.log_model(
                artifact_path="model",
                spark_model=spark_model_iris.model,
            )
        model_uri = model_info.model_uri

        reloaded_model = mlflow.spark.load_model(model_uri=model_uri)
        preds_df = reloaded_model.transform(spark_model_iris.pandas_df.copy(deep=False))
        pd.testing.assert_frame_equal(spark_model_iris.predictions, preds_df, check_dtype=False)
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_pyfunc_serve_and_score(spark_model_iris):
    input_data = pd.DataFrame(
        {"features": spark_model_iris.pandas_df.features.map(lambda x: x.tolist())}
    )

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.spark.log_model(spark_model_iris.model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=input_data,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    scores = pd.DataFrame(
        data=json.loads(resp.content.decode("utf-8"))["predictions"]
    ).values.squeeze()
    np.testing.assert_array_almost_equal(
        scores, spark_model_iris.model.transform(spark_model_iris.pandas_df)["prediction"].values
    )
