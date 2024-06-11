import numpy as np
import pandas as pd
import pyspark
import pytest
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import mlflow


@pytest.fixture(scope="module")
def spark():
    spark = (
        SparkSession.builder.remote("local[2]")
        .config(
            # The jars for spark-connect are not bundled in the pyspark package
            "spark.jars.packages",
            f"org.apache.spark:spark-connect_2.12:{pyspark.__version__}",
        )
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_spark_udf_spark_connect(spark):
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression().fit(X, y)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(model, "model")
    sdf = spark.createDataFrame(pd.DataFrame(X, columns=list("abcd")))
    udf = mlflow.pyfunc.spark_udf(spark, info.model_uri, env_manager="local")
    result = sdf.select(udf(*sdf.columns).alias("preds")).toPandas()
    np.testing.assert_almost_equal(result["preds"].to_numpy(), model.predict(X))


@pytest.mark.parametrize("env_manager", ["conda", "virtualenv"])
def test_spark_udf_spark_connect_unsupported_env_manager(spark, tmp_path, env_manager):
    with pytest.raises(
        mlflow.MlflowException,
        match=f"Environment manager {env_manager!r} is not supported",
    ):
        mlflow.pyfunc.spark_udf(spark, str(tmp_path), env_manager=env_manager)


def test_spark_udf_spark_connect_with_model_logging(spark, tmp_path):
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = LogisticRegression().fit(X, y)

    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run():
        signature = mlflow.models.infer_signature(X, y)
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, env_manager="local")
    X_test = X.head(5)
    sdf = spark.createDataFrame(X_test)
    preds = sdf.select(udf(*X_test.columns).alias("preds")).toPandas()["preds"]
    np.testing.assert_array_almost_equal(preds, model.predict(X_test))
