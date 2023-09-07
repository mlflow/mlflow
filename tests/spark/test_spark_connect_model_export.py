import pytest
import pandas as pd
from sklearn import datasets
from pyspark.sql import SparkSession
from pyspark.ml.connect.classification import (
    LogisticRegression as LORV2,
    LogisticRegressionModel as LORV2Model,
)
from pyspark.ml.connect.feature import StandardScaler
from pyspark.ml.connect.pipeline import Pipeline
import os

from mlflow import pyfunc
from mlflow import spark as sparkm
from tests.pyfunc.test_spark import score_model_as_udf
from tests.spark.test_spark_model_export import (
    SparkModelWithData,
)


def _get_spark_connect_session():
    return (
        SparkSession.builder.remote("local[2]")
            .config("spark.connect.copyFromLocalToFs.allowDestLocal", "true")
            .getOrCreate()
    )


# Specify `autouse=True` to ensure that a context is created
# before any tests are executed. This ensures that the Hadoop filesystem
# does not create its own SparkContext without the MLeap libraries required by
# other tests.
@pytest.fixture(scope="module", autouse=True)
def spark():
    spark = _get_spark_connect_session()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def iris_df(spark):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    spark_df = spark.createDataFrame([
        (features, label)
        for features, label in zip(X, y)
    ], schema="features: array<double>, label: long")

    pandas_df = spark_df.toPandas()
    return pandas_df, spark_df


@pytest.fixture(scope="module")
def spark_model_iris(iris_df):
    iris_pandas_df, iris_spark_df = iris_df
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    lr = LORV2(maxIter=50, numTrainWorkers=2, learningRate=0.001)
    pipeline = Pipeline(stages=[scaler, lr])
    # Fit the model
    model = pipeline.fit(iris_spark_df)
    preds_pandas_df = model.transform(iris_spark_df.toPandas())
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
    sparkm.save_model(spark_model_iris.model, path=model_path)
    # 1. score and compare reloaded sparkml model
    reloaded_model = sparkm.load_model(model_uri=model_path)
    preds_df = reloaded_model.transform(spark_model_iris.pandas_df)
    pd.testing.assert_frame_equal(spark_model_iris.predictions, preds_df, check_dtype=False)

    m = pyfunc.load_model(model_path)
    # 2. score and compare reloaded pyfunc
    preds2_df = m.predict(spark_model_iris.pandas_df)
    pd.testing.assert_frame_equal(spark_model_iris.predictions, preds2_df, check_dtype=False)

    # 3. score and compare reloaded pyfunc Spark udf
    preds3_df = score_model_as_udf(model_uri=model_path, pandas_df=spark_model_iris.pandas_df)
    pd.testing.assert_frame_equal(spark_model_iris.predictions, preds3_df, check_dtype=False)
