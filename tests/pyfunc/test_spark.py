import os
import sys

import numpy as np
import pandas as pd
import pytest
import pyspark
from pyspark.sql.types import ArrayType, DoubleType, LongType, StringType, FloatType, IntegerType
from pyspark.sql.utils import AnalysisException

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.pyfunc import spark_udf, PythonModel, PyFuncModel
from mlflow.pyfunc.spark_model_cache import SparkModelCache

import tests
from mlflow.types import Schema, ColSpec

prediction = [int(1), int(2), "class1", float(0.1), 0.2]
types = [np.int32, np.int, np.str, np.float32, np.double]


def score_model_as_udf(model_uri, pandas_df, result_type="double"):
    spark = get_spark_session(pyspark.SparkConf())
    spark_df = spark.createDataFrame(pandas_df)
    pyfunc_udf = spark_udf(spark=spark, model_uri=model_uri, result_type=result_type)
    new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))
    return [x["prediction"] for x in new_df.collect()]


class ConstantPyfuncWrapper:
    @staticmethod
    def predict(model_input):
        m, _ = model_input.shape
        prediction_df = pd.DataFrame(
            data={
                str(i): np.array([prediction[i] for j in range(m)], dtype=types[i])
                for i in range(len(prediction))
            },
            columns=[str(i) for i in range(len(prediction))],
        )
        return prediction_df


def _load_pyfunc(_):
    return ConstantPyfuncWrapper()


@pytest.fixture(autouse=True)
def configure_environment():
    os.environ["PYSPARK_PYTHON"] = sys.executable


def get_spark_session(conf):
    conf.set(key="spark_session.python.worker.reuse", value=True)
    return (
        pyspark.sql.SparkSession.builder.config(conf=conf)
        .master("local-cluster[2, 1, 1024]")
        .getOrCreate()
    )


@pytest.fixture(scope="session")
def spark():
    conf = pyspark.SparkConf()
    return get_spark_session(conf)


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.mark.large
def test_spark_udf(spark, model_path):
    mlflow.pyfunc.save_model(
        path=model_path,
        loader_module=__name__,
        code_path=[os.path.dirname(tests.__file__)],
    )
    reloaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_path)

    pandas_df = pd.DataFrame(data=np.ones((10, 10)), columns=[str(i) for i in range(10)])
    spark_df = spark.createDataFrame(pandas_df)

    # Test all supported return types
    type_map = {
        "float": (FloatType(), np.number),
        "int": (IntegerType(), np.int32),
        "double": (DoubleType(), np.number),
        "long": (LongType(), np.int),
        "string": (StringType(), None),
    }

    for tname, tdef in type_map.items():
        spark_type, np_type = tdef
        prediction_df = reloaded_pyfunc_model.predict(pandas_df)
        for is_array in [True, False]:
            t = ArrayType(spark_type) if is_array else spark_type
            if tname == "string":
                expected = prediction_df.applymap(str)
            else:
                expected = prediction_df.select_dtypes(np_type)
                if tname == "float":
                    expected = expected.astype(np.float32)

            expected = [list(row[1]) if is_array else row[1][0] for row in expected.iterrows()]
            pyfunc_udf = spark_udf(spark, model_path, result_type=t)
            new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))
            actual = list(new_df.select("prediction").toPandas()["prediction"])
            assert expected == actual
            if not is_array:
                pyfunc_udf = spark_udf(spark, model_path, result_type=tname)
                new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))
                actual = list(new_df.select("prediction").toPandas()["prediction"])
                assert expected == actual


def test_spark_udf_autofills_no_arguments(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [model_input.columns] * len(model_input)

    signature = ModelSignature(
        inputs=Schema([ColSpec("long", "a"), ColSpec("long", "b"), ColSpec("long", "c")]),
        outputs=Schema([ColSpec("integer")]),
    )

    good_data = spark.createDataFrame(
        pd.DataFrame(columns=["a", "b", "c", "d"], data={"a": [1], "b": [2], "c": [3], "d": [4]})
    )
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=signature)
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        res = good_data.withColumn("res", udf()).select("res").toPandas()
        assert res["res"][0] == ["a", "b", "c"]

        with pytest.raises(
            pyspark.sql.utils.PythonException,
            match=r"Model input is missing columns. Expected 3 input columns",
        ):
            res = good_data.withColumn("res", udf("b", "c")).select("res").toPandas()

        # this dataframe won't work because it's missing column a
        bad_data = spark.createDataFrame(
            pd.DataFrame(
                columns=["x", "b", "c", "d"], data={"x": [1], "b": [2], "c": [3], "d": [4]}
            )
        )
        with pytest.raises(AnalysisException, match=r"cannot resolve 'a' given input columns"):
            bad_data.withColumn("res", udf())

    nameless_signature = ModelSignature(
        inputs=Schema([ColSpec("long"), ColSpec("long"), ColSpec("long")]),
        outputs=Schema([ColSpec("integer")]),
    )
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=nameless_signature)
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        with pytest.raises(
            MlflowException,
            match=r"Cannot apply udf because no column names specified",
        ):
            good_data.withColumn("res", udf())

    with mlflow.start_run() as run:
        # model without signature
        mlflow.pyfunc.log_model("model", python_model=TestModel())
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        with pytest.raises(pyspark.sql.utils.PythonException, match=r".+"):
            res = good_data.withColumn("res", udf()).select("res").toPandas()


def test_spark_udf_autofills_column_names_with_schema(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [model_input.columns] * len(model_input)

    signature = ModelSignature(
        inputs=Schema([ColSpec("long", "a"), ColSpec("long", "b"), ColSpec("long", "c")]),
        outputs=Schema([ColSpec("integer")]),
    )
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=signature)
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        data = spark.createDataFrame(
            pd.DataFrame(
                columns=["a", "b", "c", "d"], data={"a": [1], "b": [2], "c": [3], "d": [4]}
            )
        )
        with pytest.raises(pyspark.sql.utils.PythonException, match=r".+"):
            res = data.withColumn("res1", udf("a", "b")).select("res1").toPandas()

        res = data.withColumn("res2", udf("a", "b", "c")).select("res2").toPandas()
        assert res["res2"][0] == ["a", "b", "c"]
        res = data.withColumn("res4", udf("a", "b", "c", "d")).select("res4").toPandas()
        assert res["res4"][0] == ["a", "b", "c"]


def test_spark_udf_with_datetime_columns(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [model_input.columns] * len(model_input)

    signature = ModelSignature(
        inputs=Schema([ColSpec("datetime", "timestamp"), ColSpec("datetime", "date")]),
        outputs=Schema([ColSpec("integer")]),
    )
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=signature)
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        data = spark.range(10).selectExpr(
            "current_timestamp() as timestamp", "current_date() as date"
        )

        res = data.withColumn("res", udf("timestamp", "date")).select("res")
        res = res.toPandas()
        assert res["res"][0] == ["timestamp", "date"]


@pytest.mark.large
def test_model_cache(spark, model_path):
    mlflow.pyfunc.save_model(
        path=model_path,
        loader_module=__name__,
        code_path=[os.path.dirname(tests.__file__)],
    )

    archive_path = SparkModelCache.add_local_model(spark, model_path)
    assert archive_path != model_path

    # Ensure we can use the model locally.
    local_model = SparkModelCache.get_or_load(archive_path)
    assert isinstance(local_model, PyFuncModel)
    assert isinstance(local_model._model_impl, ConstantPyfuncWrapper)

    # Define the model class name as a string so that each Spark executor can reference it
    # without attempting to resolve ConstantPyfuncWrapper, which is only available on the driver.
    constant_model_name = ConstantPyfuncWrapper.__name__

    # Request the model on all executors, and see how many times we got cache hits.
    def get_model(_):
        model = SparkModelCache.get_or_load(archive_path)
        assert isinstance(model, PyFuncModel)
        # NB: Can not use instanceof test as remote does not know about ConstantPyfuncWrapper class.
        assert type(model._model_impl).__name__ == constant_model_name
        return SparkModelCache._cache_hits

    # This will run 30 distinct tasks, and we expect most to reuse an already-loaded model.
    # Note that we can't necessarily expect an even split, or even that there were only
    # exactly 2 python processes launched, due to Spark and its mysterious ways, but we do
    # expect significant reuse.
    results = spark.sparkContext.parallelize(range(100), 30).map(get_model).collect()
    assert max(results) > 10
    # Running again should see no newly-loaded models.
    results2 = spark.sparkContext.parallelize(range(100), 30).map(get_model).collect()
    assert min(results2) > 0
