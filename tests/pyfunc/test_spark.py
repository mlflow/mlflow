from __future__ import print_function

import os
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql.types import ArrayType, DoubleType, LongType, StringType

import pytest

import mlflow
from mlflow.pyfunc import load_pyfunc, spark_udf
from mlflow.pyfunc.spark_model_cache import SparkModelCache
import mlflow.sklearn


def score_model_as_udf(model_path, run_id, pandas_df):
    spark = pyspark.sql.SparkSession.builder \
        .config(key="spark.python.worker.reuse", value=True) \
        .master("local-cluster[2, 1, 1024]") \
        .getOrCreate()
    spark_df = spark.createDataFrame(pandas_df)
    pyfunc_udf = spark_udf(spark, model_path, run_id, result_type="double")
    new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))
    return [x['prediction'] for x in new_df.collect()]


class ConstPyfunc(object):
    def __init__(self, res):
        self._res = res

    def predict(self, df):
        return pd.DataFrame(data=[self._res for _ in range(df.shape[0])])


def _load_pyfunc(path):
    with open(path, "rb") as f:
        res = pickle.load(f)
        return ConstPyfunc(res)


class TestSparkUDFs(unittest.TestCase):
    def setUp(self):
        os.environ["PYSPARK_PYTHON"] = sys.executable
        self._tmp = tempfile.mkdtemp("mlflow-spark-test", dir="/tmp")
        # NB: local-cluster mode actually sets up 2 executors, each given 1 core
        # and 1024 MB of memory. This is the best way to simulate pickling/serialization
        # behavior to ensure it will work as expected on a real cluster.
        self.spark = pyspark.sql.SparkSession.builder \
            .config(key="spark.python.worker.reuse", value=True) \
            .master("local-cluster[2, 1, 1024]") \
            .getOrCreate()
        self._prediction = [1, "class1", True, 0.1, 0.2, 0.3, 0.4]
        self._model_path = os.path.join(self._tmp, "model")
        data_path = os.path.join(self._tmp, "static_result.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(self._prediction, file=f)
        # print(os.listdir(os.path.abspath(self._model_path)))
        self._pandas_df = pd.DataFrame(np.ones((10, 10)), columns=[str(i) for i in range(10)])
        mlflow.pyfunc.save_model(self._model_path,
                                 loader_module=os.path.basename(__file__)[:-3],
                                 code_path=[__file__],
                                 data_path=data_path)

    def tearDown(self):
        shutil.rmtree(self._tmp)

    @pytest.mark.large
    def test_spark_udf(self):
        pandas_df = self._pandas_df
        spark_df = self.spark.createDataFrame(pandas_df)

        # Compare against directly running the model.
        direct_model = load_pyfunc(self._model_path)
        pandas_results = direct_model.predict(pandas_df)
        for row in pandas_results.iterrows():
            assert self._prediction == list(row[1])

        # Test all supported return types
        type_map = {"double": (DoubleType(), np.number),
                    "long": (LongType(), np.int),
                    "string": (StringType(), None)}

        for tname, tdef in type_map.items():
            spark_type, np_type = tdef
            for is_array in [True, False]:
                t = ArrayType(spark_type) if is_array else spark_type
                if tname == "string":
                    expected = pandas_results.applymap(str)
                else:
                    expected = pandas_results.select_dtypes(np_type)

                expected = [list(row[1]) if is_array else row[1][0] for row in expected.iterrows()]
                pyfunc_udf = spark_udf(self.spark, self._model_path, result_type=t)
                new_df = spark_df.withColumn("prediction", pyfunc_udf(*self._pandas_df.columns))
                actual = list(new_df.select("prediction").toPandas()['prediction'])
                assert expected == actual
                if not is_array:
                    pyfunc_udf = spark_udf(self.spark, self._model_path, result_type=tname)
                    new_df = spark_df.withColumn("prediction", pyfunc_udf(*self._pandas_df.columns))
                    actual = list(new_df.select("prediction").toPandas()['prediction'])
                    assert expected == actual

    @pytest.mark.large
    def test_model_cache(self):
        archive_path = SparkModelCache.add_local_model(self.spark, self._model_path)
        assert archive_path != self._model_path

        # Ensure we can use the model locally.
        local_model = SparkModelCache.get_or_load(archive_path)
        assert isinstance(local_model, ConstPyfunc)

        # Request the model on all executors, and see how many times we got cache hits.
        def get_model(_):
            model = SparkModelCache.get_or_load(archive_path)
            # NB: Can not use instanceof test as remote dos not know about ConstPyfunc class
            assert type(model).__name__ == "ConstPyfunc"
            return SparkModelCache._cache_hits

        # This will run 30 distinct tasks, and we expect most to reuse an already-loaded model.
        # Note that we can't necessarily expect an even split, or even that there were only
        # exactly 2 python processes launched, due to Spark and its mysterious ways, but we do
        # expect significant reuse.
        results = self.spark.sparkContext.parallelize(range(0, 100), 30).map(get_model).collect()

        # TODO(tomas): Looks like spark does not reuse python workers with python==3.x
        assert sys.version[0] == '3' or max(results) > 10
        # Running again should see no newly-loaded models.
        results2 = self.spark.sparkContext.parallelize(range(0, 100), 30).map(get_model).collect()
        assert sys.version[0] == '3' or min(results2) > 0
