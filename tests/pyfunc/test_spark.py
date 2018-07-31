from __future__ import print_function

import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd
import pyspark
import pytest
import sklearn.datasets
from sklearn.neighbors import KNeighborsClassifier

from mlflow.pyfunc import load_pyfunc, spark_udf
from mlflow.pyfunc.spark_model_cache import SparkModelCache
import mlflow.sklearn


class TestSparkUDFs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp("mlflow-spark-test", dir="/tmp")
        # NB: local-cluster mode actually sets up 2 executors, each given 1 core
        # and 1024 MB of memory. This is the best way to simulate pickling/serialization
        # behavior to ensure it will work as expected on a real cluster.
        self.spark = pyspark.sql.SparkSession.builder \
            .config(key="spark.python.worker.reuse", value=True) \
            .master("local-cluster[2, 1, 1024]") \
            .getOrCreate()
        wine = sklearn.datasets.load_wine()
        self._pandas_df = pd.DataFrame(wine.data[:, :11], columns=wine.feature_names[:11])

        knn = KNeighborsClassifier()
        knn.fit(self._pandas_df, wine.target)
        self._model_path = os.path.join(self._tmp, "model")
        mlflow.sklearn.save_model(knn, path=self._model_path)
        self._predict = knn.predict(self._pandas_df)

    def tearDown(self):
        shutil.rmtree(self._tmp)

    @pytest.mark.large
    def test_spark_udf(self):
        pandas_df = self._pandas_df
        spark_df = self.spark.createDataFrame(pandas_df)
        pyfunc_udf = spark_udf(self.spark, self._model_path, result_type="integer")
        new_df = spark_df.withColumn("prediction", pyfunc_udf(*self._pandas_df.columns))
        spark_results = new_df.collect()

        # Compare against directly running the model.
        direct_model = load_pyfunc(self._model_path)
        pandas_results = direct_model.predict(pandas_df)
        self.assertEqual(178, len(pandas_results))
        self.assertEqual(178, len(spark_results))
        for i in range(0, len(pandas_results)):  # noqa
            self.assertEqual(self._predict[i], pandas_results[i])
            self.assertEqual(pandas_results[i], spark_results[i]['prediction'])

    @pytest.mark.large
    def test_model_cache(self):
        archive_path = SparkModelCache.add_local_model(self.spark, self._model_path)
        assert archive_path != self._model_path

        # Ensure we can use the model locally.
        local_model = SparkModelCache.get_or_load(archive_path)
        assert isinstance(local_model, KNeighborsClassifier)

        # Request the model on all executors, and see how many times we got cache hits.
        def get_model(_):
            model = SparkModelCache.get_or_load(archive_path)
            assert isinstance(model, KNeighborsClassifier)
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
