import os
import time

import pytest
from pyspark.sql import Row
from pyspark.sql.types import IntegerType, StructField, StructType

import mlflow
import mlflow.spark

from tests.spark.autologging.utils import (
    _assert_spark_data_logged,
    _assert_spark_data_not_logged,
    _get_or_create_spark_session,
)


@pytest.mark.parametrize("disable", [False, True])
def test_enabling_autologging_before_spark_session_works(disable, tmp_path):
    mlflow.spark.autolog(disable=disable)

    # creating spark session AFTER autolog was enabled
    with _get_or_create_spark_session() as spark_session:
        rows = [Row(100)]
        schema = StructType([StructField("number2", IntegerType())])
        rdd = spark_session.sparkContext.parallelize(rows)
        df = spark_session.createDataFrame(rdd, schema)
        filepath = os.path.join(tmp_path, "test-data")
        df.write.option("header", "true").format("csv").save(filepath)

        read_df = (
            spark_session.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(filepath)
        )

        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            read_df.collect()
            time.sleep(1)

        run = mlflow.get_run(run_id)
        if disable:
            _assert_spark_data_not_logged(run=run)
        else:
            _assert_spark_data_logged(run=run, path=filepath, data_format="csv")
