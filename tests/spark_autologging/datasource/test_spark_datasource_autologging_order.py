import pytest

import mlflux
import mlflux.spark
import tempfile
import os
import shutil
import time

from pyspark.sql import Row
from pyspark.sql.types import StructType, IntegerType, StructField

from tests.spark_autologging.utils import _get_or_create_spark_session
from tests.spark_autologging.utils import (
    _assert_spark_data_logged,
    _assert_spark_data_not_logged,
)


@pytest.mark.large
@pytest.mark.parametrize("disable", [False, True])
def test_enabling_autologging_before_spark_session_works(disable):
    mlflux.spark.autolog(disable=disable)

    # creating spark session AFTER autolog was enabled
    spark_session = _get_or_create_spark_session()

    rows = [Row(100)]
    schema = StructType([StructField("number2", IntegerType())])
    rdd = spark_session.sparkContext.parallelize(rows)
    df = spark_session.createDataFrame(rdd, schema)
    tempdir = tempfile.mkdtemp()
    filepath = os.path.join(tempdir, "test-data")
    df.write.option("header", "true").format("csv").save(filepath)

    read_df = (
        spark_session.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(filepath)
    )

    with mlflux.start_run():
        run_id = mlflux.active_run().info.run_id
        read_df.collect()
        time.sleep(1)

    run = mlflux.get_run(run_id)
    if disable:
        _assert_spark_data_not_logged(run=run)
    else:
        _assert_spark_data_logged(run=run, path=filepath, data_format="csv")

    shutil.rmtree(tempdir)
    spark_session.stop()
