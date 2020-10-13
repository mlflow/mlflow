import pytest

import mlflow
import mlflow.spark
import tempfile
import os
import shutil

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, IntegerType, StringType, StructField

@pytest.mark.large
def test_enabling_autologging_before_spark_session_works(spark_test_scoped_session):
    print(spark_test_scoped_session)
    print("enabling autolog")
    mlflow.spark.autolog()

    rows = [Row(100)]
    schema = StructType(
        [
            StructField("number2", IntegerType()),
        ]
    )
    rdd = spark_test_scoped_session.sparkContext.parallelize(rows)
    df = spark_test_scoped_session.createDataFrame(rdd, schema)
    tempdir = tempfile.mkdtemp()
    filepath = os.path.join(tempdir, "test-data")
    df.write.option("header", "true").format("csv").save(filepath)

    read_df = (
        spark_test_scoped_session.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(filepath)
    )
    
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        read_df.collect()
        time.sleep(1)

    run = mlflow.get_run(run_id)
    _assert_spark_data_logged(run=run, path=filepath, data_format="csv")

    shutil.rmtree(tempdir)