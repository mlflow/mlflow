import os
import shutil
import tempfile
import time

import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, IntegerType, StringType, StructField

import mlflow
import mlflow.spark
from mlflow._spark_autologging import _SPARK_TABLE_INFO_TAG_NAME

def _get_mlflow_spark_jar_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pardir = os.path.pardir
    jar_dir = os.path.join(current_dir, pardir, pardir, "mlflow", "java", "spark", "target")
    jar_filenames = [fname for fname in os.listdir(jar_dir) if ".jar" in fname]
    print(current_dir, jar_dir, jar_filenames)
    res = os.path.abspath(os.path.join(jar_dir, jar_filenames[0]))
    print(res)
    return res

@pytest.fixture(scope="session", autouse=True)
def spark_session():
    jar_path = _get_mlflow_spark_jar_path()
    session = SparkSession.builder \
        .config("spark.jars", jar_path)\
        .master("local[*]") \
        .getOrCreate()
    #.config("spark.jars", "/Users/sid.murching/code/mlflow/mlflow/java/client/target/mlflow-client-1.4.1-SNAPSHOT.jar") \
    print("@SID created session with version %s" % session.sparkContext.version )
    yield session
    print("Stopping session...")
    session.stop()
    print("Done stopping")


@pytest.fixture(scope="session", autouse=True)
def format_to_file_path(spark_session):
    print("hiii")
    rows = [
        Row(8, "bat"),
        Row(64, "mouse"),
        Row(-27, "horse")
    ]
    schema = StructType([
        StructField("number", IntegerType()),
        StructField("word", StringType())
    ])
    rdd = spark_session.sparkContext.parallelize(rows)
    df = spark_session.createDataFrame(rdd, schema)
    format_to_file_path = {}
    tempdir = tempfile.mkdtemp()
    for format in ["csv"]:#, "parquet", "json"]:
        format_to_file_path[format] = os.path.join(tempdir, "test-data-%s" % format)

    for format, file_path in format_to_file_path.items():
        df.write.option("header", "true").format(format).save(file_path)
    print("wrote stuff")
    yield format_to_file_path
    shutil.rmtree(tempdir)


def test_autologging_of_datasources_with_different_formats(spark_session, format_to_file_path):
    mlflow.spark.autolog()
    for format, file_path in format_to_file_path.items():
        base_df = spark_session.read.format(format).option("header", "true").\
            option("inferSchema", "true").load(file_path)
        dfs = [
            base_df,
            base_df.filter("number > 0"),
            base_df.select("number"),
            base_df.limit(2),
            base_df.filter("number > 0").select("number").limit(2)]

        for df in dfs:
            with mlflow.start_run():
                run_id = mlflow.active_run().info.run_id
                df.collect()
            time.sleep(1)
            run = mlflow.get_run(run_id)
            assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
            table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
            assert file_path in table_info_tag
            assert format in table_info_tag


def test_autologging_does_not_throw_on_api_failures(spark_session, format_to_file_path):
    pass


# if __name__ == "__main__":
#     pytest.main()
