import argparse
import os
import shutil


from pyspark.sql import SparkSession
import mlflow


import mlflow.spark


def parse_args():
    parser = argparse.ArgumentParser(description="Spark example")
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # enable auto logging
    # mlflow.autolog()
    mlflow.spark.autolog()

    # prepare train and test data
    spark = (SparkSession.builder
                .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0")
                .master("local[4]")
                .getOrCreate())
    df = spark.createDataFrame([
            (4, "spark i j k"),
            (5, "l m n"),
            (6, "spark hadoop spark"),
            (7, "apache hadoop")], ["id", "text"])
    import tempfile
    tempdir = tempfile.mkdtemp()
    df.write.csv(os.path.join(tempdir, "my-data-path"), header=True)
    loaded_df = spark.read.csv(os.path.join(tempdir, "my-data-path"),
                    header=True, inferSchema=True)
    # Call toPandas() to trigger a read of the Spark datasource. Datasource info
    # (path and format) is logged to the current active run, or the
    # next-created MLflow run if no run is currently active
    with mlflow.start_run() as active_run:
        pandas_df = loaded_df.toPandas()


if __name__ == "__main__":
    main()
