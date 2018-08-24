"""
Converts the raw CSV form to a Parquet form with just the columns we want
"""


from __future__ import print_function

import requests
import tempfile
import os
import zipfile
import pyspark
import mlflow
import click


@click.command(help="Given a CSV file (see load_raw_data), transforms it into Parquet "
                    "in an mlflow artifact called 'ratings-parquet-dir'")
@click.option("--ratings-csv")
def etl_data(ratings_csv):
    with mlflow.start_run() as mlrun:
        tmpdir = tempfile.mkdtemp()
        ratings_parquet_dir = os.path.join(tmpdir, 'ratings-parquet')

        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        print("Converting ratings CSV %s to Parquet %s" % (ratings_csv, ratings_parquet_dir))
        ratings_df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(ratings_csv) \
            .drop("timestamp")  # Drop unused column
        ratings_df.show()
        ratings_df.write.parquet(ratings_parquet_dir)

        print("Uploading Parquet ratings: %s" % ratings_parquet_dir)
        mlflow.log_artifacts(ratings_parquet_dir, "ratings-parquet-dir")


if __name__ == '__main__':
    etl_data()
