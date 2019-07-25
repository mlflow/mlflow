"""
Converts the raw CSV form to a Parquet form with just the columns we want
"""

from __future__ import print_function

import tempfile
import os
import pyspark
import mlflow
import click


@click.command(help="Given a CSV file (see load_raw_data), transforms it into Parquet "
                    "in an mlflow artifact called 'ratings-parquet-dir'")
@click.option("--ratings-csv")
@click.option("--max-row-limit", default=10000,
              help="Limit the data size to run comfortably on a laptop.")
def etl_data(ratings_csv, max_row_limit):
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
        if max_row_limit != -1:
            ratings_df = ratings_df.limit(max_row_limit)
        ratings_df.write.parquet(ratings_parquet_dir)
        print("Uploading Parquet ratings: %s" % ratings_parquet_dir)
        mlflow.log_artifacts(ratings_parquet_dir, "ratings-parquet-dir")


if __name__ == '__main__':
    etl_data()
