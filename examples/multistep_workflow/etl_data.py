"""
Converts the raw CSV form to a Parquet form with just the columns we want
"""

import os
import tempfile

import click
import pyspark

import mlflow


@click.command(
    help="Given a CSV file (see load_raw_data), transforms it into Parquet "
    "in an mlflow artifact called 'ratings-parquet-dir'"
)
@click.option("--ratings-csv")
@click.option(
    "--max-row-limit", default=10000, help="Limit the data size to run comfortably on a laptop."
)
def etl_data(ratings_csv, max_row_limit):
    with mlflow.start_run():
        tmpdir = tempfile.mkdtemp()
        ratings_parquet_dir = os.path.join(tmpdir, "ratings-parquet")
        print(f"Converting ratings CSV {ratings_csv} to Parquet {ratings_parquet_dir}")
        with pyspark.sql.SparkSession.builder.getOrCreate() as spark:
            ratings_df = (
                spark.read.option("header", "true")
                .option("inferSchema", "true")
                .csv(ratings_csv)
                .drop("timestamp")
            )  # Drop unused column
            ratings_df.show()
            if max_row_limit != -1:
                ratings_df = ratings_df.limit(max_row_limit)
            ratings_df.write.parquet(ratings_parquet_dir)
            print(f"Uploading Parquet ratings: {ratings_parquet_dir}")
            mlflow.log_artifacts(ratings_parquet_dir, "ratings-parquet-dir")


if __name__ == "__main__":
    etl_data()
