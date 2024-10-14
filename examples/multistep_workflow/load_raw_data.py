"""
Downloads the MovieLens dataset and saves it as an artifact
"""

import os
import tempfile
import zipfile

import click
import requests

import mlflow


@click.command(
    help="Downloads the MovieLens dataset and saves it as an mlflow artifact "
    "called 'ratings-csv-dir'."
)
@click.option("--url", default="http://files.grouplens.org/datasets/movielens/ml-20m.zip")
def load_raw_data(url):
    with mlflow.start_run():
        local_dir = tempfile.mkdtemp()
        local_filename = os.path.join(local_dir, "ml-20m.zip")
        print(f"Downloading {url} to {local_filename}")
        r = requests.get(url, stream=True)
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        extracted_dir = os.path.join(local_dir, "ml-20m")
        print(f"Extracting {local_filename} into {extracted_dir}")
        with zipfile.ZipFile(local_filename, "r") as zip_ref:
            zip_ref.extractall(local_dir)

        ratings_file = os.path.join(extracted_dir, "ratings.csv")

        print(f"Uploading ratings: {ratings_file}")
        mlflow.log_artifact(ratings_file, "ratings-csv-dir")


if __name__ == "__main__":
    load_raw_data()
