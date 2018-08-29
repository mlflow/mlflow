"""
Downloads the MovieLens dataset and saves it as an artifact
"""


from __future__ import print_function

import requests
import tempfile
import os
import zipfile
import pyspark
import mlflow
import click


@click.command(help="Downloads the MovieLens dataset and saves it as an mlflow artifact "
                    " called 'ratings-csv-dir'.")
@click.option("--url", default="http://files.grouplens.org/datasets/movielens/ml-20m.zip")
def load_raw_data(url):
    with mlflow.start_run() as mlrun:
        local_dir = tempfile.mkdtemp()
        local_filename = os.path.join(local_dir, "ml-20m.zip")
        print("Downloading %s to %s" % (url, local_filename))
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        extracted_dir = os.path.join(local_dir, 'ml-20m')
        print("Extracting %s into %s" % (local_filename, extracted_dir))
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(local_dir)

        ratings_file = os.path.join(extracted_dir, 'ratings.csv')

        print("Uploading ratings: %s" % ratings_file)
        mlflow.log_artifact(ratings_file, "ratings-csv-dir")


if __name__ == '__main__':
    load_raw_data()
