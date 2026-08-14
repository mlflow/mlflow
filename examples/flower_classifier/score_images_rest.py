"""
Example of scoring images with MLflow model deployed to a REST API endpoint.

The MLflow model to be scored is expected to be an instance of KerasImageClassifierPyfunc
(e.g. produced by running this project) and deployed with MLflow prior to invoking this script.
"""

import base64
import os

import click
import pandas as pd
import requests

from mlflow.utils import cli_args


def score_model(path, host, port):
    """
    Score images on the local path with MLflow model deployed at given uri and port.

    Args:
        path: Path to a single image file or a directory of images.
        host: Host the model is deployed at.
        port: Port the model is deployed at.

    Returns:
        Server response.
    """
    if os.path.isdir(path):
        filenames = [
            os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))
        ]
    else:
        filenames = [path]

    def read_image(x):
        with open(x, "rb") as f:
            return f.read()

    data = pd.DataFrame(
        data=[base64.encodebytes(read_image(x)) for x in filenames], columns=["image"]
    ).to_json(orient="split")

    response = requests.post(
        url=f"{host}:{port}/invocations",
        data={
            "dataframe_split": data,
        },
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        raise Exception(f"Status Code {response.status_code}. {response.text}")
    return response


@click.command(help="Score images.")
@click.option("--port", type=click.INT, default=80, help="Port at which the model is deployed.")
@cli_args.HOST
@click.argument("data-path")
def run(data_path, host, port):
    """
    Score images with MLflow deployed deployed at given uri and port and print out the response
    to standard out.
    """
    print(score_model(data_path, host, port).text)


if __name__ == "__main__":
    run()
