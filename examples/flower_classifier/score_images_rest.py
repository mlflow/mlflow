"""
Example of scoring images with MLflow model deployed to a REST API endpoint.

The MLflow model to be scored is expected to be an instance of KerasImageClassifierPyfunc
(e.g. produced by running this project) and deployed with MLflow prior to invoking this script.
"""
import os
import base64
import requests

import click
import pandas as pd

from mlflow.utils import cli_args


def score_model(path, uri, port):
    """
    Score images on the local path with MLflow model deployed at given uri and port.

    :param path: Path to a single image file or a directory of images.
    :param uri: URI the model is deployed at
    :param port: Port the model is deployed at.
    :return: Server response.
    """
    if os.path.isdir(path):
        filenames = [os.path.join(path, x) for x in os.listdir(path)
                     if os.path.isfile(os.path.join(path, x))]
    else:
        filenames = [path]

    def read_image(x):
        with open(x, "rb") as f:
            return f.read()

    data = pd.DataFrame(data=[base64.encodebytes(read_image(x)) for x in filenames],
                        columns=["image"]).to_json(orient="split")

    response = requests.post(url='{uri}:{port}/invocations'.format(uri=uri, port=port),
                             data=data,
                             headers={"Content-Type": "application/json; format=pandas-split"})

    if response.status_code != 200:
        raise Exception("Status Code {status_code}. {text}".format(
            status_code=response.status_code,
            text=response.text
        ))
    return response


@click.command(help="Score images.")
@click.option("--port", type=click.INT, default=80, help="Port at which the model is deployed.")
@cli_args.MODEL_URI
@click.argument("--data-path", "-d")
def run(data_path, model_uri, port):
    """
    Score images with MLflow deployed deployed at given uri and port and print out the response
    to standard out.
    """
    print(score_model(data_path, model_uri, port).text)


if __name__ == '__main__':
    run()
