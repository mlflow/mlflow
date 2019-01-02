import os
import base64
import requests

import click
import pandas as pd


def score_model(path, uri, port):
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
@click.argument("model_uri")
@click.argument("input_data_path")
def run(input_data_path, model_uri, port):
    print(score_model(input_data_path, model_uri, port).text)


if __name__ == '__main__':
    run()
