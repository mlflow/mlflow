import os
import base64
import requests

import click
import pandas as pd

import pyspark


import mlflow
import mlflow.pyfunc


@click.command(help="Score images")
@click.argument("image_dir", type=click.STRING, help="Path to an image file or a directory of image "
                                                "files to be scored.")
@click.option("--run_id", type=click.STRING, default="", help="URI where the model is deployed.")
@click.option("--artifact-path", type=click.STRING, default="model", help="")
def run(image_dir, artifact_path, run_id):
    conf = pyspark.SparkConf()
    conf.set(key="spark_session.python.worker.reuse", value=True)
    sc = pyspark.SparkContext(master="local-cluster[2, 1, 1024]", conf=conf).getOrCreate()
    mlflow.pyfunc.spark_udf(spark=sc, path=artifact_path, run_id=run_id)
    if os.path.isdir(path):
        filenames = [os.path.abspath(os.path.join(path, x)) for x in os.listdir(path)
                     if os.path.isfile(os.path.join(path, x))]
    else:
        raise Exception("Expected a directory.")

    def read_image(x):
        with open(x, "rb") as f:
            return f.read()

    sc.
    data = pd.DataFrame(data=[base64.encodebytes(read_image(x)) for x in filenames],
                        columns=["image"]).to_json(orient="split")

    response = requests.post(url='{uri}:%{port}/invocations'.format(uri=uri, port=port),
                             data=data,
                             headers={"Content-Type": "application/json; format=pandas-split"})
    print(response)


if __name__ == '__main__':
    run()
