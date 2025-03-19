"""
Example of scoring images with MLflow model produced by running this project in Spark.

The MLflow model is loaded to Spark using ``mlflow.pyfunc.spark_udf``. The images are read as binary
data and represented as base64 encoded string column and passed to the model. The results are
returned as a column with predicted class label, class id and probabilities for each class encoded
as an array of strings.

"""

import base64
import os

import click
import pandas as pd
import pyspark
from pyspark.sql.types import ArrayType, Row, StringType, StructField, StructType

import mlflow
import mlflow.pyfunc
from mlflow.utils import cli_args


def read_image_bytes_base64(path):
    with open(path, "rb") as f:
        return str(base64.encodebytes(f.read()), encoding="utf8")


def read_images(spark, filenames):
    filenames_rdd = spark.sparkContext.parallelize(filenames)
    schema = StructType(
        [StructField("filename", StringType(), True), StructField("image", StringType(), True)]
    )
    return filenames_rdd.map(lambda x: Row(filename=x, image=read_image_bytes_base64(x))).toDF(
        schema=schema
    )


def score_model(spark, data_path, model_uri):
    if os.path.isdir(data_path):
        filenames = [
            os.path.abspath(os.path.join(data_path, x))
            for x in os.listdir(data_path)
            if os.path.isfile(os.path.join(data_path, x))
        ]
    else:
        filenames = [data_path]

    image_classifier_udf = mlflow.pyfunc.spark_udf(
        spark=spark, model_uri=model_uri, result_type=ArrayType(StringType())
    )

    image_df = read_images(spark, filenames)

    raw_preds = (
        image_df.withColumn("prediction", image_classifier_udf("image"))
        .select(["filename", "prediction"])
        .toPandas()
    )
    # load the pyfunc model to get our domain
    pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    preds = pd.DataFrame(raw_preds["filename"], index=raw_preds.index)
    preds[pyfunc_model._column_names] = pd.DataFrame(
        raw_preds["prediction"].values.tolist(),
        columns=pyfunc_model._column_names,
        index=raw_preds.index,
    )

    preds = pd.DataFrame(raw_preds["filename"], index=raw_preds.index)

    preds[pyfunc_model._column_names] = pd.DataFrame(
        raw_preds["prediction"].values.tolist(),
        columns=pyfunc_model._column_names,
        index=raw_preds.index,
    )
    return preds.to_json(orient="records")


@click.command(help="Score images.")
@cli_args.MODEL_URI
@click.argument("data-path")
def run(data_path, model_uri):
    with (
        pyspark.sql.SparkSession.builder.config(key="spark.python.worker.reuse", value=True)
        .config(key="spark.ui.enabled", value=False)
        .master("local-cluster[2, 1, 1024]")
        .getOrCreate() as spark
    ):
        # ignore spark log output
        spark.sparkContext.setLogLevel("OFF")
        print(score_model(spark, data_path, model_uri))


if __name__ == "__main__":
    run()
