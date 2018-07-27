from __future__ import absolute_import

import os
import shutil
import yaml


import pyspark
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.base import Transformer


import mleap.version
from mleap.pyspark.spark_support import SimpleSparkSerializer


import mlflow
from mlflow import javafunc 
from mlflow.models import Model

FLAVOR_NAME = "mleap"

MLEAP_MAVEN_PACKAGE_COORDINATES = "mleap-spark_2.11-0.10.1"

LOADER_MODULE = "com.databricks.mlflow.mleap.MLeapModel"

def save_model(spark_model, sample_input, path, mlflow_model=Model()):
    """
    Save Spark MLlib PipelineModel at given local path.

    Uses MLeap persistence mechanism.

    :param spark_model: Spark PipelineModel to be saved. Currently can only save PipelineModels.
    :param sample_input: A sample dataframe that `spark_model` can evaluate 
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config to which this flavor is being added.
    """
    if not isinstance(spark_model, Transformer):
        raise Exception("Unexpected type {}. SparkML model works only with Transformers".format(
            str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel. SparkML can currently only save PipelineModels.")

    dataset = spark_model.transform(sample_input)
    model_path = "file:{mp}".format(mp=os.path.join(path, "model"))
    spark_model.serializeToBundle(path=model_path,
                                  dataset=dataset)

    mlflow_model.add_flavor(FLAVOR_NAME, 
                            mleap_version=mleap.version.__version__, 
                            model_data="model")
    javafunc.add_to_model(mlflow_model, loader_module=LOADER_MODULE, data="model", packages=[MLEAP_MAVEN_PACKAGE_COORDINATES])
    mlflow_model.save(os.path.join(path, "MLmodel"))
