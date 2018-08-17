"""
MLflow integration of the Spark MLlib serialization tool.

This module provides utilities for saving models using the MLeap 
using the MLeap library's persistence mechanism.

A companion module for loading MLFlow models with the MLeap flavor format is available in the
`mlflow/java` package.

For more information about MLeap, see https://github.com/combust/mleap.
"""

from __future__ import absolute_import

import os
import sys
import traceback
import json
from six import reraise

from mlflow.models import Model

FLAVOR_NAME = "mleap"


def save_model(spark_model, sample_input, path, mlflow_model=Model()):
    _add_to_model(mlflow_model, path, spark_model, sample_input)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def _add_to_model(mlflow_model, path, spark_model, sample_input):
    """
    Internal method for adding the MLeap flavor to a pre-existing MLFlow model.

    :param mlflow_model: MLFlow model config to which this flavor is being added.
    :param path: Path of the MLFlow model to which this flavor is being added.
    :param spark_Model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                  cannot contain any custom transformers.
    :param sample_input: A sample PySpark Dataframe input that the model can evaluate. This is 
                         required by MLeap for data schema inference.
    """
    from pyspark.ml.pipeline import PipelineModel
    from pyspark.ml.base import Transformer
    from pyspark.sql import DataFrame 
    import mleap.version
    from mleap.pyspark.spark_support import SimpleSparkSerializer
    from py4j.protocol import Py4JError

    if not isinstance(spark_model, Transformer):
        raise Exception("Unexpected type {}." 
                        " MLeap serialization only works with Spark Transformers".format(
                            str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel." 
                        " MLeap can currently only save PipelineModels.")
    if sample_input is None:
        raise Exception("A sample input must be specified in order to add the MLeap flavor.")
    if not isinstance(sample_input, DataFrame):
        raise Exception("The sample input must be a PySpark dataframe of type `{df_type}`".format(
            df_type=DataFrame.__module__))

    mleap_path_full = os.path.join(path, "mleap")
    mleap_datapath_sub = os.path.join("mleap", "model")
    mleap_datapath_full = os.path.join(path, mleap_datapath_sub)
    if os.path.exists(mleap_path_full):
        raise Exception("MLeap model data path already exists at: {path}".format(
            path=mleap_path_full))
    os.makedirs(mleap_path_full)

    dataset = spark_model.transform(sample_input)
    model_path = "file:{mp}".format(mp=mleap_datapath_full)
    try:
        spark_model.serializeToBundle(path=model_path,
                                      dataset=dataset)
    except Py4JError as e:
        tb = sys.exc_info()[2]
        error_str = ("MLeap encountered an error while serializing the model. Please ensure that"
                     " the model is compatible with MLeap" 
                     " (i.e does not contain any custom transformers). Error text: {err}".format(
                         err=str(e)))
        traceback.print_exc()
        reraise(Exception, error_str, tb)

    input_schema = json.loads(sample_input.schema.json())
    mleap_schemapath_sub = os.path.join("mleap", "schema.json")
    mleap_schemapath_full = os.path.join(path, mleap_schemapath_sub)
    with open(mleap_schemapath_full, "w") as out:
        json.dump(input_schema, out, indent=4)

    mlflow_model.add_flavor(FLAVOR_NAME, 
                            mleap_version=mleap.version.__version__, 
                            model_data=mleap_datapath_sub,
                            input_schema=mleap_schemapath_sub)

