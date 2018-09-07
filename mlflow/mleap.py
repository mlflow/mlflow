"""
The ``mlflow.mleap`` module provides an API for saving Spark MLLib models using the
`MLeap <https://github.com/combust/mleap>`_ persistence mechanism.
A companion module for loading MLflow models with the MLeap flavor format is available in the
``mlflow/java`` package.
"""

from __future__ import absolute_import

import os
import sys
import traceback
import json
from six import reraise

import mlflow
from mlflow.models import Model

FLAVOR_NAME = "mleap"


def log_model(spark_model, sample_input, artifact_path):
    """
    Log a Spark MLLib model in MLeap format as an MLflow artifact
    for the current run. The logged model will have the MLeap flavor.

    NOTE:

        The MLeap model flavor cannot be loaded in Python; it must be loaded using the
        Java module within the ``mlflow/java`` package.

    :param spark_model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                        cannot contain any custom transformers.
    :param sample_input: Sample PySpark DataFrame input that the model can evaluate. This is
                         required by MLeap for data schema inference.
    :param artifact_path: Run-relative artifact path.
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.mleap,
                     spark_model=spark_model, sample_input=sample_input)


def save_model(spark_model, sample_input, path, mlflow_model=Model()):
    """
    Save a Spark MLlib PipelineModel in MLeap format at a local path.
    The saved model will have the MLeap flavor.

    NOTE:

        The MLeap model flavor cannot be loaded in Python; it must be loaded using the
        Java module within the ``mlflow/java`` package.

    :param spark_model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                  cannot contain any custom transformers.
    :param sample_input: Sample PySpark DataFrame input that the model can evaluate. This is
                         required by MLeap for data schema inference.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: :py:mod:`mlflow.models.Model` to which this flavor is being added.
    """
    add_to_model(mlflow_model, path, spark_model, sample_input)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def add_to_model(mlflow_model, path, spark_model, sample_input):
    """
    Add the MLeap flavor to an existing MLflow model.

    :param mlflow_model: :py:mod:`mlflow.models.Model` to which this flavor is being added.
    :param path: Path of the model to which this flavor is being added.
    :param spark_model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                        cannot contain any custom transformers.
    :param sample_input: Sample PySpark DataFrame input that the model can evaluate. This is
                         required by MLeap for data schema inference.
    """
    from pyspark.ml.pipeline import PipelineModel
    from pyspark.sql import DataFrame
    import mleap.version
    from mleap.pyspark.spark_support import SimpleSparkSerializer  # pylint: disable=unused-variable
    from py4j.protocol import Py4JError

    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel."
                        " MLeap can save only PipelineModels.")
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
        error_str = ("MLeap encountered an error while serializing the model. Ensure that"
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
