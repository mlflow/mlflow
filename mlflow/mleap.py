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

from mlflow.utils.exception import SaveModelException 

FLAVOR_NAME = "mleap"


def add_to_model(mlflow_model, path, spark_model, sample_input):
    """
    :param mlflow_model: MLFlow model config to which this flavor is being added
    :param path: Path of the MLFlow model to which this flavor is being added
    :param spark_Model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                  cannot contain any custom transformers.
    :param sample_input: A sample input that the model can evaluate. This is required by MLeap
                         for data schema inference.
    """
    from pyspark.ml.pipeline import PipelineModel
    from pyspark.ml.base import Transformer
    import mleap.version
    from mleap.pyspark.spark_support import SimpleSparkSerializer

    if sample_input is None:
        raise SaveModelException("A sample input must be specified" 
                                 " in order to add the MLeap flavor!")

    if not isinstance(spark_model, Transformer):
        raise SaveModelException("Unexpected type {}. MLeap model works only with Transformers".format(
            str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise SaveModelException("Not a PipelineModel." 
                                 " MLeap can currently only save PipelineModels.")

    # TODO(czumar): Additional validation - no custom transformers!

    mleap_path_full = os.path.join(path, "mleap")
    mleap_datapath_sub = os.path.join("mleap", "model")
    mleap_datapath_full = os.path.join(path, mleap_datapath_sub)
    if os.path.exists(mleap_path_full):
        raise SaveModelException("MLeap model data path already exists at: {path}".format(
            path=mleap_path_full))

    os.makedirs(mleap_path_full)

    dataset = spark_model.transform(sample_input)
    model_path = "file:{mp}".format(mp=mleap_datapath_full)
    spark_model.serializeToBundle(path=model_path,
                                  dataset=dataset)

    mlflow_model.add_flavor(FLAVOR_NAME, 
                            mleap_version=mleap.version.__version__, 
                            model_data=mleap_datapath_sub)

