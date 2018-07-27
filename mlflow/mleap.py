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
from mlflow import pyfunc
from mlflow.models import Model

PACKAGES_OUTPUT_LIST_TITLE = "packages"

FLAVOR_NAME = "mleap"

# def log_model(spark_model, artifact_path, conda_env=None, jars=None):
#     """
#     Log a Spark MLlib model as an MLflow artifact for the current run.
#
#     :param spark_model: PipelineModel to be saved.
#     :param artifact_path: Run-relative artifact path.
#     :param conda_env: Path to a Conda environment file. If provided, this defines enrionment for
#            the model. At minimum, it should specify python, pyspark and mlflow with appropriate
#            versions.
#     :param jars: List of jars needed by the model.
#     """
#     return Model.log(artifact_path=artifact_path, flavor=mlflow.spark, spark_model=spark_model,
#                      jars=jars, conda_env=conda_env)


# def save_model(spark_model, path, mlflow_model=Model(), pkg_deps=None, jars=None):
def save_model(spark_model, sample_input, path, mlflow_model=Model(), pkg_deps=None):
    """
    Save Spark MLlib PipelineModel at given local path.

    Uses MLeap persistence mechanism.

    :param spark_model: Spark PipelineModel to be saved. Currently can only save PipelineModels.
    :param sample_input: A sample dataframe that `spark_model` can evaluate 
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config to which this flavor is being added.
    # :param pkg_deps: A set of Maven coordinates referencing packages required for model evaluation.
    # :param jars: List of jars needed by the model.
    """
    # if jars:
    #     raise Exception("jar dependencies are not implemented")
    if not isinstance(spark_model, Transformer):
        raise Exception("Unexpected type {}. SparkML model works only with Transformers".format(
            str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel. SparkML can currently only save PipelineModels.")

    dataset = spark_model.transform(sample_input)
    model_path = "file:{mp}".format(mp=os.path.join(path, "model"))
    spark_model.serializeToBundle(path=model_path,
                                  dataset=dataset)

    # pkgs_yaml = _get_packages_yaml(pkg_deps)
    # pkgs_subpath = "packages"
    # with open(os.path.join(path, pkgs_subpath), "w") as f:
    #     f.write(pkgs_yaml)

    mlflow_model.add_flavor(FLAVOR_NAME, 
                            mleap_version=mleap.version.__version__, 
                            # packages="packages",
                            model_data="model")
    # javafunc.add_to_model(mlflow_model, loader_module="MLFLOW_JAVA_PACKAGE_MODULE", data="model")
    mlflow_model.save(os.path.join(path, "MLmodel"))

def _get_packages_yaml(pkg_deps):
    yaml_dict = { PACKAGES_OUTPUT_LIST_TITLE : pkg_deps }
    return yaml.safe_dump(yaml_dict, default_flow_style=False)
    
