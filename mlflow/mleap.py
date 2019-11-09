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
from six import reraise

import mlflow
from mlflow.models import Model
from mlflow.exceptions import MlflowException
from mlflow.utils import keyword_only

FLAVOR_NAME = "mleap"


@keyword_only
def log_model(spark_model, sample_input, artifact_path, registered_model_name=None):
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
    :param registered_model_name: Note:: Experimental: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.

    >>> import mlflow
    >>> import mlflow.mleap
    >>> import pyspark
    >>> from pyspark.ml import Pipeline
    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.feature import HashingTF, Tokenizer
    >>># training DataFrame
    >>> training = spark.createDataFrame([
    ...     (0, "a b c d e spark", 1.0),
    ...     (1, "b d", 0.0),
    ...     (2, "spark f g h", 1.0),
    ...     (3, "hadoop mapreduce", 0.0) ], ["id", "text", "label"])
    >>># testing DataFrame
    >>> test_df = spark.createDataFrame([
    ...     (4, "spark i j k"),
    ...     (5, "l m n"),
    ...     (6, "spark hadoop spark"),
    ...     (7, "apache hadoop")], ["id", "text"])
    >>> # Create an MLlib pipeline
    >>> tokenizer = Tokenizer(inputCol="text", outputCol="words")
    >>> hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    >>> lr = LogisticRegression(maxIter=10, regParam=0.001)
    >>> pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    >>> model = pipeline.fit(training)
    >>> #log parameters
    >>> mlflow.log_param("max_iter", 10)
    >>> mlflow.log_param("reg_param", 0.001)
    >>> #log the Spark MLlib model in MLeap format
    >>> mlflow.mleap.log_model(spark_model=model, sample_input=test_df,
    >>>                        artifact_path="mleap-model")
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.mleap,
                     spark_model=spark_model, sample_input=sample_input,
                     registered_model_name=registered_model_name)


@keyword_only
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
    add_to_model(mlflow_model=mlflow_model, path=path, spark_model=spark_model,
                 sample_input=sample_input)
    mlflow_model.save(os.path.join(path, "MLmodel"))


@keyword_only
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

    # MLeap's model serialization routine requires an absolute output path
    path = os.path.abspath(path)

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
    except Py4JError:
        _handle_py4j_error(
                MLeapSerializationException,
                "MLeap encountered an error while serializing the model. Ensure that the model is"
                " compatible with MLeap (i.e does not contain any custom transformers).")

    mlflow_model.add_flavor(FLAVOR_NAME,
                            mleap_version=mleap.version.__version__,
                            model_data=mleap_datapath_sub)


def _handle_py4j_error(reraised_error_type, reraised_error_text):
    """
    Logs information about an exception that is currently being handled
    and reraises it with the specified error text as a message.
    """
    traceback.print_exc()
    tb = sys.exc_info()[2]
    reraise(reraised_error_type, reraised_error_type(reraised_error_text), tb)


class MLeapSerializationException(MlflowException):
    """Exception thrown when a model or DataFrame cannot be serialized in MLeap format"""
    pass
