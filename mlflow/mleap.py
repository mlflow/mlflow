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
from mlflow.exceptions import MlflowException

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
    >>> #Create an MLlib pipeline
    >>> tokenizer = Tokenizer(inputCol="text", outputCol="words")
    >>> hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    >>> lr = LogisticRegression(maxIter=10, regParam=0.001)
    >>> pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    >>> model = pipeline.fit(training)
    >>> #log parameters
    >>> mlflow.log_parameter("max_iter", 10)
    >>> mlflow.log_parameter("reg_param", 0.001)
    >>> #log the Spark MLlib model in MLeap format
    >>> mlflow.mleap.log_model(model, test_df, "mleap-model")
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

    >>> import mlflow
    >>> import mlflow.mleap
    >>> #set values as appropriate
    >>> spark_model = ...
    >>> model_save_dir = ...
    >>> sample_input_df = ...
    >>> #save the spark MLlib model in MLeap flavor
    >>> mlflow.mleap.save_model(spark_model, sample_input_df, model_save_dir)
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

    >>> import mlflow
    >>> import mlflow.mleap
    >>> #set values
    >>> mlflow_model = ...
    >>> spark_model = ...
    >>> model_path_dir = ...
    >>> sample_input_df =
    >>> #add MLeap flavor to our MLflow model
    >>> mlflow.mleap.add_to_model(mlflow_model,model_path_dir, sample_input_df)
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

    try:
        input_schema = _get_mleap_schema(sample_input)
    except Py4JError:
        _handle_py4j_error(
                MLeapSerializationException,
                "Encountered an error while converting the schema of the sample input dataframe to"
                " MLeap format. Please ensure that this dataframe is compatible with MLeap."
                " For example, the dataframe must only contain supported data types, which are"
                " described here:"
                " http://mleap-docs.combust.ml/core-concepts/data-frames/data-types.html.")

    mleap_schemapath_sub = os.path.join("mleap", "schema.json")
    mleap_schemapath_full = os.path.join(path, mleap_schemapath_sub)
    with open(mleap_schemapath_full, "w") as out:
        json.dump(input_schema, out, indent=4)

    mlflow_model.add_flavor(FLAVOR_NAME,
                            mleap_version=mleap.version.__version__,
                            model_data=mleap_datapath_sub,
                            input_schema=mleap_schemapath_sub)


def _get_mleap_schema(dataframe):
    """
    :param dataframe: A PySpark dataframe object

    :return: The schema of the supplied dataframe, in MLeap format. This serialized object of type
    `ml.combust.mleap.core.types.StructType`, represented as a JSON dictionary.
    """
    from pyspark.ml.util import _jvm
    ReflectionUtil = _jvm().py4j.reflection.ReflectionUtil

    # Convert the Spark dataframe's schema to an MLeap schema object.
    # This is equivalent to the Scala function call
    # `org.apache.spark.sql.mleap.TypeConverters.sparkSchemaToMleapSchema(dataframe)`
    tc_clazz = ReflectionUtil.classForName("org.apache.spark.sql.mleap.TypeConverters$")
    tc_inst = tc_clazz.getField("MODULE$").get(tc_clazz)
    mleap_schema_struct = tc_inst.sparkSchemaToMleapSchema(dataframe._jdf)

    # Obtain a JSON representation of the MLeap schema object
    # This is equivalent to the Scala function call
    # `ml.combust.mleap.json.JsonSupport.MleapStructTypeFormat().write(mleap_schema_struct)`
    js_clazz = ReflectionUtil.classForName("ml.combust.mleap.json.JsonSupport$")
    js_inst = js_clazz.getField("MODULE$").get(js_clazz)
    mleap_schema_json = js_inst.MleapStructTypeFormat().write(mleap_schema_struct)
    return json.loads(mleap_schema_json.toString())


def _handle_py4j_error(reraised_error_type, reraised_error_text):
    """
    Logs information about an exception that is currently being handled
    and reraises it with the specified error text as a message.
    """
    traceback.print_exc()
    tb = sys.exc_info()[2]
    reraise(reraised_error_type, reraised_error_type(reraised_error_text), tb)


class MLeapSerializationException(MlflowException):
    """Exception thrown when a model or dataframe cannot be serialized in MLeap format"""
    pass
