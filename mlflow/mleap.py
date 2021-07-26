"""
The ``mlflow.mleap`` module provides an API for saving Spark MLLib models using the
`MLeap <https://github.com/combust/mleap>`_ persistence mechanism.

NOTE:

    You cannot load the MLeap model flavor in Python; you must download it using the
    Java API method ``downloadArtifacts(String runId)`` and load the model
    using the method ``MLeapLoader.loadPipeline(String modelRootPath)``.
"""
import logging
import os
import sys
import traceback
import yaml

import mlflow
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.file_utils import write_to
from mlflow.utils import reraise
from mlflow.utils.annotations import keyword_only

FLAVOR_NAME = "mleap"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    import mleap

    return ["mleap=={}".format(mleap.__version__)]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@keyword_only
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    spark_model,
    sample_input,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Log a Spark MLLib model in MLeap format as an MLflow artifact
    for the current run. The logged model will have the MLeap flavor.

    NOTE:

        You cannot load the MLeap model flavor in Python; you must download it using the
        Java API method ``downloadArtifacts(String runId)`` and load the model
        using the method ``MLeapLoader.loadPipeline(String modelRootPath)``.

    :param spark_model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                        cannot contain any custom transformers.
    :param sample_input: Sample PySpark DataFrame input that the model can evaluate. This is
                         required by MLeap for data schema inference.
    :param artifact_path: Run-relative artifact path.
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}

    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.mleap
        import pyspark
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.feature import HashingTF, Tokenizer
        # training DataFrame
        training = spark.createDataFrame([
            (0, "a b c d e spark", 1.0),
            (1, "b d", 0.0),
            (2, "spark f g h", 1.0),
            (3, "hadoop mapreduce", 0.0) ], ["id", "text", "label"])
        # testing DataFrame
        test_df = spark.createDataFrame([
            (4, "spark i j k"),
            (5, "l m n"),
            (6, "spark hadoop spark"),
            (7, "apache hadoop")], ["id", "text"])
        # Create an MLlib pipeline
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
        lr = LogisticRegression(maxIter=10, regParam=0.001)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training)
        # log parameters
        mlflow.log_param("max_iter", 10)
        mlflow.log_param("reg_param", 0.001)
        # log the Spark MLlib model in MLeap format
        mlflow.mleap.log_model(spark_model=model, sample_input=test_df, artifact_path="mleap-model")
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.mleap,
        spark_model=spark_model,
        sample_input=sample_input,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )


@keyword_only
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    spark_model,
    sample_input,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save a Spark MLlib PipelineModel in MLeap format at a local path.
    The saved model will have the MLeap flavor.

    NOTE:

        You cannot load the MLeap model flavor in Python; you must download it using the
        Java API method ``downloadArtifacts(String runId)`` and load the model
        using the method ``MLeapLoader.loadPipeline(String modelRootPath)``.

    :param spark_model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                  cannot contain any custom transformers.
    :param sample_input: Sample PySpark DataFrame input that the model can evaluate. This is
                         required by MLeap for data schema inference.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: :py:mod:`mlflow.models.Model` to which this flavor is being added.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset) and valid
                      model output (e.g. model predictions generated on the training dataset),
                      for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        signature = infer_signature(train, model.predict(train))
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()

    conda_env, pip_requirements, pip_constraints = (
        _process_pip_requirements(
            get_default_pip_requirements(), pip_requirements, extra_pip_requirements,
        )
        if conda_env is None
        else _process_conda_env(conda_env)
    )

    conda_env_subpath = "conda.yaml"
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    add_to_model(
        mlflow_model=mlflow_model, path=path, spark_model=spark_model, sample_input=sample_input
    )
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


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
        raise Exception("Not a PipelineModel." " MLeap can save only PipelineModels.")
    if sample_input is None:
        raise Exception("A sample input must be specified in order to add the MLeap flavor.")
    if not isinstance(sample_input, DataFrame):
        raise Exception(
            "The sample input must be a PySpark dataframe of type `{df_type}`".format(
                df_type=DataFrame.__module__
            )
        )

    # MLeap's model serialization routine requires an absolute output path
    path = os.path.abspath(path)

    mleap_path_full = os.path.join(path, "mleap")
    mleap_datapath_sub = os.path.join("mleap", "model")
    mleap_datapath_full = os.path.join(path, mleap_datapath_sub)
    if os.path.exists(mleap_path_full):
        raise Exception(
            "MLeap model data path already exists at: {path}".format(path=mleap_path_full)
        )
    os.makedirs(mleap_path_full)

    dataset = spark_model.transform(sample_input)
    model_path = "file:{mp}".format(mp=mleap_datapath_full)
    try:
        spark_model.serializeToBundle(path=model_path, dataset=dataset)
    except Py4JError:
        _handle_py4j_error(
            MLeapSerializationException,
            "MLeap encountered an error while serializing the model. Ensure that the model is"
            " compatible with MLeap (i.e does not contain any custom transformers).",
        )

    try:
        mleap_version = mleap.version.__version__
        _logger.warning(
            "Detected old mleap version %s. Support for logging models in mleap format with "
            "mleap versions 0.15.0 and below is deprecated and will be removed in a future "
            "MLflow release. Please upgrade to a newer mleap version.",
            mleap_version,
        )
    except AttributeError:
        mleap_version = mleap.version
    mlflow_model.add_flavor(FLAVOR_NAME, mleap_version=mleap_version, model_data=mleap_datapath_sub)


def _handle_py4j_error(reraised_error_type, reraised_error_text):
    """
    Logs information about an exception that is currently being handled
    and reraises it with the specified error text as a message.
    """
    traceback.print_exc()
    tb = sys.exc_info()[2]
    reraise(reraised_error_type, reraised_error_type(reraised_error_text), tb)


class MLeapSerializationException(MlflowException):
    """Exception thrown when a model or DataFrame cannot be serialized in MLeap format."""
