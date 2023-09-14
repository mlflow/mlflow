"""
The ``mlflow.mleap`` module provides an API for saving Spark MLLib models using the
`MLeap <https://github.com/combust/mleap>`_ persistence mechanism.

.. warning:

    The mleap flavor is deprecated and will be removed in a future release of MLflow.

.. note:

    You cannot load the MLeap model flavor in Python; you must download it using the
    Java API method ``downloadArtifacts(String runId)`` and load the model
    using the method ``MLeapLoader.loadPipeline(String modelRootPath)``.
"""
import logging
import os
import pathlib
import sys
import traceback

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.utils import reraise
from mlflow.utils.annotations import deprecated, keyword_only
from mlflow.utils.file_utils import path_to_local_file_uri

FLAVOR_NAME = "mleap"

_logger = logging.getLogger(__name__)


@deprecated(alternative="mlflow.onnx", since="2.6.0")
@keyword_only
def log_model(
    spark_model,
    sample_input,
    artifact_path,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    metadata=None,
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
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.

    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.


    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.mleap
        import pyspark
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.feature import HashingTF, Tokenizer

        # training DataFrame
        training = spark.createDataFrame(
            [
                (0, "a b c d e spark", 1.0),
                (1, "b d", 0.0),
                (2, "spark f g h", 1.0),
                (3, "hadoop mapreduce", 0.0),
            ],
            ["id", "text", "label"],
        )
        # testing DataFrame
        test_df = spark.createDataFrame(
            [(4, "spark i j k"), (5, "l m n"), (6, "spark hadoop spark"), (7, "apache hadoop")],
            ["id", "text"],
        )
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
        mlflow.mleap.log_model(
            spark_model=model, sample_input=test_df, artifact_path="mleap-model"
        )
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.mleap,
        spark_model=spark_model,
        sample_input=sample_input,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        metadata=metadata,
    )


@deprecated(alternative="mlflow.onnx", since="2.6.0")
@keyword_only
def save_model(
    spark_model,
    sample_input,
    path,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    metadata=None,
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

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset) and valid
                      model output (e.g. model predictions generated on the training dataset),
                      for example:

                      .. code-block:: python

                        from mlflow.models import infer_signature

                        train = df.drop_column("target_label")
                        signature = infer_signature(train, model.predict(train))
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    """
    if mlflow_model is None:
        mlflow_model = Model()

    add_to_model(
        mlflow_model=mlflow_model, path=path, spark_model=spark_model, sample_input=sample_input
    )
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


@deprecated(alternative="mlflow.onnx", since="2.6.0")
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
    import mleap.version

    # This import statement adds `serializeToBundle` and `deserializeFromBundle` to `Transformer`:
    # https://github.com/combust/mleap/blob/37f6f61634798118e2c2eb820ceeccf9d234b810/python/mleap/pyspark/spark_support.py#L32-L33
    from mleap.pyspark.spark_support import SimpleSparkSerializer  # noqa: F401
    from py4j.protocol import Py4JError
    from pyspark.ml.pipeline import PipelineModel
    from pyspark.sql import DataFrame

    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel. MLeap can save only PipelineModels.")
    if sample_input is None:
        raise Exception("A sample input must be specified in order to add the MLeap flavor.")
    if not isinstance(sample_input, DataFrame):
        raise Exception(
            f"The sample input must be a PySpark dataframe of type `{DataFrame.__module__}`"
        )

    # MLeap's model serialization routine requires an absolute output path
    path = os.path.abspath(path)

    mleap_path_full = os.path.join(path, "mleap")
    mleap_datapath_sub = os.path.join("mleap", "model")
    mleap_datapath_full = os.path.join(path, mleap_datapath_sub)
    if os.path.exists(mleap_path_full):
        raise Exception(f"MLeap model data path already exists at: {mleap_path_full}")
    os.makedirs(mleap_path_full)

    dataset = spark_model.transform(sample_input)
    if os.name == "nt":
        # NB: On Windows, MLeap requires the "file://" prefix in order to correctly
        # parse the model data path, even though the result is not a correct URI.
        # None of "file:", "file:/", or "file:///", which would be canonically correct,
        # work properly
        model_path = "file://" + str(pathlib.Path(mleap_datapath_full).as_posix())
    else:
        model_path = path_to_local_file_uri(mleap_datapath_full)

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
