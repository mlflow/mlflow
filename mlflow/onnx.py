"""
The ``mlflow.onnx`` module provides APIs for logging and loading ONNX models in the MLflow Model
format. This module exports MLflow Models with the following flavors:

ONNX (native) format
    This is the main flavor that can be loaded back as an ONNX model object.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import os
import yaml
import numpy as np

import pandas as pd

from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.file_utils import write_to
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "onnx"


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return list(
        map(
            _get_pinned_requirement,
            [
                "onnx",
                # The ONNX pyfunc representation requires the OnnxRuntime
                # inference engine. Therefore, the conda environment must
                # include OnnxRuntime
                "onnxruntime",
            ],
        )
    )


@experimental
def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    onnx_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save an ONNX model to a path on the local file system.

    :param onnx_model: ONNX model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    import onnx

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path), error_code=RESOURCE_ALREADY_EXISTS
        )
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    model_data_subpath = "model.onnx"
    model_data_path = os.path.join(path, model_data_subpath)

    # Save onnx-model
    onnx.save_model(onnx_model, model_data_path)

    pyfunc.add_to_model(
        mlflow_model, loader_module="mlflow.onnx", data=model_data_subpath, env=_CONDA_ENV_FILE_NAME
    )
    mlflow_model.add_flavor(FLAVOR_NAME, onnx_version=onnx.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))


def _load_model(model_file):
    import onnx

    onnx_model = onnx.load(model_file)
    # Check Formation
    onnx.checker.check_model(onnx_model)
    return onnx_model


class _OnnxModelWrapper:
    def __init__(self, path):
        import onnxruntime

        self.rt = onnxruntime.InferenceSession(path)
        assert len(self.rt.get_inputs()) >= 1
        self.inputs = [(inp.name, inp.type) for inp in self.rt.get_inputs()]
        self.output_names = [outp.name for outp in self.rt.get_outputs()]

    def _cast_float64_to_float32(self, feeds):
        for input_name, input_type in self.inputs:
            if input_type == "tensor(float)":
                feed = feeds.get(input_name)
                if feed is not None and feed.dtype == np.float64:
                    feeds[input_name] = feed.astype(np.float32)
        return feeds

    @experimental
    def predict(self, data):
        """
        :param data: Either a pandas DataFrame, numpy.ndarray or a dictionary.

                     Dictionary input is expected to be a valid ONNX model feed dictionary.

                     Numpy array input is supported iff the model has a single tensor input and is
                     converted into an ONNX feed dictionary with the appropriate key.

                     Pandas DataFrame is converted to ONNX inputs as follows:
                        - If the underlying ONNX model only defines a *single* input tensor, the
                          DataFrame's values are converted to a NumPy array representation using the
                         `DataFrame.values()
                         <https://pandas.pydata.org/pandas-docs/stable/reference/api/
                          pandas.DataFrame.values.html#pandas.DataFrame.values>`_ method.
                        - If the underlying ONNX model defines *multiple* input tensors, each column
                          of the DataFrame is converted to a NumPy array representation.

                      For more information about the ONNX Runtime, see
                      `<https://github.com/microsoft/onnxruntime>`_.
        :return: Model predictions. If the input is a pandas.DataFrame, the predictions are returned
                 in a pandas.DataFrame. If the input is a numpy array or a dictionary the
                 predictions are returned in a dictionary.
        """
        if isinstance(data, dict):
            feed_dict = data
        elif isinstance(data, np.ndarray):
            # NB: We do allow scoring with a single tensor (ndarray) in order to be compatible with
            # supported pyfunc inputs iff the model has a single input. The passed tensor is
            # assumed to be the first input.
            if len(self.inputs) != 1:
                inputs = [x[0] for x in self.inputs]
                raise MlflowException(
                    "Unable to map numpy array input to the expected model "
                    "input. "
                    "Numpy arrays can only be used as input for MLflow ONNX "
                    "models that have a single input. This model requires "
                    "{0} inputs. Please pass in data as either a "
                    "dictionary or a DataFrame with the following tensors"
                    ": {1}.".format(len(self.inputs), inputs)
                )
            feed_dict = {self.inputs[0][0]: data}
        elif isinstance(data, pd.DataFrame):
            if len(self.inputs) > 1:
                feed_dict = {name: data[name].values for (name, _) in self.inputs}
            else:
                feed_dict = {self.inputs[0][0]: data.values}

        else:
            raise TypeError(
                "Input should be a dictionary or a numpy array or a pandas.DataFrame, "
                "got '{}'".format(type(data))
            )

        # ONNXRuntime throws the following exception for some operators when the input
        # contains float64 values. Unfortunately, even if the original user-supplied input
        # did not contain float64 values, the serialization/deserialization between the
        # client and the scoring server can introduce 64-bit floats. This is being tracked in
        # https://github.com/mlflow/mlflow/issues/1286. Meanwhile, we explicitly cast the input to
        # 32-bit floats when needed. TODO: Remove explicit casting when issue #1286 is fixed.
        feed_dict = self._cast_float64_to_float32(feed_dict)
        predicted = self.rt.run(self.output_names, feed_dict)

        if isinstance(data, pd.DataFrame):

            def format_output(data):
                # Output can be list and it should be converted to a numpy array
                # https://github.com/mlflow/mlflow/issues/2499
                data = np.asarray(data)
                return data.reshape(-1)

            response = pd.DataFrame.from_dict(
                {c: format_output(p) for (c, p) in zip(self.output_names, predicted)}
            )
            return response
        else:
            return dict(zip(self.output_names, predicted))


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return _OnnxModelWrapper(path)


@experimental
def load_model(model_uri, dst_path=None):
    """
    Load an ONNX model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see the
                      `Artifacts Documentation <https://www.mlflow.org/docs/latest/
                      tracking.html#artifact-stores>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: An ONNX model instance.

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return _load_model(model_file=onnx_model_artifacts_path)


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    onnx_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Log an ONNX model as an MLflow artifact for the current run.

    :param onnx_model: ONNX model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
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

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.onnx,
        onnx_model=onnx_model,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )
