"""
The ``mlflow.onnx`` module provides APIs for logging and loading ONNX models in the MLflow Model
format. This module exports MLflow Models with the following flavors:

ONNX (native) format
    This is the main flavor that can be loaded back as an ONNX model object.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from packaging.version import Version

import mlflow.tracking
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
    _validate_onnx_session_options,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "onnx"
ONNX_EXECUTION_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

_logger = logging.getLogger(__name__)


_MODEL_DATA_SUBPATH = "model.onnx"


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
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


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    onnx_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    onnx_execution_providers=None,
    onnx_session_options=None,
    metadata=None,
    save_as_external_data=True,
):
    """
    Save an ONNX model to a path on the local file system.

    Args:
        onnx_model: ONNX model to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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

        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        onnx_execution_providers: List of strings defining onnxruntime execution providers.
            Defaults to example:
            ``['CUDAExecutionProvider', 'CPUExecutionProvider']``
            This uses GPU preferentially over CPU.
            See onnxruntime API for further descriptions:
            https://onnxruntime.ai/docs/execution-providers/
        onnx_session_options: Dictionary of options to be passed to onnxruntime.InferenceSession.
            For example:
            ``{
            'graph_optimization_level': 99,
            'intra_op_num_threads': 1,
            'inter_op_num_threads': 1,
            'execution_mode': 'sequential'
            }``
            'execution_mode' can be set to 'sequential' or 'parallel'.
            See onnxruntime API for further descriptions:
            https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
        metadata: {{ metadata }}
        save_as_external_data: Save tensors to external file(s).
    """
    import onnx

    if onnx_execution_providers is None:
        onnx_execution_providers = ONNX_EXECUTION_PROVIDERS

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata
    model_data_subpath = _MODEL_DATA_SUBPATH
    model_data_path = os.path.join(path, model_data_subpath)

    # Save onnx-model
    if Version(onnx.__version__) >= Version("1.9.0"):
        onnx.save_model(onnx_model, model_data_path, save_as_external_data=save_as_external_data)
    else:
        onnx.save_model(onnx_model, model_data_path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.onnx",
        data=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    _validate_onnx_session_options(onnx_session_options)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        onnx_version=onnx.__version__,
        data=model_data_subpath,
        providers=onnx_execution_providers,
        onnx_session_options=onnx_session_options,
        code=code_dir_subpath,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
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

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _load_model(model_file):
    import onnx

    onnx.checker.check_model(model_file)
    return onnx.load(model_file)


class _OnnxModelWrapper:
    def __init__(self, path, providers=None):
        import onnxruntime

        # Get the model meta data from the MLModel yaml file which may contain the providers
        # specification.
        local_path = str(Path(path).parent)
        model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

        # Check if the MLModel config has the providers meta data
        if "providers" in model_meta.flavors.get(FLAVOR_NAME).keys():
            providers = model_meta.flavors.get(FLAVOR_NAME)["providers"]
        # If not, then default to the predefined list.
        else:
            providers = ONNX_EXECUTION_PROVIDERS

        sess_options = onnxruntime.SessionOptions()
        options = model_meta.flavors.get(FLAVOR_NAME).get("onnx_session_options")
        if options:
            if inter_op_num_threads := options.get("inter_op_num_threads"):
                sess_options.inter_op_num_threads = inter_op_num_threads
            if intra_op_num_threads := options.get("intra_op_num_threads"):
                sess_options.intra_op_num_threads = intra_op_num_threads
            if execution_mode := options.get("execution_mode"):
                if execution_mode.upper() == "SEQUENTIAL":
                    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
                elif execution_mode.upper() == "PARALLEL":
                    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            if graph_optimization_level := options.get("graph_optimization_level"):
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel(
                    graph_optimization_level
                )
            if extra_session_config := options.get("extra_session_config"):
                for key, value in extra_session_config.items():
                    sess_options.add_session_config_entry(key, value)

        # NOTE: Some distributions of onnxruntime require the specification of the providers
        # argument on calling. E.g. onnxruntime-gpu. The package import call does not differentiate
        #  which architecture specific version has been installed, as all are imported with
        # onnxruntime. onnxruntime documentation says that from v1.9.0 some distributions require
        #  the providers list to be provided on calling an InferenceSession. Therefore the try
        #  catch structure below attempts to create an inference session with just the model path
        #  as pre v1.9.0. If that fails, it will use the providers list call.
        # At the moment this is just CUDA and CPU, and probably should be expanded.
        # A method of user customization has been provided by adding a variable in the save_model()
        # function, which allows the ability to pass the list of execution providers via a
        # optional argument e.g.
        #
        # mlflow.onnx.save_model(..., providers=['CUDAExecutionProvider'...])
        #
        # For details of the execution providers construct of onnxruntime, see:
        # https://onnxruntime.ai/docs/execution-providers/
        #
        # For a information on how execution providers are used with onnxruntime InferenceSession,
        # see the API page below:
        # https://onnxruntime.ai/docs/api/python/api_summary.html#id8
        #

        try:
            self.rt = onnxruntime.InferenceSession(path, sess_options=sess_options)
        except ValueError:
            self.rt = onnxruntime.InferenceSession(
                path, providers=providers, sess_options=sess_options
            )

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

    def predict(self, data, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            data: Either a pandas DataFrame, numpy.ndarray or a dictionary.
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
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions. If the input is a pandas.DataFrame, the predictions are returned
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
                    f"{len(self.inputs)} inputs. Please pass in data as either a "
                    "dictionary or a DataFrame with the following tensors"
                    f": {inputs}."
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
                f"got '{type(data)}'"
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

            return pd.DataFrame.from_dict(
                {c: format_output(p) for (c, p) in zip(self.output_names, predicted)}
            )
        else:
            return dict(zip(self.output_names, predicted))


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.
    """
    return _OnnxModelWrapper(path)


def load_model(model_uri, dst_path=None):
    """
    Load an ONNX model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model, for example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see the
            `Artifacts Documentation <https://www.mlflow.org/docs/latest/
            tracking.html#artifact-stores>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        An ONNX model instance.

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return _load_model(model_file=onnx_model_artifacts_path)


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    onnx_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    onnx_execution_providers=None,
    onnx_session_options=None,
    metadata=None,
    save_as_external_data=True,
):
    """
    Log an ONNX model as an MLflow artifact for the current run.

    Args:
        onnx_model: ONNX model to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        onnx_execution_providers: List of strings defining onnxruntime execution providers.
            Defaults to example:
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            This uses GPU preferentially over CPU.
            See onnxruntime API for further descriptions:
            https://onnxruntime.ai/docs/execution-providers/
        onnx_session_options: Dictionary of options to be passed to onnxruntime.InferenceSession.
            For example:
            ``{
            'graph_optimization_level': 99,
            'intra_op_num_threads': 1,
            'inter_op_num_threads': 1,
            'execution_mode': 'sequential'
            }``
            'execution_mode' can be set to 'sequential' or 'parallel'.
            See onnxruntime API for further descriptions:
            https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
        metadata: {{ metadata }}
        save_as_external_data: Save tensors to external file(s).

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.onnx,
        onnx_model=onnx_model,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        onnx_execution_providers=onnx_execution_providers,
        onnx_session_options=onnx_session_options,
        metadata=metadata,
        save_as_external_data=save_as_external_data,
    )
