"""
The ``mlflow.promptflow`` module provides an API for logging and loading Promptflow models.
This module exports Promptflow models with the following flavors:

Promptflow (native) format
    This is the main flavor that can be accessed with Promptflow APIs.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _Promptflow:
    https://microsoft.github.io/promptflow
"""

import logging
import os
import pickle
import shutil
import tempfile
from typing import Union, List, Dict, Any, Optional

import cloudpickle

import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example, ModelInputExample
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import ColSpec, DataType, Schema
from mlflow.utils.annotations import experimental
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
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "promptflow"

_MODEL_FLOW_DIRECTORY = "flow"
_MODEL_FLOW_PIP_REQUIREMENTS = "python_requirements_txt"
_MODEL_FLOW_BASE_IMAGE = "image"
_SAVED_MODEL_PATH = "model.pkl"
_UNSUPPORTED_MODEL_ERROR_MESSAGE = (
    "MLflow promptflow flavor only supports instances loaded by ~promptflow.load_flow(), "
    "found {instance_type}."
)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at a minimum, contains these requirements.
    """
    tools_package = None
    try:
        tools_package = _get_pinned_requirement("promptflow-tools")
    except Exception:  # pylint: disable=broad-except
        pass
    requirements = [tools_package] if tools_package else []
    return requirements + [_get_pinned_requirement("promptflow")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature=None,
    input_example=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Log a Promptflow model as an MLflow artifact for the current run.

    :param model: A promptflow model loaded by :func:`promptflow.load_flow()`.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.

    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output
                      :py:class:`Schema <mlflow.types.Schema>`.
                      If not specified, the model signature would be set according to
                      `lc_model.input_keys` and `lc_model.output_keys` as columns names, and
                      `DataType.string` as the column type.
                      Alternatively, you can explicitly specify the model signature.
                      The model signature can be :py:func:`inferred
                      <mlflow.models.infer_signature>` from datasets with valid model input
                      (e.g. the training dataset with target column omitted) and valid model
                      output (e.g. model predictions generated on the training dataset),
                      for example:

                      .. code-block:: python

                        from mlflow.models import infer_signature

                        chain = LLMChain(llm=llm, prompt=prompt)
                        prediction = chain.run(input_str)
                        input_columns = [
                            {"type": "string", "name": input_key} for input_key in chain.input_keys
                        ]
                        signature = infer_signature(input_columns, predictions)

    :param input_example: {{ input_example }}

    :param await_registration_for: Number of seconds to wait for the model version
                        to finish being created and is in ``READY`` status.
                        By default, the function waits for five minutes.
                        Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.promptflow,
        registered_model_name=registered_model_name,
        model=model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
    )


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None):
    """
    Save a Promptflow model to a path on the local file system.

    :param model: A promptflow model loaded by :func:`promptflow.load_flow()`.
    :param path: Local path where the serialized model (as YAML) is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      If not specified, the model signature would be set according to
                      `lc_model.input_keys` and `lc_model.output_keys` as columns names, and
                      `DataType.string` as the column type.
                      Alternatively, you can explicitly specify the model signature.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models import infer_signature

                        chain = LLMChain(llm=llm, prompt=prompt)
                        prediction = chain.run(input_str)
                        input_columns = [
                            {"type": "string", "name": input_key} for input_key in chain.input_keys
                        ]
                        signature = infer_signature(input_columns, predictions)

    :param input_example: {{ input_example }}
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    """
    import promptflow
    from promptflow import load_flow
    from promptflow.contracts.tool import ValueType
    from promptflow._sdk.entities._flow import Flow
    from promptflow._sdk._utils import _merge_local_code_and_additional_includes
    from promptflow._sdk.operations._run_submitter import remove_additional_includes

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    # check if path exists
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    model_flow_path = os.path.join(path, _MODEL_FLOW_DIRECTORY)

    if not isinstance(model, Flow):
        raise mlflow.MlflowException.invalid_parameter_value(
            _UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=type(model).__name__)
        )

    # Resolve additional includes in flow
    with _merge_local_code_and_additional_includes(code_path=model.code) as resolved_model_dir:
        remove_additional_includes(Path(resolved_model_dir))
        shutil.copytree(src=resolved_model_dir, dst=model_flow_path)
    # Get flow env in flow dag
    flow_env = _resolve_env_from_flow(model.flow_dag_path)

    def _parse_pf_type(type_str):
        if type_str == ValueType.INT.value:
            return DataType.integer
        if type_str == ValueType.DOUBLE.value:
            return DataType.double
        if type_str == ValueType.BOOL.value:
            return DataType.boolean
        # Return string for list, object and other types
        return DataType.string

    # infer signature if signature is not provided
    if signature is None:
        input_columns = [
            ColSpec(type=_parse_pf_type(typ), name=k) for k, typ in model.inputs.items()
        ]
        input_schema = Schema(input_columns)

        output_columns = [
            ColSpec(type=_parse_pf_type(typ), name=k) for k, typ in model.outputs.items()
        ]
        output_schema = Schema(output_columns)

        signature = ModelSignature(input_schema, output_schema)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature

    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    # update flavor info to mlflow_model
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        code=code_dir_subpath,
        promptflow_version=promptflow.__version__,
        entry=_MODEL_FLOW_DIRECTORY,
        **flow_env
    )

    # append loader_module, data and env data to mlflow_model
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.promptflow",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    # save mlflow_model to path/MLmodel
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
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


def _resolve_env_from_flow(flow_dag_path):
    with open(flow_dag_path, "r") as f:
        flow_dict = yaml.safe_load(f)
    environment = flow_dict.get("environment", {})
    return environment


class _PromptflowModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model_invoker = None

    def predict(  # pylint: disable=unused-argument
        self,
        data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]]],
        params: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    ):
        """
        :param data: Model input data.
        :param params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        :return: Model predictions.
        """
        from promptflow._sdk._serving.flow_invoker import FlowInvoker
        if self.model_invoker is None:
            # TODO: Support more choice in model config?
            self.model_invoker = FlowInvoker(self.model, connection_provider="local")

        if isinstance(data, pd.DataFrame):
            messages = data.to_dict(orient="records")
        elif isinstance(data, list) and (
            all(isinstance(d, str) for d in data) or all(isinstance(d, dict) for d in data)
        ):
            messages = data
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                "Input must be a pandas DataFrame or a list of strings or a list of dictionaries",
            )

        results = []
        for message in messages:
            results.append(self.model_invoker.invoke(message))
        return results


def _load_pyfunc(path):
    """
    Load PyFunc implementation for Promptflow. Called by ``pyfunc.load_model``.
    :param path: Local filesystem path to the MLflow Model with the ``promptflow`` flavor.
    """
    model_flow_path = os.path.join(path, _MODEL_FLOW_DIRECTORY)
    # with open(os.path.join(path, _SAVED_MODEL_PATH), "rb") as _in:
    #     model = cloudpickle.load(_in)
    return _PromptflowModelWrapper(model_flow_path)


@experimental
def load_model(model_uri, dst_path=None):
    """
    Load a Promptflow model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A Promptflow model instance
    """
    from promptflow._sdk._load_functions import load_flow
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    model_data_path = os.path.join(local_model_path, _MODEL_FLOW_DIRECTORY)
    return load_flow(model_data_path)
