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
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
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
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "promptflow"

_MODEL_FLOW_DIRECTORY = "flow"
_FLOW_ENV_REQUIREMENTS = "python_requirements_txt"
_UNSUPPORTED_MODEL_ERROR_MESSAGE = (
    "MLflow promptflow flavor only supports instance defined with 'flow.dag.yaml' file "
    "and loaded by ~promptflow.load_flow(), found {instance_type}."
)
_INVALID_PREDICT_INPUT_ERROR_MESSAGE = (
    "Input must be a pandas DataFrame with only 1 row "
    "or a dictionary contains flow inputs key-value pairs."
)
_CONNECTION_PROVIDER_CONFIG_KEY = "connection_provider"
_CONNECTION_OVERRIDES_CONFIG_KEY = "connection_overrides"


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at a minimum, contains these requirements.
    """
    tools_package = None
    try:
        # Note: If user don't use built-in tool in their flow,
        # then promptflow-tools is not a mandatory dependency.
        tools_package = _get_pinned_requirement("promptflow-tools")
    except ImportError:  # pylint: disable=broad-except
        pass
    requirements = [tools_package] if tools_package else []
    return requirements + [_get_pinned_requirement("promptflow")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
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
    model_config: Optional[Dict[str, Any]] = None,
    example_no_conversion=None,
):
    """
    Log a Promptflow model as an MLflow artifact for the current run.

    Args:
        model: A promptflow model loaded by `promptflow.load_flow()`.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        model_config: A dict of valid overrides that can be applied to a flow instance
            during inference. These arguments are used exclusively for the case of loading
            the model as a ``pyfunc`` Model.
            These values are not applied to a returned flow from a call to
            ``mlflow.promptflow.load_model()``.
            To override configs for a loaded flow with promptflow flavor,
            please update the ``pf_model.context`` directly.

            Configs that can be overridden includes:

            ``connection_provider`` - The connection provider to use for the flow. Reach
            https://microsoft.github.io/promptflow/how-to-guides/set-global-configs.html#connection-provider
            for more details on how to set connection provider.

            ``connection_overrides`` - The connection name overrides to use for the flow.
            Example: ``{"aoai_connection": "azure_open_ai_connection"}``.
            The node with reference to connection 'aoai_connection' will be resolved to
            the actual connection 'azure_open_ai_connection'.


            An example of providing overrides for a model to use azure machine
            learning workspace connection:

            .. code-block:: python

                flow_folder = Path(__file__).parent / "basic"
                flow = load_flow(flow_folder)

                workspace_resource_id = (
                    "azureml://subscriptions/{your-subscription}/resourceGroups/{your-resourcegroup}"
                    "/providers/Microsoft.MachineLearningServices/workspaces/{your-workspace}"
                )
                model_config = {
                    "connection_provider": workspace_resource_id,
                    "connection_overrides": {"local_conn_name": "remote_conn_name"},
                }

                with mlflow.start_run():
                    logged_model = mlflow.promptflow.log_model(
                        flow, artifact_path="promptflow_model", model_config=model_config
                    )
        example_no_conversion: {{ example_no_conversion }}

    Returns
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
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
        model_config=model_config,
        example_no_conversion=example_no_conversion,
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
    metadata=None,
    model_config: Optional[Dict[str, Any]] = None,
    example_no_conversion=None,
):
    """
    Save a Promptflow model to a path on the local file system.

    Args:
        model: A promptflow model loaded by `promptflow.load_flow()`.
        path: Local path where the serialized model (as YAML) is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        model_config: A dict of valid overrides that can be applied to a flow instance
            during inference. These arguments are used exclusively for the case of loading
            the model as a ``pyfunc`` Model.
            These values are not applied to a returned flow from a call to
            ``mlflow.promptflow.load_model()``.
            To override configs for a loaded flow with promptflow flavor,
            please update the ``pf_model.context`` directly.

            Configs that can be overridden includes:

            ``connection_provider`` - The connection provider to use for the flow. Reach
            https://microsoft.github.io/promptflow/how-to-guides/set-global-configs.html#connection-provider
            for more details on how to set connection provider.

            ``connection_overrides`` - The connection name overrides to use for the flow.
            Example: ``{"aoai_connection": "azure_open_ai_connection"}``.
            The node with reference to connection 'aoai_connection' will be resolved to
            the actual connection 'azure_open_ai_connection'.


            An example of providing overrides for a model to use azure machine
            learning workspace connection:

            .. code-block:: python

                flow_folder = Path(__file__).parent / "basic"
                flow = load_flow(flow_folder)

                workspace_resource_id = (
                    "azureml://subscriptions/{your-subscription}/resourceGroups/{your-resourcegroup}"
                    "/providers/Microsoft.MachineLearningServices/workspaces/{your-workspace}"
                )
                model_config = {
                    "connection_provider": workspace_resource_id,
                    "connection_overrides": {"local_conn_name": "remote_conn_name"},
                }

                with mlflow.start_run():
                    logged_model = mlflow.promptflow.log_model(
                        flow, artifact_path="promptflow_model", model_config=model_config
                    )
        example_no_conversion: {{ example_no_conversion }}
    """
    import promptflow
    from promptflow._sdk._mlflow import (
        DAG_FILE_NAME,
        Flow,
        _merge_local_code_and_additional_includes,
        remove_additional_includes,
    )

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if (
        not isinstance(model, Flow)
        or not hasattr(model, "flow_dag_path")
        or not hasattr(model, "code")
    ):
        raise mlflow.MlflowException.invalid_parameter_value(
            _UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=type(model).__name__)
        )

    # check if path exists
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    # Copy to 'flow' directory to get files merged with flow files.
    code_dir_subpath = _validate_and_copy_code_paths(
        code_paths, path, default_subpath=_MODEL_FLOW_DIRECTORY
    )

    model_flow_path = os.path.join(path, _MODEL_FLOW_DIRECTORY)

    # Resolve additional includes in flow
    with _merge_local_code_and_additional_includes(code_path=model.code) as resolved_model_dir:
        remove_additional_includes(Path(resolved_model_dir))
        shutil.copytree(src=resolved_model_dir, dst=model_flow_path, dirs_exist_ok=True)
    # Get flow env in flow dag
    flow_env = _resolve_env_from_flow(model.flow_dag_path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path, example_no_conversion)

    if signature is None and saved_example is not None:
        wrapped_model = _PromptflowModelWrapper(model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    # update flavor info to mlflow_model
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        version=promptflow.__version__,
        entry=f"{_MODEL_FLOW_DIRECTORY}/{DAG_FILE_NAME}",
        **flow_env,
    )

    # append loader_module, data and env data to mlflow_model
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.promptflow",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        model_config=model_config,
    )

    # save mlflow_model to path/MLmodel
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
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
    with open(flow_dag_path) as f:
        flow_dict = yaml.safe_load(f)
    environment = flow_dict.get("environment", {})
    if _FLOW_ENV_REQUIREMENTS in environment:
        # Append entry path to requirements
        environment[
            _FLOW_ENV_REQUIREMENTS
        ] = f"{_MODEL_FLOW_DIRECTORY}/{environment[_FLOW_ENV_REQUIREMENTS]}"
    return environment


class _PromptflowModelWrapper:
    def __init__(self, model, model_config: Optional[Dict[str, Any]] = None):
        from promptflow._sdk._mlflow import FlowInvoker

        self.model = model
        # TODO: Improve this if we have more configs afterwards
        model_config = model_config or {}
        connection_provider = model_config.get(_CONNECTION_PROVIDER_CONFIG_KEY, "local")
        _logger.info("Using connection provider: %s", connection_provider)
        connection_overrides = model_config.get(_CONNECTION_OVERRIDES_CONFIG_KEY, None)
        _logger.info("Using connection overrides: %s", connection_overrides)
        self.model_invoker = FlowInvoker(
            self.model,
            connection_provider=connection_provider,
            connections_name_overrides=connection_overrides,
        )

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.model

    def predict(  # pylint: disable=unused-argument
        self,
        data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]]],
        params: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    ) -> Union[dict, list]:
        """
        Args:
            data: Model input data. Either a pandas DataFrame with only 1 row or a dictionary.

                     .. code-block:: python
                        loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
                        # Predict on a flow input dictionary.
                        print(loaded_model.predict({"text": "Python Hello World!"}))

            params: Additional parameters to pass to the model for inference.

        Returns
            Model predictions. Dict type, example ``{"output": "\n\nprint('Hello World!')"}``
        """
        if isinstance(data, pd.DataFrame):
            messages = data.to_dict(orient="records")
            if len(messages) > 1:
                raise mlflow.MlflowException.invalid_parameter_value(
                    _INVALID_PREDICT_INPUT_ERROR_MESSAGE
                )
            messages = messages[0]
            return [self.model_invoker.invoke(messages)]
        elif isinstance(data, dict):
            messages = data
            return self.model_invoker.invoke(messages)
        raise mlflow.MlflowException.invalid_parameter_value(_INVALID_PREDICT_INPUT_ERROR_MESSAGE)


def _load_pyfunc(path, model_config: Optional[Dict[str, Any]] = None):
    """
    Load PyFunc implementation for Promptflow. Called by ``pyfunc.load_model``.

    Args
        path: Local filesystem path to the MLflow Model with the ``promptflow`` flavor.
    """
    from promptflow import load_flow

    model_flow_path = os.path.join(path, _MODEL_FLOW_DIRECTORY)
    model = load_flow(model_flow_path)
    return _PromptflowModelWrapper(model=model, model_config=model_config)


@experimental
def load_model(model_uri, dst_path=None):
    """
    Load a Promptflow model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns
        A Promptflow model instance
    """
    from promptflow import load_flow

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    model_data_path = os.path.join(local_model_path, _MODEL_FLOW_DIRECTORY)
    return load_flow(model_data_path)
