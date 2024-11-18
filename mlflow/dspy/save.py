"""Functions for saving DSPY models to MLflow."""

import os
from pathlib import Path
from typing import Any, Optional, Union

import cloudpickle
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.dspy.wrapper import DspyChatModelWrapper, DspyModelWrapper
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import (
    Model,
    ModelInputExample,
    ModelSignature,
    infer_pip_requirements,
)
from mlflow.models.dependencies_schemas import _get_dependencies_schemas
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.rag_signatures import SIGNATURE_FOR_LLM_INFERENCE_TASK
from mlflow.models.resources import Resource, _ResourceBuilder
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracing.provider import trace_disabled
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
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
)
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "dspy"

_MODEL_SAVE_PATH = "model"
_MODEL_DATA_PATH = "data"


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by Dspy flavor. Calls to
        `save_model()` and `log_model()` produce a pip environment that, at minimum, contains these
        requirements.
    """
    return [_get_pinned_requirement("dspy")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to `save_model()` and
        `log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
@trace_disabled  # Suppress traces for internal predict calls while logging model
def save_model(
    model,
    path: str,
    task: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
    code_paths: Optional[list[str]] = None,
    mlflow_model: Optional[Model] = None,
    conda_env: Optional[Union[list[str], str]] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    pip_requirements: Optional[Union[list[str], str]] = None,
    extra_pip_requirements: Optional[Union[list[str], str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    resources: Optional[Union[str, Path, list[Resource]]] = None,
):
    """
    Save a Dspy model.

    This method saves a Dspy model along with metadata such as model signature and conda
    environments to local file system. This method is called inside `mlflow.dspy.log_model()`.

    Args:
        model: an instance of `dspy.Module`. The Dspy model/module to be saved.
        path: local path where the MLflow model is to be saved.
        task: defaults to None. The task type of the model. Can only be `llm/v1/chat` or None for
            now.
        model_config: keyword arguments to be passed to the Dspy Module at instantiation.
        code_paths: {{ code_paths }}
        mlflow_model: an instance of `mlflow.models.Model`, defaults to None. MLflow model
            configuration to which to add the Dspy model metadata. If None, a blank instance will
            be created.
        conda_env: {{ conda_env }}
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        resources: A list of model resources or a resources.yaml file containing a list of
            resources required to serve the model.
    """

    import dspy

    from mlflow.transformers.llm_inference_utils import (
        _LLM_INFERENCE_TASK_KEY,
        _METADATA_LLM_INFERENCE_TASK_KEY,
    )

    if signature:
        num_inputs = len(signature.inputs.inputs)
        if num_inputs == 0:
            raise MlflowException(
                "The model signature's input schema must contain at least one field.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    if task and task not in SIGNATURE_FOR_LLM_INFERENCE_TASK:
        raise MlflowException(
            "Invalid task: {task} at `mlflow.dspy.save_model()` call. The task must be None or one "
            f"of: {list(SIGNATURE_FOR_LLM_INFERENCE_TASK.keys())}",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    saved_example = None
    if input_example is not None:
        path = os.path.abspath(path)
        _validate_and_prepare_target_save_path(path)
        saved_example = _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    with _get_dependencies_schemas() as dependencies_schemas:
        schema = dependencies_schemas.to_dict()
        if schema is not None:
            if mlflow_model.metadata is None:
                mlflow_model.metadata = {}
            mlflow_model.metadata.update(schema)

    model_data_subpath = _MODEL_DATA_PATH
    # Construct new data folder in existing path.
    data_path = os.path.join(path, model_data_subpath)
    os.makedirs(data_path, exist_ok=True)
    # Set the model path to end with ".pkl" as we use cloudpickle for serialization.
    model_subpath = os.path.join(model_data_subpath, _MODEL_SAVE_PATH) + ".pkl"
    model_path = os.path.join(path, model_subpath)
    # Dspy has a global context `dspy.settings`, and we need to save it along with the model.
    dspy_settings = dict(dspy.settings.config)

    # Don't save the trace in the model, which is only useful during the training phase.
    dspy_settings.pop("trace", None)

    # Store both dspy model and settings in `DspyChatModelWrapper` or `DspyModelWrapper` for
    # serialization.
    if task == "llm/v1/chat":
        wrapped_dspy_model = DspyChatModelWrapper(model, dspy_settings, model_config)
    else:
        wrapped_dspy_model = DspyModelWrapper(model, dspy_settings, model_config)

    with open(model_path, "wb") as f:
        cloudpickle.dump(wrapped_dspy_model, f)

    flavor_options = {
        "model_path": model_subpath,
    }

    if task:
        if mlflow_model.signature is None:
            mlflow_model.signature = SIGNATURE_FOR_LLM_INFERENCE_TASK[task]
        flavor_options.update({_LLM_INFERENCE_TASK_KEY: task})
        if mlflow_model.metadata:
            mlflow_model.metadata[_METADATA_LLM_INFERENCE_TASK_KEY] = task
        else:
            mlflow_model.metadata = {_METADATA_LLM_INFERENCE_TASK_KEY: task}

    if saved_example and mlflow_model.signature is None:
        signature = _infer_signature_from_input_example(saved_example, wrapped_dspy_model)
        mlflow_model.signature = signature
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    # Add flavor info to `mlflow_model`.
    mlflow_model.add_flavor(FLAVOR_NAME, code=code_dir_subpath, **flavor_options)
    # Add loader_module, data and env data to `mlflow_model`.
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.dspy",
        code=code_dir_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
    )

    # Add model file size to `mlflow_model`.
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size

    # Add resources if specified.
    if resources is not None:
        if isinstance(resources, (Path, str)):
            serialized_resource = _ResourceBuilder.from_yaml_file(resources)
        else:
            serialized_resource = _ResourceBuilder.from_resources(resources)

        mlflow_model.resources = serialized_resource

    # Save mlflow_model to path/MLmodel.
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = infer_pip_requirements(path, FLAVOR_NAME, fallback=default_reqs)
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

    # Save `constraints.txt` if necessary.
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`.
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
@trace_disabled  # Suppress traces for internal predict calls while logging model
def log_model(
    dspy_model,
    artifact_path: str,
    task: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
    code_paths: Optional[list[str]] = None,
    conda_env: Optional[Union[list[str], str]] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    registered_model_name: Optional[str] = None,
    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Optional[Union[list[str], str]] = None,
    extra_pip_requirements: Optional[Union[list[str], str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    resources: Optional[Union[str, Path, list[Resource]]] = None,
):
    """
    Log a Dspy model along with metadata to MLflow.

    This method saves a Dspy model along with metadata such as model signature and conda
    environments to MLflow.

    Args:
        dspy_model: an instance of `dspy.Module`. The Dspy model to be saved.
        artifact_path: the run-relative path to which to log model artifacts.
        task: defaults to None. The task type of the model. Can only be `llm/v1/chat` or None for
            now.
        model_config: keyword arguments to be passed to the Dspy Module at instantiation.
        code_paths: {{ code_paths }}
        conda_env: {{ conda_env }}
        signature: {{ signature }}
        input_example: {{ input_example }}
        registered_model_name: defaults to None. If set, create a model version under
            `registered_model_name`, also create a registered model if one with the given name does
            not exist.
        await_registration_for: defaults to
            `mlflow.tracking._model_registry.DEFAULT_AWAIT_MAX_SLEEP_SECONDS`. Number of
            seconds to wait for the model version to finish being created and is in ``READY``
            status. By default, the function waits for five minutes. Specify 0 or None to skip
            waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel
            file.
        resources: A list of model resources or a resources.yaml file containing a list of
            resources required to serve the model.

    .. code-block:: python
        :caption: Example

        import dspy
        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.types.schema import ColSpec, Schema

        # Set up the LM.
        lm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=250)
        dspy.settings.configure(lm=lm)


        class CoT(dspy.Module):
            def __init__(self):
                super().__init__()
                self.prog = dspy.ChainOfThought("question -> answer")

            def forward(self, question):
                return self.prog(question=question)


        dspy_model = CoT()

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("test-dspy-logging")

        from mlflow.dspy import log_model

        input_schema = Schema([ColSpec("string")])
        output_schema = Schema([ColSpec("string")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        with mlflow.start_run():
            log_model(
                dspy_model,
                "model",
                input_example="what is 2 + 2?",
                signature=signature,
            )
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.dspy,
        model=dspy_model,
        task=task,
        model_config=model_config,
        code_paths=code_paths,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        resources=resources,
    )
