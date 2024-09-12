"""Functions for saving DSPY models to MLflow."""

import logging
import os
from importlib.metadata import version

import cloudpickle
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.dspy.wrapper import DspyModelWrapper
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import (
    Model,
    ModelInputExample,
    ModelSignature,
    infer_pip_requirements,
)
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
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
from mlflow.utils.model_utils import _validate_and_prepare_target_save_path
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "dspy"

_MODEL_SAVE_PATH = "model"
_MODEL_DATA_PATH = "data"

_logger = logging.getLogger(__name__)


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
def save_model(
    model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    save_model_kwargs=None,
    metadata=None,
):
    """
    Save a Dspy model.

    This method saves a Dspy model along with metadata such as model signature and conda
    environments to local file system. This method is called inside `mlflow.dspy.log_model()`.

    Args:
        model: an instance of `dspy.Module`. The Dspy model/module to be saved.
        path: local path where the MLflow model is to be saved.
        conda_env: {{ conda_env }}
        mlflow_model: an instance of `mlflow.models.Model`, defaults to None. MLflow model
            configuration to which to add the Dspy model metadata. If None, a blank instance will
            be created.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        save_model_kwargs: A dict of kwargs to pass save method.
        metadata: {{ metadata }}
    """

    import dspy

    if signature is None:
        _logger.warning("You are saving a dspy model without specifying model signature.")
    else:
        num_inputs = len(signature.inputs.inputs)
        if num_inputs == 0:
            raise MlflowException(
                "The model signature's input schema must contain at least one field.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        path = os.path.abspath(path)
        _validate_and_prepare_target_save_path(path)
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    save_model_kwargs = save_model_kwargs or {}

    model_data_subpath = _MODEL_DATA_PATH
    # Construct new data folder in existing path.
    data_path = os.path.join(path, model_data_subpath)
    os.makedirs(data_path, exist_ok=True)

    model_subpath = os.path.join(model_data_subpath, _MODEL_SAVE_PATH)
    # Set the model path to end with ".pkl" as we use cloudpickle for serialization.
    model_path = os.path.join(path, model_subpath) + ".pkl"
    dspy_settings = dict(dspy.settings.config)
    if "trace" in dspy_settings:
        # Don't save the trace in the model, which is only useful during the training phase.
        del dspy_settings["trace"]
    wrapped_dspy_model = DspyModelWrapper(model, dspy_settings)

    with open(model_path, "wb") as f:
        cloudpickle.dump(wrapped_dspy_model, f)

    flavor_options = {
        "model_path": model_subpath + ".pkl",
        "dspy_version": version("dspy-ai"),
    }

    # Add flavor info to `mlflow_model`.
    mlflow_model.add_flavor(FLAVOR_NAME, **flavor_options)

    # Add loader_module, data and env data to `mlflow_model`.
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.dspy",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
    )

    # Add model file size to `mlflow_model`.
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size

    # save mlflow_model to path/MLmodel
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

    # dspy's pypi name is dspy-ai, so we need to remove "dspy" from the pip_requirements
    pip_requirements = list(filter(lambda x: not x.startswith("dspy=="), pip_requirements))
    conda_env["dependencies"][-1]["pip"] = pip_requirements
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
def log_model(
    model,
    artifact_path,
    conda_env=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Log a Dspy model along with metadata to MLflow.

    This method saves a Dspy model along with metadata such as model signature and conda
    environments to MLflow.

    Args:
        model: an instance of `dspy.module`. The Dspy model to be saved.
        artifact_path: the run-relative path to which to log model artifacts.
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
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.dspy,
        model=model,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
    )
