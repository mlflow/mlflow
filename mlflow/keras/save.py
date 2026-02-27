"""Functions for saving Keras models to MLflow."""

import importlib
import logging
import os
import shutil
import tempfile
from typing import Any

import keras
import yaml

import mlflow
from mlflow import pyfunc
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
from mlflow.types.schema import TensorSpec
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
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "keras"

_MODEL_SAVE_PATH = "model"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"

_logger = logging.getLogger(__name__)


_MODEL_DATA_PATH = "data"


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by Keras flavor. Calls to
        `save_model()` and `log_model()` produce a pip environment that, at minimum, contains these
        requirements.
    """
    return [_get_pinned_requirement("keras")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to `save_model()` and
        `log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _export_keras_model(model, path, signature):
    if signature is None:
        raise ValueError(
            "`signature` cannot be None when `save_exported_model=True` for "
            "`mlflow.keras.save_model()` method."
        )
    try:
        import tensorflow as tf
    except ImportError:
        raise MlflowException(
            "`tensorflow` must be installed if you want to export a Keras 3 model, please "
            "install `tensorflow` by `pip install tensorflow`, or set `save_exported_model=False`."
        )
    input_schema = signature.inputs.to_dict()
    export_signature = []
    for schema in input_schema:
        dtype = schema["tensor-spec"]["dtype"]
        shape = schema["tensor-spec"]["shape"]
        # Replace -1 with None in shape.
        new_shape = [size if size != -1 else None for size in shape]
        export_signature.append(tf.TensorSpec(shape=new_shape, dtype=dtype))

    export_archive = keras.export.ExportArchive()
    export_archive.track(model)
    export_archive.add_endpoint(
        name="serve",
        fn=model.call,
        input_signature=export_signature,
    )
    export_archive.write_out(path)


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    path,
    save_exported_model=False,
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
    Save a Keras model along with metadata.

    This method saves a Keras model along with metadata such as model signature and conda
    environments to local file system. This method is called inside `mlflow.keras.log_model()`.

    Args:
        model: an instance of `keras.Model`. The Keras model to be saved.
        path: local path where the MLflow model is to be saved.
        save_exported_model: If True, save Keras model in exported model
            format, otherwise save in `.keras` format. For more information, please
            refer to https://keras.io/guides/serialization_and_saving/.
        conda_env: {{ conda_env }}
        mlflow_model: an instance of `mlflow.models.Model`, defaults to None. MLflow model
            configuration to which to add the Keras model metadata. If None, a blank instance will
            be created.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        save_model_kwargs: A dict of kwargs to pass to `keras.Model.save`
            method.
        metadata: {{ metadata }}

    .. code-block:: python
        :caption: Example

        import keras
        import mlflow

        model = keras.Sequential(
            [
                keras.Input([28, 28, 3]),
                keras.layers.Flatten(),
                keras.layers.Dense(2),
            ]
        )
        with mlflow.start_run() as run:
            mlflow.keras.save_model(model, "./model")
    """

    import keras

    if signature is None:
        _logger.warning("You are saving a Keras model without specifying model signature.")
    else:
        num_inputs = len(signature.inputs.inputs)
        if num_inputs == 0:
            raise MlflowException(
                "The model signature's input schema must contain at least one field.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        for field in signature.inputs.inputs:
            if not isinstance(field, TensorSpec):
                raise MlflowException(
                    "All fields in the model signature's input schema must be of type TensorSpec.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if field.shape[0] != -1:
                raise MlflowException(
                    "All input schema' first dimension should be -1, which represents the dynamic "
                    "batch dimension.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    save_model_kwargs = save_model_kwargs or {}

    data_subpath = _MODEL_DATA_PATH
    # Construct new data folder in existing path.
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)

    model_subpath = os.path.join(data_subpath, _MODEL_SAVE_PATH)
    keras_module = importlib.import_module("keras")

    # Save keras module spec to path/data/keras_module.txt
    with open(os.path.join(data_path, _KERAS_MODULE_SPEC_PATH), "w") as f:
        f.write(keras_module.__name__)

    if save_exported_model:
        model_path = os.path.join(path, model_subpath)
        _export_keras_model(model, model_path, signature)
    else:
        # Save path requires ".keras" suffix.
        file_extension = ".keras"
        model_path = os.path.join(path, model_subpath) + file_extension
        if path.startswith("/dbfs/"):
            # The Databricks Filesystem uses a FUSE implementation that does not support
            # random writes. It causes an error.
            with tempfile.NamedTemporaryFile(suffix=".keras") as f:
                model.save(f.name, **save_model_kwargs)
                f.flush()  # force flush the data
                shutil.copy2(src=f.name, dst=model_path)
        else:
            model.save(model_path, **save_model_kwargs)

    flavor_options = {
        "data": data_subpath,
        "keras_version": keras.__version__,
        "keras_backend": keras.backend.backend(),
        "save_exported_model": save_exported_model,
    }

    # Add flavor info to `mlflow_model`.
    mlflow_model.add_flavor(FLAVOR_NAME, **flavor_options)

    # Add loader_module, data and env data to `mlflow_model`.
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.keras",
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

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary.
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`.
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    artifact_path: str | None = None,
    save_exported_model=False,
    conda_env=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    save_model_kwargs=None,
    metadata=None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
):
    """
    Log a Keras model along with metadata to MLflow.

    This method saves a Keras model along with metadata such as model signature and conda
    environments to MLflow.

    Args:
        model: an instance of `keras.Model`. The Keras model to be saved.
        artifact_path: Deprecated. Use `name` instead.
        save_exported_model: defaults to False. If True, save Keras model in exported
            model format, otherwise save in `.keras` format. For more information, please
            refer to `Keras doc <https://keras.io/guides/serialization_and_saving/>`_.
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
        save_model_kwargs: defaults to None. A dict of kwargs to pass to
            `keras.Model.save` method.
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel
            file.
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}

    .. code-block:: python
        :caption: Example

        import keras
        import mlflow

        model = keras.Sequential(
            [
                keras.Input([28, 28, 3]),
                keras.layers.Flatten(),
                keras.layers.Dense(2),
            ]
        )
        with mlflow.start_run() as run:
            mlflow.keras.log_model(model, name="model")
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=mlflow.keras,
        model=model,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        save_model_kwargs=save_model_kwargs,
        save_exported_model=save_exported_model,
        metadata=metadata,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
    )
