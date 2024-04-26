"""
The ``mlflow.prophet`` module provides an API for logging and loading Prophet models.
This module exports univariate Prophet models in the following flavors:

Prophet (native) format
    This is the main flavor that can be accessed with Prophet APIs.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and for batch auditing
    of historical forecasts.

.. _Prophet:
    https://facebook.github.io/prophet/docs/quick_start.html#python-api
"""
import json
import logging
import os
from typing import Any, Dict, Optional

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
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
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "prophet"
_MODEL_BINARY_KEY = "data"
_MODEL_BINARY_FILE_NAME = "model.pr"
_MODEL_TYPE_KEY = "model_type"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at a minimum, contains these requirements.
    """
    # Note: Prophet's whl build process will fail due to missing dependencies, defaulting
    # to setup.py installation process.
    # If a pystan installation error occurs, ensure gcc>=8 is installed in your environment.
    # See: https://gcc.gnu.org/install/
    return [_get_pinned_requirement("prophet")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    pr_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Save a Prophet model to a path on the local file system.

    Args:
        pr_model: Prophet model (an instance of Prophet() forecaster that has been fit
            on a temporal series.
        path: Local path where the serialized model (as JSON) is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            class that describes the model's inputs and outputs. If not specified but an
            ``input_example`` is supplied, a signature will be automatically inferred
            based on the supplied input example and model. To disable automatic signature
            inference when providing an input example, set ``signature`` to ``False``.
            To manually infer a model signature, call
            :py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets
            with valid model inputs, such as a training dataset with the target column
            omitted, and valid model outputs, like model predictions made on the training
            dataset, for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                model = Prophet().fit(df)
                train = model.history
                predictions = model.predict(model.make_future_dataframe(30))
                signature = infer_signature(train, predictions)
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
    """
    import prophet

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if signature is None and input_example is not None:
        wrapped_model = _ProphetModelWrapper(pr_model)
        signature = _infer_signature_from_input_example(input_example, wrapped_model)
    elif signature is False:
        signature = None

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    model_data_path = os.path.join(path, _MODEL_BINARY_FILE_NAME)
    _save_model(pr_model, model_data_path)

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.prophet",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_bin_kwargs,
    )
    flavor_conf = {
        _MODEL_TYPE_KEY: pr_model.__class__.__name__,
        **model_bin_kwargs,
    }
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        prophet_version=prophet.__version__,
        code=code_dir_subpath,
        **flavor_conf,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        default_reqs = None
        if pip_requirements is None:
            # cannot use inferred requirements due to prophet's build process
            # as the package installation of pystan requires Cython to be present
            # in the path. Prophet's installation itself requires imports of
            # existing libraries, preventing the execution of a batched pip install
            # and instead using a a strictly defined list of dependencies.
            # NOTE: if Prophet .whl build architecture is changed, this should be
            # modified to a standard inferred approach.
            default_reqs = get_default_pip_requirements()
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    pr_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Logs a Prophet model as an MLflow artifact for the current run.

    Args:
        pr_model: Prophet model to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: This argument may change or be removed in a
            future release without warning. If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: An instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            class that describes the model's inputs and outputs. If not specified but an
            ``input_example`` is supplied, a signature will be automatically inferred
            based on the supplied input example and model. To disable automatic signature
            inference when providing an input example, set ``signature`` to ``False``.
            To manually infer a model signature, call
            :py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets
            with valid model inputs, such as a training dataset with the target column
            omitted, and valid model outputs, like model predictions made on the training
            dataset, for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                model = Prophet().fit(df)
                train = model.history
                predictions = model.predict(model.make_future_dataframe(30))
                signature = infer_signature(train, predictions)

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version
            to finish being created and is in ``READY`` status.
            By default, the function waits for five minutes.
            Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.prophet,
        registered_model_name=registered_model_name,
        pr_model=pr_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
    )


def _save_model(model, path):
    from prophet.serialize import model_to_json

    model_ser = model_to_json(model)
    with open(path, "w") as f:
        json.dump(model_ser, f)


def _load_model(path):
    from prophet.serialize import model_from_json

    with open(path) as f:
        model = json.load(f)
    return model_from_json(model)


def _load_pyfunc(path):
    """
    Loads PyFunc implementation for Prophet. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``prophet`` flavor.
    """
    return _ProphetModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """
    Load a Prophet model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A Prophet model instance
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    pr_model_path = os.path.join(
        local_model_path, flavor_conf.get(_MODEL_BINARY_KEY, _MODEL_BINARY_FILE_NAME)
    )

    return _load_model(pr_model_path)


class _ProphetModelWrapper:
    def __init__(self, pr_model):
        self.pr_model = pr_model

    def predict(self, dataframe, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        return self.pr_model.predict(dataframe)
