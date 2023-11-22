"""
The ``mlflow.paddle`` module provides an API for logging and loading paddle models.
This module exports paddle models with the following flavors:

Paddle (native) format
    This is the main flavor that can be loaded back into paddle.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
    NOTE: The `mlflow.pyfunc` flavor is only added for paddle models that define `predict()`,
    since `predict()` is required for pyfunc model inference.
"""

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
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
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

FLAVOR_NAME = "paddle"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("paddlepaddle", module="paddle")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    pd_model,
    path,
    training=False,
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
    Save a paddle model to a path on the local file system. Produces an MLflow Model
    containing the following flavors:
    
        - :py:mod:`mlflow.paddle`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for paddle models
          that define `predict()`, since `predict()` is required for pyfunc model inference.
    
    Args:
        pd_model: paddle model to be saved.
        path: Local path where the model is to be saved.
        training: Only valid when saving a model trained using the PaddlePaddle high level API.
                  If set to True, the saved model supports both re-training and
                  inference. If set to False, it only supports inference.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
                    containing file dependencies). These files are *prepended* to the system
                    path when the model is loaded.
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.
    
                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    """
    import paddle

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if signature is None and input_example is not None:
        wrapped_model = _PaddleWrapper(pd_model)
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

    model_data_subpath = "model"
    output_path = os.path.join(path, model_data_subpath)

    if isinstance(pd_model, paddle.Model):
        pd_model.save(output_path, training=training)
    else:
        paddle.jit.save(pd_model, output_path)

    # `PyFuncModel` only works for paddle models that define `predict()`.
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.paddle",
        model_path=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        paddle_version=paddle.__version__,
        code=code_dir_subpath,
    )
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


def load_model(model_uri, model=None, dst_path=None, **kwargs):
    """
    Load a paddle model from a local file or a run.
    
    Args:
        model_uri: The location, in URI format, of the MLflow model, for example:
            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``
        model: Required when loading a `paddle.Model` model saved with `training=True`.
        dst_path: The local filesystem path to which to download the model artifact.
                  This directory must already exist. If unspecified, a local output
                  path will be created.
        kwargs: The keyword arguments to pass to `paddle.jit.load`
                or `model.load`.
    
    For more information about supported URI schemes, see
    `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
    artifact-locations>`_.
    
    Returns:
        A paddle model.
    
    .. code-block:: python
        :caption: Example
    
        import mlflow.paddle
    
        pd_model = mlflow.paddle.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2/pd_models")
        # use Pandas DataFrame to make predictions
        np_array = ...
        predictions = pd_model(np_array)
    """
    import paddle

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    pd_model_artifacts_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    if model is None:
        return paddle.jit.load(pd_model_artifacts_path, **kwargs)
    elif not isinstance(model, paddle.Model):
        raise TypeError(f"Invalid object type `{type(model)}` for `model`, must be `paddle.Model`")
    else:
        contains_pdparams = _contains_pdparams(local_model_path)
        if not contains_pdparams:
            raise TypeError(
                "This model can't be loaded via `model.load` because a '.pdparams' file "
                "doesn't exist. Please leave `model` unspecified to load the model via "
                "`paddle.jit.load` or set `training` to True when saving a model."
            )

        model.load(pd_model_artifacts_path, **kwargs)
        return model


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    pd_model,
    artifact_path,
    training=False,
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
    Log a paddle model as an MLflow artifact for the current run. Produces an MLflow Model
    containing the following flavors:
    
        - :py:mod:`mlflow.paddle`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for paddle models
          that define `predict()`, since `predict()` is required for pyfunc model inference.
    
    Args:
        pd_model: paddle model to be saved.
        artifact_path: Run-relative artifact path.
        training: Only valid when saving a model trained using the PaddlePaddle high level API.
                   If set to True, the saved model supports both re-training and
                   inference. If set to False, it only supports inference.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
                    containing file dependencies). These files are *prepended* to the system
                    path when the model is loaded.
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
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.
    
                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    
    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.paddle,
        pd_model=pd_model,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        training=training,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
    )


def _load_pyfunc(path):
    """
    Loads PyFunc implementation. Called by ``pyfunc.load_model``.
    
    Args:
        path: Local filesystem path to the MLflow Model with the ``paddle`` flavor.
    """
    return _PaddleWrapper(load_model(path))


class _PaddleWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, pd_model):
        self.pd_model = pd_model

    def predict(
        self, data, params: Optional[Dict[str, Any]] = None  # pylint: disable=unused-argument
    ):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.
        
                .. Note:: Experimental: This parameter may change or be removed in a future
                           release without warning.
        
        Returns:
            Model predictions.
        """
        import numpy as np
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            inp_data = data.values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            inp_data = data
        elif isinstance(data, (list, dict)):
            raise TypeError(
                "The paddle flavor does not support List or Dict input types. "
                "Please use a pandas.DataFrame or a numpy.ndarray"
            )
        else:
            raise TypeError("Input data should be pandas.DataFrame or numpy.ndarray")
        inp_data = np.squeeze(inp_data)

        self.pd_model.eval()

        predicted = self.pd_model(inp_data)
        return pd.DataFrame(predicted.numpy())


def _contains_pdparams(path):
    file_list = os.listdir(path)
    return any(".pdparams" in file for file in file_list)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_every_n_epoch=1,
    log_models=True,
    disable=False,
    exclusive=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging from PaddlePaddle to MLflow.
    
    Autologging is performed when the `fit` method of `paddle.Model`_ is called.
    
    .. _paddle.Model:
        https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/Model_en.html
    
    Args:
        log_every_n_epoch: If specified, logs metrics once every `n` epochs. By default, metrics
                           are logged after every epoch.
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
                    If ``False``, trained models are not logged.
        disable: If ``True``, disables the PaddlePaddle autologging integration.
                 If ``False``, enables the PaddlePaddle autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                   If ``False``, autologged content is logged to the active fluent run,
                   which may be user-created.
        silent: If ``True``, suppress all event logs and warnings from MLflow during PyTorch
                Lightning autologging. If ``False``, show all events and warnings during
                PaddlePaddle autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
                               new model version of the registered model with this name.
                               The registered model is created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """
    import paddle

    from mlflow.paddle._paddle_autolog import patched_fit

    safe_patch(
        FLAVOR_NAME, paddle.Model, "fit", patched_fit, manage_run=True, extra_tags=extra_tags
    )
