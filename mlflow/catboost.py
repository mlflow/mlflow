"""
The ``mlflow.catboost`` module provides an API for logging and loading CatBoost models.
This module exports CatBoost models with the following flavors:

CatBoost (native) format
    This is the main flavor that can be loaded back into CatBoost.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _CatBoost:
    https://catboost.ai/docs/concepts/python-reference_catboost.html
.. _CatBoost.save_model:
    https://catboost.ai/docs/concepts/python-reference_catboost_save_model.html
.. _CatBoostClassifier:
    https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
.. _CatBoostRanker:
    https://catboost.ai/docs/concepts/python-reference_catboostranker.html
.. _CatBoostRegressor:
    https://catboost.ai/docs/concepts/python-reference_catboostregressor.html
"""

import contextlib
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
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "catboost"
_MODEL_TYPE_KEY = "model_type"
_SAVE_FORMAT_KEY = "save_format"
_MODEL_BINARY_KEY = "data"
_MODEL_BINARY_FILE_NAME = "model.cb"


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("catboost")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    cb_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Save a CatBoost model to a path on the local file system.

    :param cb_model: CatBoost model (an instance of `CatBoost`_, `CatBoostClassifier`_,
                    `CatBoostRanker`_, or `CatBoostRegressor`_) to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param signature: {{ signature }}
    :param input_example: {{ input_example }}
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param kwargs: kwargs to pass to `CatBoost.save_model`_ method.
    """
    import catboost as cb

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if signature is None and input_example is not None:
        wrapped_model = _CatboostModelWrapper(cb_model)
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
    cb_model.save_model(model_data_path, **kwargs)

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.catboost",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_bin_kwargs,
    )

    flavor_conf = {
        _MODEL_TYPE_KEY: cb_model.__class__.__name__,
        _SAVE_FORMAT_KEY: kwargs.get("format", "cbm"),
        **model_bin_kwargs,
    }
    mlflow_model.add_flavor(
        FLAVOR_NAME, catboost_version=cb.__version__, code=code_dir_subpath, **flavor_conf
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


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    cb_model,
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
    **kwargs,
):
    """
    Log a CatBoost model as an MLflow artifact for the current run.

    :param cb_model: CatBoost model (an instance of `CatBoost`_, `CatBoostClassifier`_,
                    `CatBoostRanker`_, or `CatBoostRegressor`_) to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param signature: {{ signature }}
    :param input_example: {{ input_example }}
    :param await_registration_for: Number of seconds to wait for the model version to finish
                        being created and is in ``READY`` status. By default, the function
                        waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param kwargs: kwargs to pass to `CatBoost.save_model`_ method.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.catboost,
        registered_model_name=registered_model_name,
        cb_model=cb_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


def _init_model(model_type):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

    model_types = {c.__name__: c for c in [CatBoost, CatBoostClassifier, CatBoostRegressor]}

    with contextlib.suppress(ImportError):
        from catboost import CatBoostRanker

        model_types[CatBoostRanker.__name__] = CatBoostRanker

    if model_type not in model_types:
        raise TypeError(
            f"Invalid model type: '{model_type}'. Must be one of {list(model_types.keys())}"
        )

    return model_types[model_type]()


def _load_model(path, model_type, save_format):
    model = _init_model(model_type)
    model.load_model(os.path.abspath(path), save_format)
    return model


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the ``catboost`` flavor.
    """
    flavor_conf = _get_flavor_configuration(
        model_path=os.path.dirname(path), flavor_name=FLAVOR_NAME
    )
    return _CatboostModelWrapper(
        _load_model(path, flavor_conf.get(_MODEL_TYPE_KEY), flavor_conf.get(_SAVE_FORMAT_KEY))
    )


def load_model(model_uri, dst_path=None):
    """
    Load a CatBoost model from a local file or a run.

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

    :return: A CatBoost model (an instance of `CatBoost`_, `CatBoostClassifier`_, `CatBoostRanker`_,
             or `CatBoostRegressor`_)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    cb_model_file_path = os.path.join(
        local_model_path, flavor_conf.get(_MODEL_BINARY_KEY, _MODEL_BINARY_FILE_NAME)
    )
    return _load_model(
        cb_model_file_path, flavor_conf.get(_MODEL_TYPE_KEY), flavor_conf.get(_SAVE_FORMAT_KEY)
    )


class _CatboostModelWrapper:
    def __init__(self, cb_model):
        self.cb_model = cb_model

    def predict(
        self, dataframe, params: Optional[Dict[str, Any]] = None
    ):  # pylint: disable=unused-argument
        """
        :param dataframe: Model input data.
        :param params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        :return: Model predictions.
        """
        return self.cb_model.predict(dataframe)


# TODO: Support autologging
