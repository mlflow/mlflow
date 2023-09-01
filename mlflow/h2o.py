"""
The ``mlflow.h2o`` module provides an API for logging and loading H2O models. This module exports
H2O models with the following flavors:

H20 (native) format
    This is the main flavor that can be loaded back into H2O.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import os
import warnings
from typing import Any, Dict, Optional

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
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
from mlflow.utils.file_utils import (
    write_to,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "h2o"


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("h2o")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    h2o_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    settings=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Save an H2O model to a path on the local file system.

    :param h2o_model: H2O model to be saved.
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
    """
    import h2o

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    model_data_subpath = "model.h2o"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if signature is None and input_example is not None:
        wrapped_model = _H2OModelWrapper(h2o_model)
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

    # Save h2o-model
    if hasattr(h2o, "download_model"):
        h2o_save_location = h2o.download_model(model=h2o_model, path=model_data_path)
    else:
        warnings.warn(
            "If your cluster is remote, H2O may not store the model correctly. "
            "Please upgrade H2O version to a newer version"
        )
        h2o_save_location = h2o.save_model(model=h2o_model, path=model_data_path, force=True)
    model_file = os.path.basename(h2o_save_location)

    # Save h2o-settings
    if settings is None:
        settings = {}
    settings["full_file"] = h2o_save_location
    settings["model_file"] = model_file
    settings["model_dir"] = model_data_path
    with open(os.path.join(model_data_path, "h2o.yaml"), "w") as settings_file:
        yaml.safe_dump(settings, stream=settings_file)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.h2o",
        data=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME, h2o_version=h2o.__version__, data=model_data_subpath, code=code_dir_subpath
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
    h2o_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Log an H2O model as an MLflow artifact for the current run.

    :param h2o_model: H2O model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: {{ signature }}
    :param input_example: {{ input_example }}
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param kwargs: kwargs to pass to ``h2o.save_model`` method.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.h2o,
        registered_model_name=registered_model_name,
        h2o_model=h2o_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


def _load_model(path, init=False):
    import h2o

    path = os.path.abspath(path)
    with open(os.path.join(path, "h2o.yaml")) as f:
        params = yaml.safe_load(f.read())
    if init:
        h2o.init(**(params["init"] if "init" in params else {}))
        h2o.no_progress()

    model_path = os.path.join(path, params["model_file"])
    if hasattr(h2o, "upload_model"):
        model = h2o.upload_model(model_path)
    else:
        warnings.warn(
            "If your cluster is remote, H2O may not load the model correctly. "
            "Please upgrade H2O version to a newer version"
        )
        model = h2o.load_model(model_path)

    return model


class _H2OModelWrapper:
    def __init__(self, h2o_model):
        self.h2o_model = h2o_model

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
        import h2o

        predicted = self.h2o_model.predict(h2o.H2OFrame(dataframe)).as_data_frame()
        predicted.index = dataframe.index
        return predicted


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the ``h2o`` flavor.
    """
    return _H2OModelWrapper(_load_model(path, init=True))


def load_model(model_uri, dst_path=None):
    """
    Load an H2O model from a local file (if ``run_id`` is ``None``) or a run.
    This function expects there is an H2O instance initialised with ``h2o.init``.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: An `H2OEstimator model object
             <http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html#models>`_.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    # Flavor configurations for models saved in MLflow version <= 0.8.0 may not contain a
    # `data` key; in this case, we assume the model artifact path to be `model.h2o`
    h2o_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.h2o"))
    return _load_model(path=h2o_model_file_path)
