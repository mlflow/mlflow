import os
import sys
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, RESOURCE_ALREADY_EXISTS
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.file_utils import _copy_file_or_tree

FLAVOR_CONFIG_CODE = "code"


def _get_flavor_configuration(model_path, flavor_name):
    """
    Obtains the configuration for the specified flavor from the specified
    MLflow model path. If the model does not contain the specified flavor,
    an exception will be thrown.

    :param model_path: The path to the root directory of the MLflow model for which to load
                       the specified flavor configuration.
    :param flavor_name: The name of the flavor configuration to load.
    :return: The flavor configuration as a dictionary.
    """
    model_configuration_path = os.path.join(model_path, MLMODEL_FILE_NAME)
    if not os.path.exists(model_configuration_path):
        raise MlflowException(
            f'Could not find an "{MLMODEL_FILE_NAME}" configuration file at "{model_path}"',
            RESOURCE_DOES_NOT_EXIST,
        )

    model_conf = Model.load(model_configuration_path)
    if flavor_name not in model_conf.flavors:
        raise MlflowException(
            f'Model does not have the "{flavor_name}" flavor',
            RESOURCE_DOES_NOT_EXIST,
        )
    conf = model_conf.flavors[flavor_name]
    return conf


def _get_flavor_configuration_from_uri(model_uri, flavor_name, logger):
    """
    Obtains the configuration for the specified flavor from the specified
    MLflow model uri. If the model does not contain the specified flavor,
    an exception will be thrown.

    :param model_uri: The path to the root directory of the MLflow model for which to load
                       the specified flavor configuration.
    :param flavor_name: The name of the flavor configuration to load.
    :param logger: The local flavor's logger to report the resolved path of the model uri.
    :return: The flavor configuration as a dictionary.
    """
    try:
        resolved_uri = model_uri
        if RunsArtifactRepository.is_runs_uri(model_uri):
            resolved_uri = RunsArtifactRepository.get_underlying_uri(model_uri)
            logger.info("'%s' resolved as '%s'", model_uri, resolved_uri)
        elif ModelsArtifactRepository.is_models_uri(model_uri):
            resolved_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
            logger.info("'%s' resolved as '%s'", model_uri, resolved_uri)

        ml_model_file = _download_artifact_from_uri(
            artifact_uri=append_to_uri_path(resolved_uri, MLMODEL_FILE_NAME)
        )
    except Exception as ex:
        raise MlflowException(
            f'Failed to download an "{MLMODEL_FILE_NAME}" model file from "{model_uri}"',
            RESOURCE_DOES_NOT_EXIST,
        ) from ex
    return _get_flavor_configuration_from_ml_model_file(ml_model_file, flavor_name)


def _get_flavor_configuration_from_ml_model_file(ml_model_file, flavor_name):
    model_conf = Model.load(ml_model_file)
    if flavor_name not in model_conf.flavors:
        raise MlflowException(
            f'Model does not have the "{flavor_name}" flavor',
            RESOURCE_DOES_NOT_EXIST,
        )
    return model_conf.flavors[flavor_name]


def _validate_code_paths(code_paths):
    if code_paths is not None:
        if not isinstance(code_paths, list):
            raise TypeError("Argument code_paths should be a list, not {}".format(type(code_paths)))


def _validate_and_copy_code_paths(code_paths, path, default_subpath="code"):
    """
    Validates that a code path is a valid list and copies the code paths to a directory. This
    can later be used to log custom code as an artifact.

    :param code_paths: A list of files or directories containing code that should be logged
    as artifacts
    :param path: The local model path.
    :param default_subpath: The default directory name used to store code artifacts.
    """
    _validate_code_paths(code_paths)
    if code_paths is not None:
        code_dir_subpath = default_subpath
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir=code_dir_subpath)
    else:
        code_dir_subpath = None
    return code_dir_subpath


def _add_code_to_system_path(code_path):
    sys.path = [code_path] + sys.path
    # Delete cached modules so they will get reloaded anew from the correct code path
    # Otherwise python will use the cached modules
    modules = [
        p.stem
        for p in Path(code_path).rglob("*.py")
        if p.is_file() and p.name != "__init__.py" and p.name != "__main__.py"
    ]
    for module in modules:
        sys.modules.pop(module, None)


def _validate_and_prepare_target_save_path(path):
    if os.path.exists(path) and any(os.scandir(path)):
        raise MlflowException(
            message=f"Path '{path}' already exists and is not empty",
            error_code=RESOURCE_ALREADY_EXISTS,
        )

    os.makedirs(path, exist_ok=True)


def _add_code_from_conf_to_system_path(local_path, conf, code_key=FLAVOR_CONFIG_CODE):
    """
    Checks if any code_paths were logged with the model in the flavor conf and prepends
    the directory to the system path.

    :param local_path: The local path containing model artifacts.
    :param conf: The flavor-specific conf that should contain the FLAVOR_CONFIG_CODE
    key, which specifies the directory containing custom code logged as artifacts.
    :param code_key: The key used by the flavor to indicate custom code artifacts.
    By default this is FLAVOR_CONFIG_CODE.
    """
    assert isinstance(conf, dict), "`conf` argument must be a dict."
    if code_key in conf and conf[code_key]:
        code_path = os.path.join(local_path, conf[code_key])
        _add_code_to_system_path(code_path)
