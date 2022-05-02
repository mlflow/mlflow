import os
import sys

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, RESOURCE_ALREADY_EXISTS
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
            'Could not find an "{model_file}" configuration file at "{model_path}"'.format(
                model_file=MLMODEL_FILE_NAME, model_path=model_path
            ),
            RESOURCE_DOES_NOT_EXIST,
        )

    model_conf = Model.load(model_configuration_path)
    if flavor_name not in model_conf.flavors:
        raise MlflowException(
            'Model does not have the "{flavor_name}" flavor'.format(flavor_name=flavor_name),
            RESOURCE_DOES_NOT_EXIST,
        )
    conf = model_conf.flavors[flavor_name]
    return conf


def _get_flavor_configuration_from_uri(model_uri, flavor_name):
    """
    Obtains the configuration for the specified flavor from the specified
    MLflow model uri. If the model does not contain the specified flavor,
    an exception will be thrown.

    :param model_uri: The path to the root directory of the MLflow model for which to load
                       the specified flavor configuration.
    :param flavor_name: The name of the flavor configuration to load.
    :return: The flavor configuration as a dictionary.
    """
    try:
        ml_model_file = _download_artifact_from_uri(
            artifact_uri=append_to_uri_path(model_uri, MLMODEL_FILE_NAME)
        )
    except Exception as ex:
        raise MlflowException(
            'Failed to download an "{model_file}" model file from "{model_uri}": {ex}'.format(
                model_file=MLMODEL_FILE_NAME, model_uri=model_uri, ex=ex
            ),
            RESOURCE_DOES_NOT_EXIST,
        )
    model_conf = Model.load(ml_model_file)
    if flavor_name not in model_conf.flavors:
        raise MlflowException(
            'Model does not have the "{flavor_name}" flavor'.format(flavor_name=flavor_name),
            RESOURCE_DOES_NOT_EXIST,
        )
    return model_conf.flavors[flavor_name]


def _get_code_dirs(src_code_path, dst_code_path=None):
    """
    Obtains the names of the subdirectories contained under the specified source code
    path and joins them with the specified destination code path.
    :param src_code_path: The path of the source code directory for which to list subdirectories.
    :param dst_code_path: The destination directory path to which subdirectory names should be
                          joined.
    """
    if not dst_code_path:
        dst_code_path = src_code_path
    return [
        (os.path.join(dst_code_path, x))
        for x in os.listdir(src_code_path)
        if os.path.isdir(os.path.join(src_code_path, x)) and not x == "__pycache__"
    ]


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
    sys.path = [code_path] + _get_code_dirs(code_path) + sys.path


def _validate_and_prepare_target_save_path(path):
    if os.path.exists(path) and any(os.scandir(path)):
        raise MlflowException(
            message="Path '{}' already exists and is not empty".format(path),
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
    if code_key in conf and conf[code_key]:
        code_path = os.path.join(local_path, conf[code_key])
        _add_code_to_system_path(code_path)
