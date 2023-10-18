import json
import os
import sys
from typing import Any, Dict

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.uri import append_to_uri_path

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
    return model_conf.flavors[flavor_name]


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

        try:
            ml_model_file = _download_artifact_from_uri(
                artifact_uri=append_to_uri_path(resolved_uri, MLMODEL_FILE_NAME)
            )
        except Exception:
            logger.debug(
                f'Failed to download an "{MLMODEL_FILE_NAME}" model file from '
                f"resolved URI {resolved_uri}. "
                f"Falling back to downloading from original model URI {model_uri}",
                exc_info=True,
            )
            ml_model_file = get_artifact_repository(artifact_uri=model_uri).download_artifacts(
                artifact_path=MLMODEL_FILE_NAME
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
            raise TypeError(f"Argument code_paths should be a list, not {type(code_paths)}")


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


def _validate_onnx_session_options(onnx_session_options):
    """
    Validates that the specified onnx_session_options dict is valid.

    :param ort_session_options: The onnx_session_options dict to validate.
    """
    import onnxruntime as ort

    if onnx_session_options is not None:
        if not isinstance(onnx_session_options, dict):
            raise TypeError(
                f"Argument onnx_session_options should be a dict, not {type(onnx_session_options)}"
            )
        for key, value in onnx_session_options.items():
            if key != "extra_session_config" and not hasattr(ort.SessionOptions, key):
                raise ValueError(
                    f"Key {key} in onnx_session_options is not a valid "
                    "ONNX Runtime session options key"
                )
            elif key == "extra_session_config" and not isinstance(value, dict):
                raise TypeError(
                    f"Value for key {key} in onnx_session_options should be a dict, "
                    "not {type(value)}"
                )
            elif key == "execution_mode" and value.upper() not in ["PARALLEL", "SEQUENTIAL"]:
                raise ValueError(
                    f"Value for key {key} in onnx_session_options should be "
                    "'parallel' or 'sequential', not {value}"
                )
            elif key == "graph_optimization_level" and value not in [0, 1, 2, 99]:
                raise ValueError(
                    f"Value for key {key} in onnx_session_options should be 0, 1, 2, or 99, "
                    "not {value}"
                )
            elif key in ["intra_op_num_threads", "intra_op_num_threads"] and value < 0:
                raise ValueError(
                    f"Value for key {key} in onnx_session_options should be >= 0, not {value}"
                )


def _get_overridden_pyfunc_model_config(
    pyfunc_config: Dict[str, Any], load_config: Dict[str, Any], logger
) -> Dict[str, Any]:
    """
    Updates the inference configuration according to the model's configuration and the overrides.
    Only arguments already present in the inference configuration can be updated. The environment
    variable ``MLFLOW_PYFUNC_INFERENCE_CONFIG`` can also be used to provide additional inference
    configuration.
    """

    overrides = {}
    if env_overrides := os.getenv("MLFLOW_PYFUNC_INFERENCE_CONFIG"):
        logger.debug(
            "Inference configuration is being loaded from ``MLFLOW_PYFUNC_INFERENCE_CONFIG``"
            " environ."
        )
        overrides.update(dict(json.loads(env_overrides)))

    if load_config:
        overrides.update(load_config)

    if not overrides:
        return pyfunc_config

    if not pyfunc_config:
        logger.warning(
            f"Argument(s) {', '.join(overrides.keys())} were ignored since the model's ``pyfunc``"
            " flavor doesn't accept model configuration. Use ``model_config`` when logging"
            " the model to allow it."
        )

        return None

    valid_keys = set(pyfunc_config.keys()) & set(overrides.keys())
    ignored_keys = set(overrides.keys()) - valid_keys
    allowed_config = {key: overrides[key] for key in valid_keys}
    if ignored_keys:
        logger.warning(
            f"Argument(s) {', '.join(ignored_keys)} were ignored since they are not valid keys in"
            " the corresponding section of the ``pyfunc`` flavor. Use ``model_config`` when"
            " logging the model to include the keys you plan to indicate. Current allowed"
            f" configuration includes {', '.join(pyfunc_config.keys())}"
        )
    pyfunc_config.update(allowed_config)
    return pyfunc_config


def _validate_pyfunc_model_config(model_config):
    """
    Validates the values passes in the model_config section. There are no typing
    restrictions but we require them being JSON-serializable.
    """

    if not model_config:
        return

    if not isinstance(model_config, dict) or not all(isinstance(key, str) for key in model_config):
        raise MlflowException(
            "An invalid ``model_config`` structure was passed. ``model_config`` must be of type "
            "``dict`` with string keys."
        )

    try:
        json.dumps(model_config)
    except (TypeError, OverflowError):
        raise MlflowException(
            "Values in the provided ``model_config`` are of an unsupported type. Only "
            "JSON-serializable data types can be provided as values."
        )
