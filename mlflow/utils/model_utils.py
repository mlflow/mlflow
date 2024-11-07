import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import get_parent_module
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.requirements_utils import _capture_imported_modules
from mlflow.utils.uri import append_to_uri_path

FLAVOR_CONFIG_CODE = "code"


def _get_all_flavor_configurations(model_path):
    """Obtains all the flavor configurations from the specified MLflow model path.

    Args:
        model_path: The path to the root directory of the MLflow model for which to load
            the specified flavor configuration.

    Returns:
        The dictionary contains all flavor configurations with flavor name as key.

    """

    return Model.load(model_path).flavors


def _get_flavor_configuration(model_path, flavor_name):
    """Obtains the configuration for the specified flavor from the specified
    MLflow model path. If the model does not contain the specified flavor,
    an exception will be thrown.

    Args:
        model_path: The path to the root directory of the MLflow model for which to load
            the specified flavor configuration.
        flavor_name: The name of the flavor configuration to load.

    Returns:
        The flavor configuration as a dictionary.

    """
    try:
        return Model.load(model_path).flavors[flavor_name]
    except KeyError as ex:
        raise MlflowException(
            f'Model does not have the "{flavor_name}" flavor', RESOURCE_DOES_NOT_EXIST
        ) from ex


def _get_flavor_configuration_from_uri(model_uri, flavor_name, logger):
    """Obtains the configuration for the specified flavor from the specified
    MLflow model uri. If the model does not contain the specified flavor,
    an exception will be thrown.

    Args:
        model_uri: The path to the root directory of the MLflow model for which to load
            the specified flavor configuration.
        flavor_name: The name of the flavor configuration to load.
        logger: The local flavor's logger to report the resolved path of the model uri.

    Returns:
        The flavor configuration as a dictionary.
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
    """Validates that a code path is a valid list and copies the code paths to a directory. This
    can later be used to log custom code as an artifact.

    Args:
        code_paths: A list of files or directories containing code that should be logged
            as artifacts.
        path: The local model path.
        default_subpath: The default directory name used to store code artifacts.
    """
    _validate_code_paths(code_paths)
    if code_paths is not None:
        code_dir_subpath = default_subpath
        for code_path in code_paths:
            try:
                _copy_file_or_tree(src=code_path, dst=path, dst_dir=code_dir_subpath)
            except OSError as e:
                # A common error is code-paths includes Databricks Notebook. We include it in error
                # message when running in Databricks, but not in other envs tp avoid confusion.
                example = ", such as Databricks Notebooks" if is_in_databricks_runtime() else ""
                raise MlflowException(
                    message=(
                        f"Failed to copy the specified code path '{code_path}' into the model "
                        "artifacts. It appears that your code path includes file(s) that cannot "
                        f"be copied{example}. Please specify a code path that does not include "
                        "such files and try again.",
                    ),
                    error_code=INVALID_PARAMETER_VALUE,
                ) from e
    else:
        code_dir_subpath = None
    return code_dir_subpath


def _infer_and_copy_code_paths(flavor, path, default_subpath="code"):
    # Capture all imported modules with full module name during loading model.
    modules = _capture_imported_modules(path, flavor, record_full_module=True)

    all_modules = set(modules)

    for module in modules:
        parent_module = module
        while "." in parent_module:
            parent_module = get_parent_module(parent_module)
            all_modules.add(parent_module)

    # Generate code_paths set from the imported modules full name list.
    # It only picks necessary files, because:
    #  1. Reduce risk of logging files containing user credentials to MLflow
    #     artifact repository.
    #  2. In databricks runtime, notebook files might exist under a code_path directory,
    #     if logging the whole directory to MLflow artifact repository, these
    #     notebook files are not accessible and trigger exceptions. On the other
    #     hand, these notebook files are not used as code_paths modules because
    #     code in notebook files are loaded into python `__main__` module.
    code_paths = set()
    for full_module_name in all_modules:
        relative_path_str = full_module_name.replace(".", os.sep)
        relative_path = Path(relative_path_str)
        if relative_path.is_dir():
            init_file_path = relative_path / "__init__.py"
            if init_file_path.exists():
                code_paths.add(init_file_path)

        py_module_path = Path(relative_path_str + ".py")
        if py_module_path.is_file():
            code_paths.add(py_module_path)

    if code_paths:
        for code_path in code_paths:
            src_dir_path = code_path.parent
            src_file_name = code_path.name
            dest_dir_path = Path(path) / default_subpath / src_dir_path
            dest_file_path = dest_dir_path / src_file_name
            dest_dir_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(code_path, dest_file_path)
        return default_subpath

    return None


def _validate_infer_and_copy_code_paths(
    code_paths, path, infer_code_paths, flavor, default_subpath="code"
):
    if infer_code_paths:
        if code_paths:
            raise MlflowException(
                "If 'infer_code_path' is set to True, 'code_paths' param cannot be set."
            )
        return _infer_and_copy_code_paths(flavor, path, default_subpath)
    else:
        return _validate_and_copy_code_paths(code_paths, path, default_subpath)


def _validate_path_exists(path, name):
    if path and not os.path.exists(path):
        raise MlflowException(
            message=(
                f"Failed to copy the specified {name} path '{path}' into the model "
                f"artifacts. The specified {name }path does not exist. Please specify a valid "
                f"{name} path and try again."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_and_copy_file_to_directory(file_path: str, dir_path: str, name: str):
    """Copies the file at file_path to the directory at dir_path.

    Args:
        file_path: A file that should be logged as an artifact.
        dir_path: The path of the directory to save the file to.
        name: The name for the kind of file being copied.
    """
    _validate_path_exists(file_path, name)
    try:
        _copy_file_or_tree(src=file_path, dst=dir_path)
    except OSError as e:
        # A common error is code-paths includes Databricks Notebook. We include it in error
        # message when running in Databricks, but not in other envs tp avoid confusion.
        example = ", such as Databricks Notebooks" if is_in_databricks_runtime() else ""
        raise MlflowException(
            message=(
                f"Failed to copy the specified code path '{file_path}' into the model "
                "artifacts. It appears that your code path includes file(s) that cannot "
                f"be copied{example}. Please specify a code path that does not include "
                "such files and try again.",
            ),
            error_code=INVALID_PARAMETER_VALUE,
        ) from e


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
    """Checks if any code_paths were logged with the model in the flavor conf and prepends
    the directory to the system path.

    Args:
        local_path: The local path containing model artifacts.
        conf: The flavor-specific conf that should contain the FLAVOR_CONFIG_CODE
            key, which specifies the directory containing custom code logged as artifacts.
        code_key: The key used by the flavor to indicate custom code artifacts.
            By default this is FLAVOR_CONFIG_CODE.
    """
    assert isinstance(conf, dict), "`conf` argument must be a dict."

    if code_key in conf and conf[code_key]:
        code_path = os.path.join(local_path, conf[code_key])
        _add_code_to_system_path(code_path)


def _validate_onnx_session_options(onnx_session_options):
    """Validates that the specified onnx_session_options dict is valid.

    Args:
        onnx_session_options: The onnx_session_options dict to validate.
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
            elif key == "execution_mode" and value.upper() not in [
                "PARALLEL",
                "SEQUENTIAL",
            ]:
                raise ValueError(
                    f"Value for key {key} in onnx_session_options should be "
                    f"'parallel' or 'sequential', not {value}"
                )
            elif key == "graph_optimization_level" and value not in [0, 1, 2, 99]:
                raise ValueError(
                    f"Value for key {key} in onnx_session_options should be 0, 1, 2, or 99, "
                    f"not {value}"
                )
            elif key in ["intra_op_num_threads", "intra_op_num_threads"] and value < 0:
                raise ValueError(
                    f"Value for key {key} in onnx_session_options should be >= 0, not {value}"
                )


def _get_overridden_pyfunc_model_config(
    pyfunc_config: dict[str, Any], load_config: dict[str, Any], logger
) -> dict[str, Any]:
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


def _validate_and_get_model_config_from_file(model_config):
    model_config = os.path.abspath(model_config)
    if os.path.exists(model_config):
        with open(model_config) as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise MlflowException(
                    f"The provided `model_config` file '{model_config}' is not a valid YAML "
                    f"file: {e}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
    else:
        raise MlflowException(
            "An invalid `model_config` file was passed. The provided `model_config` "
            f"file '{model_config}'is not a valid file path.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_pyfunc_model_config(model_config):
    """
    Validates the values passes in the model_config section. There are no typing
    restrictions but we require them being JSON-serializable.
    """

    if not model_config:
        return

    if isinstance(model_config, Path):
        _validate_and_get_model_config_from_file(os.fspath(model_config))
    elif isinstance(model_config, str):
        _validate_and_get_model_config_from_file(model_config)
    elif isinstance(model_config, dict) and all(isinstance(key, str) for key in model_config):
        try:
            json.dumps(model_config)
        except (TypeError, OverflowError):
            raise MlflowException(
                "Values in the provided ``model_config`` are of an unsupported type. Only "
                "JSON-serializable data types can be provided as values.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    else:
        raise MlflowException(
            "An invalid ``model_config`` structure was passed. ``model_config`` must be a "
            "valid file path or of type ``dict`` with string keys.",
            error_code=INVALID_PARAMETER_VALUE,
        )


RECORD_ENV_VAR_ALLOWLIST = {
    # api key related
    "API_KEY",
    "API_TOKEN",
    # databricks auth related
    "DATABRICKS_HOST",
    "DATABRICKS_USERNAME",
    "DATABRICKS_PASSWORD",
    "DATABRICKS_TOKEN",
    "DATABRICKS_INSECURE",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "_DATABRICKS_WORKSPACE_HOST",
    "_DATABRICKS_WORKSPACE_ID",
}


@contextlib.contextmanager
def env_var_tracker():
    """
    Context manager for temporarily tracking environment variables accessed.
    It tracks environment variables accessed during the context manager's lifetime.
    """
    from mlflow.environment_variables import MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING

    tracked_env_names = set()

    if MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING.get():
        original_getitem = os._Environ.__getitem__
        original_get = os._Environ.get

        def updated_get_item(self, key):
            result = original_getitem(self, key)
            tracked_env_names.add(key)
            return result

        def updated_get(self, key, *args, **kwargs):
            if key in self:
                tracked_env_names.add(key)
            return original_get(self, key, *args, **kwargs)

        try:
            os._Environ.__getitem__ = updated_get_item
            os._Environ.get = updated_get
            yield tracked_env_names
        finally:
            os._Environ.__getitem__ = original_getitem
            os._Environ.get = original_get
    else:
        yield tracked_env_names
