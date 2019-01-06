import os

import mlflow.pyfunc
from mlflow.pyfunc import ENV, FLAVOR_NAME
from mlflow.utils import PYTHON_VERSION, get_major_minor_py_version
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking.utils import _get_model_log_dir


def _get_code_dirs(src_code_path, dst_code_path=None):
    if not dst_code_path:
        dst_code_path = src_code_path
    return [(os.path.join(dst_code_path, x))
            for x in os.listdir(src_code_path) if not x.endswith(".py") and not
            x.endswith(".pyc") and not x == "__pycache__"]


def _warn_potentially_incompatible_py_version_if_necessary(model_py_version):
    if model_py_version is None:
        mlflow.pyfunc._logger.warning(
            "The specified model does not have a specified Python version. It may be"
            " incompatible with the version of Python that is currently running: Python %s",
            PYTHON_VERSION)
    elif get_major_minor_py_version(model_py_version) != get_major_minor_py_version(PYTHON_VERSION):
        mlflow.pyfunc._logger.warning(
            "The version of Python that the model was saved in, `Python %s`, differs"
            " from the version of Python that is currently running, `Python %s`,"
            " and may be incompatible",
            model_py_version, PYTHON_VERSION)


def _load_model_env(path, run_id=None):
    """
        Get ENV file string from a model configuration stored in Python Function format.
        Returned value is a model-relative path to a Conda Environment file,
        or None if none was specified at model save time
    """
    if run_id is not None:
        path = _get_model_log_dir(path, run_id)
    return _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME).get(ENV, None)
