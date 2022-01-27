import importlib
import re
import yaml

from packaging.version import Version, InvalidVersion
from pkg_resources import resource_filename

import mlflow
from mlflow.utils.databricks_utils import is_in_databricks_runtime


# A map FLAVOR_NAME -> a tuple of (dependent_module_name, key_in_module_version_info_dict)
FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY = {
    "fastai": ("fastai", "fastai"),
    "gluon": ("mxnet", "gluon"),
    "keras": ("keras", "keras"),
    "lightgbm": ("lightgbm", "lightgbm"),
    "statsmodels": ("statsmodels", "statsmodels"),
    "tensorflow": ("tensorflow", "tensorflow"),
    "xgboost": ("xgboost", "xgboost"),
    "sklearn": ("sklearn", "sklearn"),
    "pytorch": ("pytorch_lightning", "pytorch-lightning"),
    "pyspark.ml": ("pyspark", "spark"),
}


def _check_version_in_range(ver, min_ver, max_ver):
    return Version(min_ver) <= Version(ver) <= Version(max_ver)


def _violates_pep_440(ver):
    try:
        _ = Version(ver)
        return False
    except InvalidVersion:
        return True


def _is_pre_or_dev_release(ver):
    v = Version(ver)
    return v.is_devrelease or v.is_prerelease


def _strip_dev_version_suffix(version):
    return re.sub(r"(\.?)dev.*", "", version)


def _load_version_file_as_dict():
    version_file_path = resource_filename(mlflow.__name__, "ml-package-versions.yml")
    with open(version_file_path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


_module_version_info_dict = _load_version_file_as_dict()


def get_min_max_version_and_pip_release(module_key):
    min_version = _module_version_info_dict[module_key]["autologging"]["minimum"]
    max_version = _module_version_info_dict[module_key]["autologging"]["maximum"]
    pip_release = _module_version_info_dict[module_key]["package_info"]["pip_release"]
    return min_version, max_version, pip_release


def is_flavor_supported_for_associated_package_versions(flavor_name):
    """
    :return: True if the specified flavor is supported for the currently-installed versions of its
             associated packages
    """
    module_name, module_key = FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY[flavor_name]
    actual_version = importlib.import_module(module_name).__version__

    # In Databricks, treat 'pyspark 3.x.y.dev0' as 'pyspark 3.x.y'
    if module_name == "pyspark" and is_in_databricks_runtime():
        actual_version = _strip_dev_version_suffix(actual_version)

    if _violates_pep_440(actual_version) or _is_pre_or_dev_release(actual_version):
        return False
    min_version, max_version, _ = get_min_max_version_and_pip_release(module_key)
    return _check_version_in_range(actual_version, min_version, max_version)
