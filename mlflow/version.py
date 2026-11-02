# Copyright 2018 Databricks, Inc.
import importlib.metadata
import re

VERSION = "3.18.0"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))


def _is_package_installed(package_name: str) -> bool:
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# A flag to indicate whether the environment only has the tracing SDK
# installed, or includes the full MLflow or mlflow-skinny package.
# This is used to determine whether to import modules that require
# dependencies that are not included in the tracing SDK.
IS_TRACING_SDK_ONLY = not any(_is_package_installed(pkg) for pkg in ["mlflow", "mlflow-skinny"])

# A flag to indicate whether the environment only has the mlflow-skinny package
IS_MLFLOW_SKINNY = _is_package_installed("mlflow-skinny") and not _is_package_installed("mlflow")

IS_FULL_MLFLOW = _is_package_installed("mlflow")

# A flag to indicate whether the flavor model logging APIs can be imported. They depend on
# numpy and pandas, which mlflow-skinny and the tracing SDK don't require.
HAS_MODEL_LOGGING_DEPS = not IS_TRACING_SDK_ONLY and all(
    _is_package_installed(pkg) for pkg in ["numpy", "pandas"]
)
