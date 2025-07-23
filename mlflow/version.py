# Copyright 2018 Databricks, Inc.
import importlib.metadata
import re

VERSION = "3.1.4"


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
