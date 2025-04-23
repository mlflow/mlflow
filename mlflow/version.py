# Copyright 2018 Databricks, Inc.
import importlib.metadata
import re

VERSION = "2.21.4.dev0"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))


# A flag to indicate whether the environment only has the tracing SDK
# installed, or includes the full MLflow or mlflow-skinny package.
# This is used to determine whether to import modules that require
# dependencies that are not included in the tracing SDK.
IS_TRACING_SDK_ONLY = True
for pkg in ["mlflow", "mlflow-skinny"]:
    try:
        importlib.metadata.version(pkg)

        IS_TRACING_SDK_ONLY = False
    except importlib.metadata.PackageNotFoundError:
        pass
