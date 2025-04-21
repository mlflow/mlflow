# Copyright 2018 Databricks, Inc.
import importlib.metadata
import re

VERSION = "2.21.4.dev0"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))


# A flag to indicate whether MLflow (or mlflow-skinny) in installed.
# This is False when only mlflow-trace is installed and used for deciding
# which modules to import and what tests to run.
IS_FULL_MLFLOW_INSTALLED = False
for pkg in ["mlflow", "mlflow-skinny"]:
    try:
        importlib.metadata.version(pkg)

        IS_FULL_MLFLOW_INSTALLED = True
    except importlib.metadata.PackageNotFoundError:
        pass
