# Copyright 2018 Databricks, Inc.
import importlib.metadata
import re

VERSION = "2.21.4.dev0"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))


# A flag to indicate whether MLflow is installed in a skinny environment.
# This is False when only mlflow-trace is installed and used for deciding
# which modules to import and what tests to run.
IS_MLFLOW_SKINNY_INSTALLED = False
try:
    importlib.metadata.version("mlflow-skinny")
    IS_MLFLOW_SKINNY_INSTALLED = True
except importlib.metadata.PackageNotFoundError:
    pass
