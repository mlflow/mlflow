"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import argparse
import builtins
import functools
import importlib
import json
import os
import sys

import mlflow
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.pyfunc import MAIN
from mlflow.utils._spark_utils import _prepare_subprocess_environ_for_creating_local_spark_session
from mlflow.utils.file_utils import write_to
from mlflow.utils.requirements_utils import DATABRICKS_MODULES_TO_PACKAGES


def _get_top_level_module(full_module_name):
    return full_module_name.split(".")[0]


def _get_second_level_module(full_module_name):
    return ".".join(full_module_name.split(".")[:2])


class _CaptureImportedModules:
    """
    A context manager to capture imported modules by temporarily applying a patch to
    `builtins.__import__` and `importlib.import_module`.
    """

    def __init__(self):
        self.imported_modules = set()
        self.original_import = None
        self.original_import_module = None

    def _wrap_import(self, original):
        # pylint: disable=redefined-builtin
        @functools.wraps(original)
        def wrapper(name, globals=None, locals=None, fromlist=(), level=0):
            is_absolute_import = level == 0
            if is_absolute_import:
                self._record_imported_module(name)
            return original(name, globals, locals, fromlist, level)

        return wrapper

    def _wrap_import_module(self, original):
        @functools.wraps(original)
        def wrapper(name, *args, **kwargs):
            self._record_imported_module(name)
            return original(name, *args, **kwargs)

        return wrapper

    def _record_imported_module(self, full_module_name):
        # If the module is an internal module (prefixed by "_") or is the "databricks"
        # module, which is populated by many different packages, don't record it (specific
        # module imports within the databricks namespace are still recorded and mapped to
        # their corresponding packages)
        if full_module_name.startswith("_") or full_module_name == "databricks":
            return

        top_level_module = _get_top_level_module(full_module_name)
        second_level_module = _get_second_level_module(full_module_name)

        if top_level_module == "databricks":
            # Multiple packages populate the `databricks` module namespace on Databricks;
            # to avoid bundling extraneous Databricks packages into model dependencies, we
            # scope each module to its relevant package
            if second_level_module in DATABRICKS_MODULES_TO_PACKAGES:
                self.imported_modules.add(second_level_module)
                return

            for databricks_module in DATABRICKS_MODULES_TO_PACKAGES:
                if full_module_name.startswith(databricks_module):
                    self.imported_modules.add(databricks_module)
                    return

        self.imported_modules.add(top_level_module)

    def __enter__(self):
        # Patch `builtins.__import__` and `importlib.import_module`
        self.original_import = builtins.__import__
        self.original_import_module = importlib.import_module
        builtins.__import__ = self._wrap_import(self.original_import)
        importlib.import_module = self._wrap_import_module(self.original_import_module)
        return self

    def __exit__(self, *_, **__):
        # Revert the patches
        builtins.__import__ = self.original_import
        importlib.import_module = self.original_import_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--sys-path", required=True)
    parser.add_argument("--module-to-throw", required=False)
    return parser.parse_args()


def store_imported_modules(cap_cm, model_path, flavor, output_file):
    # If `model_path` refers to an MLflow model directory, load the model using
    # `mlflow.pyfunc.load_model`
    if os.path.isdir(model_path) and MLMODEL_FILE_NAME in os.listdir(model_path):
        mlflow_model = Model.load(model_path)
        pyfunc_conf = mlflow_model.flavors.get(mlflow.pyfunc.FLAVOR_NAME)
        input_example = mlflow_model.load_input_example(model_path)
        loader_module = importlib.import_module(pyfunc_conf[MAIN])
        original = loader_module._load_pyfunc

        @functools.wraps(original)
        def _load_pyfunc_patch(*args, **kwargs):
            with cap_cm:
                model = original(*args, **kwargs)
                if input_example is not None:
                    model.predict(input_example)
                return model

        loader_module._load_pyfunc = _load_pyfunc_patch
        try:
            mlflow.pyfunc.load_model(model_path)
        finally:
            loader_module._load_pyfunc = original
    # Otherwise, load the model using `mlflow.<flavor>._load_pyfunc`.
    # For models that don't contain pyfunc flavor (e.g. scikit-learn estimator
    # that doesn't implement a `predict` method),
    # we need to directly pass a model data path to this script.
    else:
        with cap_cm:
            importlib.import_module(f"mlflow.{flavor}")._load_pyfunc(model_path)

    # Store the imported modules in `output_file`
    write_to(output_file, "\n".join(cap_cm.imported_modules))


def main():
    args = parse_args()
    model_path = args.model_path
    flavor = args.flavor
    output_file = args.output_file
    # Mirror `sys.path` of the parent process
    sys.path = json.loads(args.sys_path)

    if flavor == mlflow.spark.FLAVOR_NAME:
        # Create a local spark environment within the subprocess
        from mlflow.utils._spark_utils import _create_local_spark_session_for_loading_spark_model

        _prepare_subprocess_environ_for_creating_local_spark_session()
        _create_local_spark_session_for_loading_spark_model()

    cap_cm = _CaptureImportedModules()
    store_imported_modules(cap_cm, model_path, flavor, output_file)

    # Clean up a spark session created by `mlflow.spark._load_pyfunc`
    if flavor == mlflow.spark.FLAVOR_NAME:
        from mlflow.utils._spark_utils import _get_active_spark_session

        spark = _get_active_spark_session()
        if spark:
            try:
                spark.stop()
            except Exception:
                # Swallow unexpected exceptions
                pass


if __name__ == "__main__":
    main()
