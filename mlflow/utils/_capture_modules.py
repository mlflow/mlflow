"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import argparse
import builtins
import functools
import importlib
import os

import mlflow
from mlflow.utils.file_utils import write_to
from mlflow.pyfunc import DATA
from mlflow.models.model import MLMODEL_FILE_NAME


def _get_top_level_module(full_module_name):
    return full_module_name.split(".")[0]


class _CaptureImportedModules:
    """
    A context manager to capture imported modules in `IMPORTED_MODULES` by temporarily
    applying a patch to `builtins.__import__` and `importlib.import_module`.
    """

    def _wrap_import(self, original_import):
        @functools.wraps(original_import)
        def wrapper(name, globals=None, locals=None, fromlist=(), level=0):
            is_absolute_import = level == 0
            if not name.startswith("_") and is_absolute_import:
                top_level_module = _get_top_level_module(name)
                self.imported_modules.add(top_level_module)
            return original_import(name, globals, locals, fromlist, level)

        return wrapper

    def _wrap_import_module(self, original_import_module):
        @functools.wraps(original_import_module)
        def wrapper(name, *args, **kwargs):
            if not name.startswith("_"):
                top_level_module = _get_top_level_module(name)
                self.imported_modules.add(top_level_module)
            return original_import_module(name, *args, **kwargs)

        return wrapper

    def __enter__(self):
        self.imported_modules = set()
        self.original_import = builtins.__import__
        self.original_import_module = importlib.import_module
        builtins.__import__ = self._wrap_import(self.original_import)
        importlib.import_module = self._wrap_import_module(self.original_import_module)
        return self

    def __exit__(self, *_, **__):
        builtins.__import__ = self.original_import
        importlib.import_module = self.original_import_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path
    flavor = args.flavor
    output_file = args.output_file

    # If `model_path` is a directory containing an `MLmodel` file, attempt to get the model data
    # path from it.
    if os.path.isdir(model_path) and MLMODEL_FILE_NAME in os.listdir(model_path):
        conf = mlflow.models.Model.load(model_path).flavors.get(flavor)
        data_path = os.path.join(model_path, conf[DATA]) if (DATA in conf) else model_path
    # Otherwise, assume `model_path` is a file or directory storing the model data.
    else:
        data_path = model_path

    # Load the model and capture modules imported during the loading procedure.
    flavor_module = getattr(mlflow, flavor)
    with _CaptureImportedModules() as cap:
        flavor_module._load_pyfunc(data_path)

    # Store the imported modules in `output_file`.
    write_to(output_file, "\n".join(cap.imported_modules))


if __name__ == "__main__":
    main()
