"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import json
import os
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils._capture_modules import (
    _CaptureImportedModules,
    parse_args,
    store_imported_modules,
)


class _CaptureImportedModulesForHF(_CaptureImportedModules):
    """
    A context manager to capture imported modules by temporarily applying a patch to
    `builtins.__import__` and `importlib.import_module`.
    Used for 'transformers' flavor only.
    """

    def __init__(self, module_to_throw):
        super().__init__()
        self.module_to_throw = module_to_throw

    def _record_imported_module(self, full_module_name):
        if full_module_name == self.module_to_throw or full_module_name.startswith(
            f"{self.module_to_throw}."
        ):
            raise ImportError(f"Disabled package {full_module_name}")
        return super()._record_imported_module(full_module_name)


def main():
    args = parse_args()
    model_path = args.model_path
    flavor = args.flavor
    output_file = args.output_file
    module_to_throw = args.module_to_throw
    # Mirror `sys.path` of the parent process
    sys.path = json.loads(args.sys_path)

    if flavor != mlflow.transformers.FLAVOR_NAME:
        raise MlflowException(
            f"This script is only applicable to '{mlflow.transformers.FLAVOR_NAME}' flavor, "
            "if you're applying other flavors, please use _capture_modules script.",
        )

    if module_to_throw == "":
        raise MlflowException("Please specify the module to throw.")
    elif module_to_throw == "tensorflow":
        if os.environ.get("USE_TORCH", None) != "TRUE":
            raise MlflowException(
                "The environment variable USE_TORCH has to be set to TRUE to disable Tensorflow.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    elif module_to_throw == "torch":
        if os.environ.get("USE_TF", None) != "TRUE":
            raise MlflowException(
                "The environment variable USE_TF has to be set to TRUE to disable Pytorch.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    cap_cm = _CaptureImportedModulesForHF(module_to_throw)
    store_imported_modules(cap_cm, model_path, flavor, output_file)


if __name__ == "__main__":
    main()
