"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import json
import os
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.utils._capture_modules import _CaptureImportedModules, parse_args, store_imported_module


class _CaptureImportedModulesForHF(_CaptureImportedModules):
    """
    A context manager to capture imported modules by temporarily applying a patch to
    `builtins.__import__` and `importlib.import_module`.
    Used for 'transformers' flavor only.
    """

    def __init__(self, module_to_throw):
        super().__init__()
        self.module_to_throw = module_to_throw

    def _wrap_package(self, name):
        if name == self.module_to_throw or name.startswith(f"{self.module_to_throw}."):
            raise ImportError(f"Disabled package {name}")

    def _record_imported_module(self, full_module_name):
        self._wrap_package(full_module_name)
        return super()._record_imported_module(full_module_name)

    def __enter__(self):
        # Patch the environment variables to disable module_to_throw
        # https://github.com/huggingface/transformers/blob/3658488ff77ff8d45101293e749263acf437f4d5/src/transformers/utils/import_utils.py#L60-L62
        self.use_tf = os.environ.get("USE_TF")
        self.use_torch = os.environ.get("USE_TORCH")
        if self.module_to_throw == "tensorflow":
            os.environ["USE_TORCH"] = "TRUE"
        elif self.module_to_throw == "torch":
            os.environ["USE_TF"] = "TRUE"

        return super().__enter__()

    def _restore_env_var(self, key, value):
        if key in os.environ:
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value

    def __exit__(self, *_, **__):
        # Revert the patches
        self._restore_env_var("USE_TF", self.use_tf)
        self._restore_env_var("USE_TORCH", self.use_torch)
        super().__exit__()


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
    cap_cm = _CaptureImportedModulesForHF(module_to_throw)
    store_imported_module(cap_cm, model_path, flavor, output_file)


if __name__ == "__main__":
    main()
