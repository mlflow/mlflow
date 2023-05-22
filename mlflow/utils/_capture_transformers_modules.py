"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import importlib
import json
import sys
import types

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
        import transformers

        self.original_tf_available = transformers.utils.import_utils._tf_available
        self.original_torch_available = transformers.utils.import_utils._torch_available
        self.original_torch_fx_available = transformers.utils.import_utils._torch_fx_available
        if self.module_to_throw == "tensorflow":
            transformers.utils.import_utils._tf_available = False
        elif self.module_to_throw == "torch":
            transformers.utils.import_utils._torch_available = False
            transformers.utils.import_utils._torch_fx_available = False

        # Reload transformers to make sure the patch is applied
        importlib.reload(transformers)
        import transformers  # pylint: disable=W0404

        for module_name in transformers.__all__:
            module = getattr(transformers, module_name)
            if isinstance(module, types.ModuleType):
                try:
                    importlib.reload(module)
                except ImportError:
                    pass

        return super().__enter__()

    def __exit__(self, *_, **__):
        # Revert the patches
        import transformers

        transformers.utils.import_utils._tf_available = self.original_tf_available
        transformers.utils.import_utils._torch_available = self.original_torch_available
        transformers.utils.import_utils._torch_fx_available = self.original_torch_fx_available
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
            f"This script is only applicable to '{mlflow.transformers.FLAVOR_NAME}' flavor, ",
            "if you're applying other flavors, please use _capture_modules script.",
        )

    if module_to_throw:
        cap_cm = _CaptureImportedModulesForHF(module_to_throw)
    else:
        cap_cm = _CaptureImportedModules()
    store_imported_module(cap_cm, model_path, flavor, output_file)


if __name__ == "__main__":
    main()
