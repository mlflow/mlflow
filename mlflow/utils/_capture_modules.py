"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import argparse
import builtins
import functools

import mlflow


IMPORTED_MODULES = set()


def _wrap_import(original_import):
    """
    Wraps `builtins.__import__` to capture imported modules and store them in `IMPORTED_MODULES`.
    """

    # pylint: disable=redefined-builtin
    @functools.wraps(original_import)
    def wrapper(name, globals=None, locals=None, fromlist=(), level=0):
        is_absolute_import = level == 0
        if not name.startswith("_") and is_absolute_import:
            IMPORTED_MODULES.add(name.split(".")[0])

        original_result = original_import(name, globals, locals, fromlist, level)
        return original_result

    return wrapper


def _wrap_load_pyfunc(original_load_pyfunc):
    """
    Wraps `mlflow.*._load_pyfunc` to capture modules imported during the loading procedure
    by temporarily applying a patch to `builtins.__import__`.
    """

    @functools.wraps(original_load_pyfunc)
    def wrapper(*args, **kwargs):
        original_import = builtins.__import__
        builtins.__import__ = _wrap_import(original_import)
        result = original_load_pyfunc(*args, **kwargs)
        builtins.__import__ = original_import
        return result

    return wrapper


def _is_mlflow_model(model_path):
    try:
        mlflow.models.Model.load(model_path)
        return True
    except Exception:
        return False


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

    flavor_module = getattr(mlflow, flavor)
    wrapped_load_pyfunc = _wrap_load_pyfunc(flavor_module._load_pyfunc)
    # Load the model and capture modules imported during the loading procedure.
    if _is_mlflow_model(model_path):
        flavor_module._load_pyfunc = wrapped_load_pyfunc
        mlflow.pyfunc.load_model(model_path)
    else:
        wrapped_load_pyfunc(model_path)

    # Store the imported modules in `output_file`.
    with open(output_file, "w") as f:
        f.write("\n".join(IMPORTED_MODULES))


if __name__ == "__main__":
    main()
