import subprocess
import sys


def test_request_utils_does_not_import_mlflow(tmp_path):
    import mlflow.utils.request_utils

    file_content = f"""
import importlib.util
import os
import sys

file_path = r"{mlflow.utils.request_utils.__file__}"
module_name = "mlflow.utils.request_utils"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

assert "mlflow" not in sys.modules
assert "mlflow.utils.request_utils" in sys.modules
"""

    test_file = tmp_path.joinpath("test_request_utils_does_not_import_mlflow.py")
    test_file.write_text(file_content)

    subprocess.run([sys.executable, str(test_file)], check=True)
