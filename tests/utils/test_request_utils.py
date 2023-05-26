import os
import subprocess


def test_request_utils_does_not_import_mlflow():
    file_content = """
import importlib.util
import os
import sys

file_path = os.path.join(os.path.dirname(__file__), "../../mlflow/utils/request_utils.py")
module_name = "mlflow.utils.request_utils"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

assert "mlflow" not in sys.modules
assert "mlflow.utils.request_utils" in sys.modules
"""

    test_file_name = os.path.join(
        os.path.dirname(__file__), "test_request_utils_does_not_import_mlflow.py"
    )

    try:
        with open(test_file_name, "w") as f:
            f.write(file_content)
        subprocess.run(["python", test_file_name], check=True)
    finally:
        os.remove(test_file_name)
