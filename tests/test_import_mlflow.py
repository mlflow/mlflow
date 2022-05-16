import os
import sys
import subprocess
import uuid

import pytest


@pytest.mark.skipif(os.name == "nt", reason="Shell script in this test doesn't work on Windows")
def test_mlflow_can_be_imported_without_any_extra_dependencies(tmpdir):
    """
    Ensures that mlflow can be imported without any extra dependencies, such as scikit-learn.
    """
    cwd = os.getcwd()
    try:
        os.chdir(tmpdir)
        code_template = """
set -ex

python -m venv .venv
source .venv/bin/activate
pip install {mlflow_dir}
pip list

python -c '
# Make sure extra dependencies are not installed
for module in ["sklearn", "xgboost", "pyspark"]:
    try:
        __import__(module)
        raise Exception(f"{{module}} should not be installed")
    except ImportError:
        pass

import mlflow
'
"""
        env_location = tmpdir.join(uuid.uuid4().hex).strpath
        python_version = ".".join(map(str, sys.version_info[:2]))
        code = code_template.format(
            env_location=env_location, python_version=python_version, mlflow_dir=cwd
        )
        process = subprocess.run(["bash", "-c", code])
        assert process.returncode == 0
    finally:
        os.chdir(cwd)


def test_importing_mlflow_does_not_create_mlruns_directory(tmpdir):
    """
    Ensures that importing mlflow does not create an `mlruns` directory to prevent issues
    such as https://github.com/mlflow/mlflow/issues/3400
    """
    cwd = os.getcwd()
    try:
        os.chdir(tmpdir)
        process = subprocess.run(["python", "-c", "import mlflow"])
        assert process.returncode == 0
        assert "mlruns" not in os.listdir()
    finally:
        os.chdir(cwd)
