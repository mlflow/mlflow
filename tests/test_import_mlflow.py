import os
import sys
import subprocess
import uuid

import pytest


@pytest.mark.skipif(os.name == "nt", reason="This test fails on Windows")
def test_mlflow_can_be_imported_without_any_extra_dependencies(tmpdir):
    """
    Ensures that mlflow can be imported without any extra dependencies such as scikit-learn.
    """
    cwd = os.getcwd()
    try:
        os.chdir(tmpdir)
        code_template = """
set -ex

export PATH="$HOME/miniconda3/bin:$PATH"
conda create --yes --prefix {env_location} python={python_version}
source deactivate && source activate {env_location}
conda info

pip install {mlflow_dir}
pip list

python -c '
try:
    import sklearn
    raise Exception("scikit-learn should not be installed")
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
