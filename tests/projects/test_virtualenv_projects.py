import os
from unittest import mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.utils.virtualenv import _create_virtualenv
from mlflow.utils.yaml_utils import read_yaml, write_yaml

from tests.projects.utils import (
    TEST_VIRTUALENV_CONDA_PROJECT_DIR,
    TEST_VIRTUALENV_NO_PYTHON_ENV,
    TEST_VIRTUALENV_PROJECT_DIR,
)

spy_on_create_virtualenv = mock.patch(
    "mlflow.projects.backend.local._create_virtualenv", wraps=_create_virtualenv
)


@pytest.fixture(autouse=True, scope="module")
def use_dev_mlflow_for_projects():
    mlflow_root = os.path.dirname(os.path.dirname(mlflow.__file__))

    conda_env = read_yaml(TEST_VIRTUALENV_CONDA_PROJECT_DIR, "conda.yaml")
    conda_pip_dependencies = [
        item for item in conda_env["dependencies"] if isinstance(item, dict) and "pip" in item
    ][0]["pip"]
    if "mlflow" in conda_pip_dependencies:
        conda_pip_dependencies.remove("mlflow")
        conda_pip_dependencies.append(mlflow_root)
    write_yaml(TEST_VIRTUALENV_CONDA_PROJECT_DIR, "conda.yaml", conda_env, overwrite=True)

    for proj_dir in (TEST_VIRTUALENV_PROJECT_DIR, TEST_VIRTUALENV_NO_PYTHON_ENV):
        virtualenv_requirements_path = os.path.join(proj_dir, "requirements.txt")
        with open(virtualenv_requirements_path) as f:
            virtualenv_requirements = f.readlines()

        with open(virtualenv_requirements_path, "w") as f:
            for line in virtualenv_requirements:
                if line.rstrip("\n") != "mlflow":
                    f.write(line)
                else:
                    f.write(mlflow_root)
                    f.write("\n")


@spy_on_create_virtualenv
def test_virtualenv_project_execution_virtualenv(create_virtualenv_spy):
    submitted_run = mlflow.projects.run(
        TEST_VIRTUALENV_PROJECT_DIR, entry_point="test", env_manager="virtualenv"
    )
    submitted_run.wait()
    create_virtualenv_spy.assert_called_once()


@spy_on_create_virtualenv
def test_virtualenv_project_execution_uv(create_virtualenv_spy):
    submitted_run = mlflow.projects.run(
        TEST_VIRTUALENV_PROJECT_DIR, entry_point="test", env_manager="uv"
    )
    submitted_run.wait()
    create_virtualenv_spy.assert_called_once()


@spy_on_create_virtualenv
def test_virtualenv_project_execution_without_env_manager(create_virtualenv_spy):
    # python_env project should be executed using virtualenv without explicitly specifying
    # env_manager="virtualenv"
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_PROJECT_DIR, entry_point="test")
    submitted_run.wait()
    create_virtualenv_spy.assert_called_once()


@spy_on_create_virtualenv
def test_virtualenv_project_execution_no_python_env(create_virtualenv_spy):
    """
    When an MLproject file doesn't contain a `python_env` key but python_env.yaml exists,
    virtualenv should be used as an environment manager.
    """
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_NO_PYTHON_ENV, entry_point="test")
    submitted_run.wait()
    create_virtualenv_spy.assert_called_once()


@spy_on_create_virtualenv
def test_virtualenv_project_execution_local(create_virtualenv_spy):
    submitted_run = mlflow.projects.run(
        TEST_VIRTUALENV_PROJECT_DIR, entry_point="main", env_manager="local"
    )
    submitted_run.wait()
    create_virtualenv_spy.assert_not_called()


@spy_on_create_virtualenv
def test_virtualenv_conda_project_execution(create_virtualenv_spy):
    submitted_run = mlflow.projects.run(
        TEST_VIRTUALENV_CONDA_PROJECT_DIR, entry_point="test", env_manager="virtualenv"
    )
    submitted_run.wait()
    create_virtualenv_spy.assert_called_once()


def test_virtualenv_project_execution_conda():
    with pytest.raises(MlflowException, match="python_env project cannot be executed using conda"):
        mlflow.projects.run(TEST_VIRTUALENV_PROJECT_DIR, env_manager="conda")


@spy_on_create_virtualenv
def test_virtualenv_project_no_env_file(create_virtualenv_spy, tmp_path):
    """
    When neither python_env.yaml nor conda.yaml is present, virtualenv should be used as an
    environment manager.
    """
    ml_project_file = tmp_path.joinpath("MLproject")
    ml_project_file.write_text(
        """
name: test
entry_points:
  main:
    command: |
      python test.py
"""
    )
    tmp_path.joinpath("test.py").write_text(
        """
import os

assert "VIRTUAL_ENV" in os.environ
"""
    )
    mlflow.projects.run(str(tmp_path))
    create_virtualenv_spy.assert_called_once()


@spy_on_create_virtualenv
def test_virtualenv_project_no_mlmodel_file(create_virtualenv_spy, tmp_path):
    tmp_path.joinpath("test.py").write_text(
        """
import os

assert "VIRTUAL_ENV" in os.environ
"""
    )
    mlflow.projects.run(str(tmp_path), entry_point="test.py")
    create_virtualenv_spy.assert_called_once()
