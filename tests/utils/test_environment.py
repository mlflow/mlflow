import os
import pytest
import yaml

from mlflow.utils.environment import _mlflow_conda_env


@pytest.fixture
def conda_env_path(tmpdir):
    return os.path.join(tmpdir.strpath, "conda_env.yaml")


def test_mlflow_conda_env_returns_env_text_when_output_path_is_specified(conda_env_path):
    env_text = _mlflow_conda_env(
            path=conda_env_path, 
            additional_conda_deps=["conda-dep-1=0.0.1", "conda-dep-2"],
            additional_pip_deps=["pip-dep-1", "pip-dep2==0.1.0"])

    with open(conda_env_path, "r") as f:
        assert env_text == f.read()


def test_mlflow_conda_env_returns_env_text_when_output_path_is_not_specified(conda_env_path):
    conda_deps = ["conda-dep-1=0.0.1", "conda-dep-2"]

    env_text = _mlflow_conda_env(
                path=conda_env_path, 
                additional_conda_deps=conda_deps)

    env_parsed = yaml.safe_load(env_text)
    for conda_dep in conda_deps:
        assert conda_dep in env_parsed["dependencies"]
