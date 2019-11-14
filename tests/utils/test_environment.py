import os
import pytest

from mlflow.utils.environment import _mlflow_conda_env


@pytest.fixture
def conda_env_path(tmpdir):
    return os.path.join(tmpdir.strpath, "conda_env.yaml")


def test_mlflow_conda_env_returns_none_when_output_path_is_specified(conda_env_path):
    env_creation_output = _mlflow_conda_env(
            path=conda_env_path,
            additional_conda_deps=["conda-dep-1=0.0.1", "conda-dep-2"],
            additional_pip_deps=["pip-dep-1", "pip-dep2==0.1.0"])

    assert env_creation_output is None


def test_mlflow_conda_env_returns_expected_env_dict_when_output_path_is_not_specified():
    conda_deps = ["conda-dep-1=0.0.1", "conda-dep-2"]
    env = _mlflow_conda_env(
                path=None,
                additional_conda_deps=conda_deps)

    for conda_dep in conda_deps:
        assert conda_dep in env["dependencies"]
