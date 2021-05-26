import os
import pytest

from mlflow.utils.environment import _mlflow_conda_env


@pytest.fixture
def conda_env_path(tmpdir):
    return os.path.join(tmpdir.strpath, "conda_env.yaml")


def test_mlflow_conda_env_returns_none_when_output_path_is_specified(conda_env_path):
    env_creation_output = _mlflow_conda_env(
        path=conda_env_path, additional_pip_deps=["pip-dep-1", "pip-dep2==0.1.0"],
    )

    assert env_creation_output is None


def test_mlflow_conda_env_returns_expected_env_dict_when_output_path_is_not_specified():
    pip_deps = ["pip-dep-1=0.0.1", "pip-dep-2"]
    env = _mlflow_conda_env(path=None, additional_pip_deps=pip_deps)

    for pip_dep in pip_deps:
        assert pip_dep in env["dependencies"][-1]["pip"]


def test_mlflow_conda_env_includes_pip_dependencies_but_pip_is_not_specified():
    additional_pip_deps = ["pip-dep==0.0.1"]
    env = _mlflow_conda_env(path=None, additional_pip_deps=additional_pip_deps)
    assert "pip" in env["dependencies"]


@pytest.mark.parametrize("pip_version", [None, "20.0.02"])
def test_mlflow_conda_env_includes_pip_dependencies_and_pip_is_specified(pip_version):
    additional_pip_deps = ["pip-dep==0.0.1"]
    env = _mlflow_conda_env(
        path=None, additional_pip_deps=additional_pip_deps, pip_version=pip_version
    )
    if pip_version:
        assert f"pip=={pip_version}" in env["dependencies"]
    else:
        assert "pip" in env["dependencies"]
    assert "pip-dep==0.0.1" in env["dependencies"][-1]["pip"]
