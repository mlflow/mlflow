import os

import pytest

from mlflow.exceptions import ExecutionException
from mlflow.projects import _project_spec
from tests.projects.utils import load_project


def test_project_get_entry_point():
    """ Test that `Project` correctly parses entry point information from an MLproject file."""
    project = load_project()
    entry_point = project.get_entry_point("greeter")
    assert entry_point.name == "greeter"
    assert entry_point.command == "python greeter.py {greeting} {name}"
    # Validate parameters
    assert set(entry_point.parameters.keys()) == set(["name", "greeting"])
    name_param = entry_point.parameters["name"]
    assert name_param.type == "string"
    assert name_param.default is None
    greeting_param = entry_point.parameters["greeting"]
    assert greeting_param.type == "string"
    assert greeting_param.default == "hi"


def test_project_get_unspecified_entry_point():
    """ Test that `Project` can run Python & bash scripts directly as entry points """
    project = load_project()
    entry_point = project.get_entry_point("my_script.py")
    assert entry_point.name == "my_script.py"
    assert entry_point.command == "python my_script.py"
    assert entry_point.parameters == {}
    entry_point = project.get_entry_point("my_script.sh")
    assert entry_point.name == "my_script.sh"
    assert entry_point.command == "%s my_script.sh" % os.environ.get("SHELL", "bash")
    assert entry_point.parameters == {}
    with pytest.raises(ExecutionException):
        project.get_entry_point("my_program.scala")


@pytest.mark.parametrize("mlproject, conda_env_path, conda_env_contents", [
    (None, None, ""),
    ("key: value", "conda.yaml", "hi"),
    ("conda_env: some-env.yaml", "some-env.yaml", "hi")
])
def test_load_project(tmpdir, mlproject, conda_env_path, conda_env_contents):
    """
    Test that we can load a project with various combinations of an MLproject / conda.yaml file
    :param mlproject: Contents of MLproject file. If None, no MLproject file will be written
    :param conda_env_path: Path to conda environment file. If None, no conda environment file will
                           be written.
    :param conda_env_contents: Contents of conda environment file (written if conda_env_path is
                               not None)
    """
    if mlproject:
        tmpdir.join("MLproject").write(mlproject)
    if conda_env_path:
        tmpdir.join(conda_env_path).write(conda_env_contents)
    project = _project_spec.load_project(tmpdir.strpath)
    assert project._entry_points == {}
    expected_env_path = \
        os.path.abspath(os.path.join(tmpdir.strpath, conda_env_path)) if conda_env_path else None
    assert project.conda_env_path == expected_env_path
    if conda_env_path:
        assert open(project.conda_env_path).read() == conda_env_contents
