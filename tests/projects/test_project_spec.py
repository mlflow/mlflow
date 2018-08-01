import os

import pytest

from mlflow.projects.utils import ExecutionException
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


# Parameters: conda env contents, MLproject contents, conda env path, expected results in project
def test_load_project_partial_fields(tmpdir):
    # Test that we can load a project from a directory without an MLproject file
    project = _project_spec.load_project(tmpdir.strpath)
    assert project.entry_points == {}
    assert project.conda_env_path is None
    assert project.load_conda_env() == ""
    # Test that we can detect a default conda.yaml file to use if not explicitly specified in an
    # MLproject file
    for i, mlproject_exists in enumerate([True, False]):
        project_dir = tmpdir.mkdir("conda-yaml-%s" % i)
        if mlproject_exists:
            project_dir.join("MLproject").write("key: value")
        project_dir.join("conda.yaml").write("hi")
        project = _project_spec.load_project(project_dir.strpath)
        assert project.conda_env_path == os.path.abspath(
            os.path.join(project_dir.strpath, "conda.yaml"))
        assert project.load_conda_env() == "hi"
    # Test that we can detect a conda environment file specified in an MLproject file
    conda_yaml_path = "some-env.yml"
    project_dir = tmpdir.mkdir("configure-conda-yaml")
    project_dir.join("MLproject").write("conda_env: %s" % conda_yaml_path)
    project_dir.join(conda_yaml_path).write("hi")
    project = _project_spec.load_project(project_dir.strpath)
    assert project.conda_env_path == os.path.abspath(
        os.path.join(project_dir.strpath, conda_yaml_path))
    assert project.load_conda_env() == "hi"
