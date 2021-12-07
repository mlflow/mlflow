import os

import pytest
import textwrap

from mlflow.exceptions import ExecutionException
from mlflow.projects import _project_spec
from tests.projects.utils import load_project


def test_project_get_entry_point():
    """Test that `Project` correctly parses entry point information from an MLproject file."""
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
    """Test that `Project` can run Python & bash scripts directly as entry points"""
    project = load_project()
    entry_point = project.get_entry_point("my_script.py")
    assert entry_point.name == "my_script.py"
    assert entry_point.command == "python my_script.py"
    assert entry_point.parameters == {}
    entry_point = project.get_entry_point("my_script.sh")
    assert entry_point.name == "my_script.sh"
    assert entry_point.command == "%s my_script.sh" % os.environ.get("SHELL", "bash")
    assert entry_point.parameters == {}
    with pytest.raises(ExecutionException, match="Could not find my_program.scala"):
        project.get_entry_point("my_program.scala")


@pytest.mark.parametrize(
    "mlproject, conda_env_path, conda_env_contents, mlproject_path",
    [
        (None, None, "", None),
        ("key: value", "conda.yaml", "hi", "MLproject"),
        ("conda_env: some-env.yaml", "some-env.yaml", "hi", "mlproject"),
    ],
)
def test_load_project(tmpdir, mlproject, conda_env_path, conda_env_contents, mlproject_path):
    """
    Test that we can load a project with various combinations of an MLproject / conda.yaml file
    :param mlproject: Contents of MLproject file. If None, no MLproject file will be written
    :param conda_env_path: Path to conda environment file. If None, no conda environment file will
                           be written.
    :param conda_env_contents: Contents of conda environment file (written if conda_env_path is
                               not None)
    """
    if mlproject:
        tmpdir.join(mlproject_path).write(mlproject)
    if conda_env_path:
        tmpdir.join(conda_env_path).write(conda_env_contents)
    project = _project_spec.load_project(tmpdir.strpath)
    assert project._entry_points == {}
    expected_env_path = (
        os.path.abspath(os.path.join(tmpdir.strpath, conda_env_path)) if conda_env_path else None
    )
    assert project.conda_env_path == expected_env_path
    if conda_env_path:
        assert open(project.conda_env_path).read() == conda_env_contents


def test_load_docker_project(tmpdir):
    tmpdir.join("MLproject").write(
        textwrap.dedent(
            """
    docker_env:
        image: some-image
    """
        )
    )
    project = _project_spec.load_project(tmpdir.strpath)
    assert project._entry_points == {}
    assert project.conda_env_path is None
    assert project.docker_env.get("image") == "some-image"


@pytest.mark.parametrize(
    "invalid_project_contents, expected_error_msg",
    [
        (
            textwrap.dedent(
                """
    docker_env:
        image: some-image
    conda_env: some-file.yaml
    """
            ),
            "cannot contain both a docker and conda env",
        ),
        (
            textwrap.dedent(
                """
    docker_env:
        not-image-attribute: blah
    """
            ),
            "no image attribute found",
        ),
    ],
)
def test_load_invalid_project(tmpdir, invalid_project_contents, expected_error_msg):
    tmpdir.join("MLproject").write(invalid_project_contents)
    with pytest.raises(ExecutionException, match=expected_error_msg) as e:
        _project_spec.load_project(tmpdir.strpath)
    assert expected_error_msg in str(e.value)
