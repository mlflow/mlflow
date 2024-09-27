import os
import textwrap

import pytest

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
    assert set(entry_point.parameters.keys()) == {"name", "greeting"}
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
    assert entry_point.command == "{} my_script.sh".format(os.environ.get("SHELL", "bash"))
    assert entry_point.parameters == {}
    with pytest.raises(ExecutionException, match="Could not find my_program.scala"):
        project.get_entry_point("my_program.scala")


@pytest.mark.parametrize(
    ("mlproject", "conda_env_path", "conda_env_contents", "mlproject_path"),
    [
        (None, None, "", None),
        ("key: value", "conda.yaml", "hi", "MLproject"),
        ("conda_env: some-env.yaml", "some-env.yaml", "hi", "mlproject"),
    ],
)
def test_load_project(tmp_path, mlproject, conda_env_path, conda_env_contents, mlproject_path):  # noqa: D417
    """
    Test that we can load a project with various combinations of an MLproject / conda.yaml file

    Args:
        mlproject: Contents of MLproject file. If None, no MLproject file will be written.
        conda_env_path: Path to conda environment file. If None, no conda environment file will
            be written.
        conda_env_contents: Contents of conda environment file (written if conda_env_path is
            not None).
    """
    if mlproject:
        tmp_path.joinpath(mlproject_path).write_text(mlproject)
    if conda_env_path:
        tmp_path.joinpath(conda_env_path).write_text(conda_env_contents)
    project = _project_spec.load_project(str(tmp_path))
    assert project._entry_points == {}
    expected_env_path = str(tmp_path.joinpath(conda_env_path)) if conda_env_path else None
    assert project.env_config_path == expected_env_path
    if conda_env_path:
        with open(project.env_config_path) as f:
            assert f.read() == conda_env_contents


def test_load_docker_project(tmp_path):
    tmp_path.joinpath("MLproject").write_text(
        textwrap.dedent(
            """
    docker_env:
        image: some-image
    """
        )
    )
    project = _project_spec.load_project(str(tmp_path))
    assert project._entry_points == {}
    assert project.env_config_path is None
    assert project.docker_env.get("image") == "some-image"


def test_load_virtualenv_project(tmp_path):
    tmp_path.joinpath("MLproject").write_text("python_env: python_env.yaml")
    python_env = tmp_path.joinpath("python_env.yaml")
    python_env.write_text("python: 3.8.15")
    project = _project_spec.load_project(tmp_path)
    assert project._entry_points == {}
    assert python_env.samefile(project.env_config_path)


@pytest.mark.parametrize(
    ("invalid_project_contents", "expected_error_msg"),
    [
        (
            textwrap.dedent(
                """
    docker_env:
        image: some-image
    conda_env: some-file.yaml
    """
            ),
            "cannot contain multiple environment fields",
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
def test_load_invalid_project(tmp_path, invalid_project_contents, expected_error_msg):
    tmp_path.joinpath("MLproject").write_text(invalid_project_contents)
    with pytest.raises(ExecutionException, match=expected_error_msg) as e:
        _project_spec.load_project(str(tmp_path))
    assert expected_error_msg in str(e.value)
