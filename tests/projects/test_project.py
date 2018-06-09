import os

import pytest

from mlflow.projects import ExecutionException
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
