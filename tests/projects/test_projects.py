from contextlib import contextmanager
import filecmp
import mock
import os
import shutil
import tempfile

import pytest
from six.moves import shlex_quote
import yaml

import mlflow
from mlflow.projects import Project, ExecutionException

TEST_DIR = "tests"
TEST_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project")


@contextmanager
def temp_directory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def load_project():
    with open(os.path.join(TEST_PROJECT_DIR, "MLproject")) as mlproject_file:
        project_yaml = yaml.safe_load(mlproject_file.read())
    return Project(uri=TEST_PROJECT_DIR, yaml_obj=project_yaml)


def test_entry_point_compute_params():
    project = load_project()
    entry_point = project.get_entry_point("greeter")
    # Pass extra "excitement" param, use default value for `greeting` param
    with temp_directory() as storage_dir:
        params, extra_params = entry_point.compute_parameters(
            {"name": "friend", "excitement": 10}, storage_dir)
        assert params == {"name": "friend", "greeting": "hi"}
        assert extra_params == {"excitement": "10"}
        # Don't pass extra "excitement" param, pass value for `greeting`
        params, extra_params = entry_point.compute_parameters(
            {"name": "friend", "greeting": "hello"}, storage_dir)
        assert params == {"name": "friend", "greeting": "hello"}
        assert extra_params == {}
        # Raise exception on missing required parameter
        with pytest.raises(ExecutionException):
            entry_point.compute_parameters({}, storage_dir)


def test_entry_point_compute_command():
    project = load_project()
    entry_point = project.get_entry_point("greeter")
    with temp_directory() as storage_dir:
        command = entry_point.compute_command({"name": "friend", "excitement": 10}, storage_dir)
        assert command == "python greeter.py hi friend --excitement 10"
        with pytest.raises(ExecutionException):
            entry_point.compute_command({}, storage_dir)
        # Test shell escaping
        name_value = "friend; echo 'hi'"
        command = entry_point.compute_command({"name": name_value}, storage_dir)
        assert command == "python greeter.py %s %s" % (shlex_quote("hi"), shlex_quote(name_value))


def test_project_get_entry_point():
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


def test_fetch_project():
    # Fetch local project, verify contents match
    with temp_directory() as dst_dir:
        mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, version=None, dst_dir=dst_dir)
        dir_comparison = filecmp.dircmp(TEST_PROJECT_DIR, dst_dir)
        assert len(dir_comparison.left_only) == 0
        assert len(dir_comparison.right_only) == 0
        assert len(dir_comparison.diff_files) == 0
        assert len(dir_comparison.funny_files) == 0
    # Passing `version` raises an exception for local projects
    with temp_directory() as dst_dir:
        with pytest.raises(ExecutionException):
            mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, version="some-version",
                                           dst_dir=dst_dir)


def test_path_parameter():
    # Test that download gets called for arguments of type `path`
    project = load_project()
    entry_point = project.get_entry_point("line_count")
    with mock.patch("mlflow.data.download_uri") as download_uri_mock:
        download_uri_mock.return_value = 0
        # Verify that we don't attempt to call download_uri when passing a local file to a
        # parameter of type "path"
        with temp_directory() as dst_dir:
            local_path = os.path.join(TEST_PROJECT_DIR, "MLproject")
            params, _ = entry_point.compute_parameters(
                user_parameters={"path": local_path},
                storage_dir=dst_dir)
            assert params["path"] == os.path.abspath(local_path)
            assert download_uri_mock.call_count == 0
        # Verify that we raise an exception when passing a non-existent local file to a
        # parameter of type "path"
        with temp_directory() as dst_dir, pytest.raises(ExecutionException):
            entry_point.compute_parameters(
                user_parameters={"path": os.path.join(dst_dir, "some/nonexistent/file")},
                storage_dir=dst_dir)
        # Verify that we do call `download_uri` when passing a URI to a parameter of type "path"
        for i, prefix in enumerate(["dbfs:/", "s3://"]):
            with temp_directory() as dst_dir:
                entry_point.compute_parameters(
                    user_parameters={"path": os.path.join(prefix, "some/path")},
                    storage_dir=dst_dir)
                assert download_uri_mock.call_count == i + 1


def test_uri_parameter():
    project = load_project()
    entry_point = project.get_entry_point("download_uri")
    with mock.patch("mlflow.data.download_uri") as download_uri_mock, temp_directory() as dst_dir:
        # Test that we don't attempt to locally download parameters of type URI
        entry_point.compute_command(user_parameters={"uri": "file://%s" % dst_dir},
                                    storage_dir=dst_dir)
        assert download_uri_mock.call_count == 0
        # Test that we raise an exception if a local path is passed to a parameter of type URI
        with pytest.raises(ExecutionException):
            entry_point.compute_command(user_parameters={"uri": dst_dir}, storage_dir=dst_dir)
