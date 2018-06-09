import os

import mock
import pytest
from six.moves import shlex_quote


from mlflow.projects import ExecutionException
from mlflow.utils.file_utils import TempDir
from tests.projects.utils import load_project, TEST_PROJECT_DIR


def test_entry_point_compute_params():
    """
    Tests that EntryPoint correctly computes a final set of parameters to use when running a project
    """
    project = load_project()
    entry_point = project.get_entry_point("greeter")
    # Pass extra "excitement" param, use default value for `greeting` param
    with TempDir() as storage_dir:
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
    """
    Tests that EntryPoint correctly computes the command to execute in order to run the entry point.
    """
    project = load_project()
    entry_point = project.get_entry_point("greeter")
    with TempDir() as tmp:
        storage_dir = tmp.path()
        command = entry_point.compute_command({"name": "friend", "excitement": 10}, storage_dir)
        assert command == "python greeter.py hi friend --excitement 10"
        with pytest.raises(ExecutionException):
            entry_point.compute_command({}, storage_dir)
        # Test shell escaping
        name_value = "friend; echo 'hi'"
        command = entry_point.compute_command({"name": name_value}, storage_dir)
        assert command == "python greeter.py %s %s" % (shlex_quote("hi"), shlex_quote(name_value))


def test_path_parameter():
    """
    Tests that MLflow file-download APIs get called when necessary for arguments of type `path`.
    """
    project = load_project()
    entry_point = project.get_entry_point("line_count")
    with mock.patch("mlflow.data.download_uri") as download_uri_mock:
        download_uri_mock.return_value = 0
        # Verify that we don't attempt to call download_uri when passing a local file to a
        # parameter of type "path"
        with TempDir() as tmp:
            dst_dir = tmp.path()
            local_path = os.path.join(TEST_PROJECT_DIR, "MLproject")
            params, _ = entry_point.compute_parameters(
                user_parameters={"path": local_path},
                storage_dir=dst_dir)
            assert params["path"] == os.path.abspath(local_path)
            assert download_uri_mock.call_count == 0
        # Verify that we raise an exception when passing a non-existent local file to a
        # parameter of type "path"
        with TempDir() as tmp, pytest.raises(ExecutionException):
            dst_dir = tmp.path()
            entry_point.compute_parameters(
                user_parameters={"path": os.path.join(dst_dir, "some/nonexistent/file")},
                storage_dir=dst_dir)
        # Verify that we do call `download_uri` when passing a URI to a parameter of type "path"
        for i, prefix in enumerate(["dbfs:/", "s3://"]):
            with TempDir() as tmp:
                dst_dir = tmp.path()
                params, _ = entry_point.compute_parameters(
                    user_parameters={"path": os.path.join(prefix, "some/path")},
                    storage_dir=dst_dir)
                assert os.path.dirname(params["path"]) == dst_dir
                assert download_uri_mock.call_count == i + 1


def test_uri_parameter():
    """Tests parameter resolution for parameters of type `uri`."""
    project = load_project()
    entry_point = project.get_entry_point("download_uri")
    with mock.patch("mlflow.data.download_uri") as download_uri_mock, TempDir() as tmp:
        dst_dir = tmp.path()
        # Test that we don't attempt to locally download parameters of type URI
        entry_point.compute_command(user_parameters={"uri": "file://%s" % dst_dir},
                                    storage_dir=dst_dir)
        assert download_uri_mock.call_count == 0
        # Test that we raise an exception if a local path is passed to a parameter of type URI
        with pytest.raises(ExecutionException):
            entry_point.compute_command(user_parameters={"uri": dst_dir}, storage_dir=dst_dir)
