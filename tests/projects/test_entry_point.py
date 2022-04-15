import os

import pytest
from shlex import quote
from unittest import mock

from mlflow.exceptions import ExecutionException
from mlflow.projects._project_spec import EntryPoint
from mlflow.utils.file_utils import TempDir, path_to_local_file_uri
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
            {"name": "friend", "excitement": 10}, storage_dir
        )
        assert params == {"name": "friend", "greeting": "hi"}
        assert extra_params == {"excitement": "10"}
        # Don't pass extra "excitement" param, pass value for `greeting`
        params, extra_params = entry_point.compute_parameters(
            {"name": "friend", "greeting": "hello"}, storage_dir
        )
        assert params == {"name": "friend", "greeting": "hello"}
        assert extra_params == {}
        # Raise exception on missing required parameter
        with pytest.raises(
            ExecutionException, match="No value given for missing parameters: 'name'"
        ):
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
        with pytest.raises(
            ExecutionException, match="No value given for missing parameters: 'name'"
        ):
            entry_point.compute_command({}, storage_dir)
        # Test shell escaping
        name_value = "friend; echo 'hi'"
        command = entry_point.compute_command({"name": name_value}, storage_dir)
        assert command == "python greeter.py %s %s" % (quote("hi"), quote(name_value))


def test_path_parameter():
    """
    Tests that MLflow file-download APIs get called when necessary for arguments of type `path`.
    """
    project = load_project()
    entry_point = project.get_entry_point("line_count")
    with mock.patch(
        "mlflow.tracking.artifact_utils._download_artifact_from_uri"
    ) as download_uri_mock:
        download_uri_mock.return_value = 0
        # Verify that we don't attempt to call download_uri when passing a local file to a
        # parameter of type "path"
        with TempDir() as tmp:
            dst_dir = tmp.path()
            local_path = os.path.join(TEST_PROJECT_DIR, "MLproject")
            params, _ = entry_point.compute_parameters(
                user_parameters={"path": local_path}, storage_dir=dst_dir
            )
            assert params["path"] == os.path.abspath(local_path)
            assert download_uri_mock.call_count == 0

            params, _ = entry_point.compute_parameters(
                user_parameters={"path": path_to_local_file_uri(local_path)}, storage_dir=dst_dir
            )
            assert params["path"] == os.path.abspath(local_path)
            assert download_uri_mock.call_count == 0

        # Verify that we raise an exception when passing a non-existent local file to a
        # parameter of type "path"
        with TempDir() as tmp, pytest.raises(ExecutionException, match="no such file or directory"):
            dst_dir = tmp.path()
            entry_point.compute_parameters(
                user_parameters={"path": os.path.join(dst_dir, "some/nonexistent/file")},
                storage_dir=dst_dir,
            )
        # Verify that we do call `download_uri` when passing a URI to a parameter of type "path"
        for i, prefix in enumerate(["dbfs:/", "s3://", "gs://"]):
            with TempDir() as tmp:
                dst_dir = tmp.path()
                file_to_download = "images.tgz"
                download_path = "%s/%s" % (dst_dir, file_to_download)
                download_uri_mock.return_value = download_path
                params, _ = entry_point.compute_parameters(
                    user_parameters={"path": os.path.join(prefix, file_to_download)},
                    storage_dir=dst_dir,
                )
                assert params["path"] == download_path
                assert download_uri_mock.call_count == i + 1


def test_uri_parameter():
    """Tests parameter resolution for parameters of type `uri`."""
    project = load_project()
    entry_point = project.get_entry_point("download_uri")
    with mock.patch(
        "mlflow.tracking.artifact_utils._download_artifact_from_uri"
    ) as download_uri_mock, TempDir() as tmp:
        dst_dir = tmp.path()
        # Test that we don't attempt to locally download parameters of type URI
        entry_point.compute_command(
            user_parameters={"uri": "file://%s" % dst_dir}, storage_dir=dst_dir
        )
        assert download_uri_mock.call_count == 0
        # Test that we raise an exception if a local path is passed to a parameter of type URI
        with pytest.raises(ExecutionException, match="Expected URI for parameter uri"):
            entry_point.compute_command(user_parameters={"uri": dst_dir}, storage_dir=dst_dir)


def test_params():
    defaults = {
        "alpha": "float",
        "l1_ratio": {"type": "float", "default": 0.1},
        "l2_ratio": {"type": "float", "default": 0.0003},
        "random_str": {"type": "string", "default": "hello"},
    }
    entry_point = EntryPoint("entry_point_name", defaults, "command_name script.py")

    user1 = {}
    with pytest.raises(ExecutionException, match="No value given for missing parameters"):
        entry_point._validate_parameters(user1)

    user_2 = {"beta": 0.004}
    with pytest.raises(ExecutionException, match="No value given for missing parameters"):
        entry_point._validate_parameters(user_2)

    user_3 = {"alpha": 0.004, "gamma": 0.89}
    expected_final_3 = {
        "alpha": "0.004",
        "l1_ratio": "0.1",
        "l2_ratio": "0.0003",
        "random_str": "hello",
    }
    expected_extra_3 = {"gamma": "0.89"}
    final_3, extra_3 = entry_point.compute_parameters(user_3, None)
    assert expected_extra_3 == extra_3
    assert expected_final_3 == final_3

    user_4 = {"alpha": 0.004, "l1_ratio": 0.0008, "random_str_2": "hello"}
    expected_final_4 = {
        "alpha": "0.004",
        "l1_ratio": "0.0008",
        "l2_ratio": "0.0003",
        "random_str": "hello",
    }
    expected_extra_4 = {"random_str_2": "hello"}
    final_4, extra_4 = entry_point.compute_parameters(user_4, None)
    assert expected_extra_4 == extra_4
    assert expected_final_4 == final_4

    user_5 = {"alpha": -0.99, "random_str": "hi"}
    expected_final_5 = {
        "alpha": "-0.99",
        "l1_ratio": "0.1",
        "l2_ratio": "0.0003",
        "random_str": "hi",
    }
    expected_extra_5 = {}
    final_5, extra_5 = entry_point.compute_parameters(user_5, None)
    assert expected_final_5 == final_5
    assert expected_extra_5 == extra_5

    user_6 = {"alpha": 0.77, "ALPHA": 0.89}
    expected_final_6 = {
        "alpha": "0.77",
        "l1_ratio": "0.1",
        "l2_ratio": "0.0003",
        "random_str": "hello",
    }
    expected_extra_6 = {"ALPHA": "0.89"}
    final_6, extra_6 = entry_point.compute_parameters(user_6, None)
    assert expected_extra_6 == extra_6
    assert expected_final_6 == final_6


def test_path_params():
    data_file = "s3://path.test/resources/data_file.csv"
    defaults = {
        "constants": {"type": "uri", "default": "s3://path.test/b1"},
        "data": {"type": "path", "default": data_file},
    }
    entry_point = EntryPoint("entry_point_name", defaults, "command_name script.py")

    with mock.patch(
        "mlflow.tracking.artifact_utils._download_artifact_from_uri"
    ) as download_uri_mock:
        final_1, extra_1 = entry_point.compute_parameters({}, None)
        assert final_1 == {"constants": "s3://path.test/b1", "data": data_file}
        assert extra_1 == {}
        assert download_uri_mock.call_count == 0

    with mock.patch(
        "mlflow.tracking.artifact_utils._download_artifact_from_uri"
    ) as download_uri_mock:
        user_2 = {"alpha": 0.001, "constants": "s3://path.test/b_two"}
        final_2, extra_2 = entry_point.compute_parameters(user_2, None)
        assert final_2 == {"constants": "s3://path.test/b_two", "data": data_file}
        assert extra_2 == {"alpha": "0.001"}
        assert download_uri_mock.call_count == 0

    with mock.patch(
        "mlflow.tracking.artifact_utils._download_artifact_from_uri"
    ) as download_uri_mock, TempDir() as tmp:
        dest_path = tmp.path()
        download_path = "%s/data_file.csv" % dest_path
        download_uri_mock.return_value = download_path
        user_3 = {"alpha": 0.001}
        final_3, extra_3 = entry_point.compute_parameters(user_3, dest_path)
        assert final_3 == {"constants": "s3://path.test/b1", "data": download_path}
        assert extra_3 == {"alpha": "0.001"}
        assert download_uri_mock.call_count == 1

    with mock.patch(
        "mlflow.tracking.artifact_utils._download_artifact_from_uri"
    ) as download_uri_mock, TempDir() as tmp:
        dest_path = tmp.path()
        download_path = "%s/images.tgz" % dest_path
        download_uri_mock.return_value = download_path
        user_4 = {"data": "s3://another.example.test/data_stash/images.tgz"}
        final_4, extra_4 = entry_point.compute_parameters(user_4, dest_path)
        assert final_4 == {"constants": "s3://path.test/b1", "data": download_path}
        assert extra_4 == {}
        assert download_uri_mock.call_count == 1
