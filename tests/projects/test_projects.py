import os
import filecmp
import tempfile

import mock
import pytest

import mlflow
from mlflow.projects import ExecutionException
from mlflow.store.file_store import FileStore
from mlflow.utils.file_utils import TempDir
from mlflow.utils import env

from tests.projects.utils import TEST_PROJECT_DIR, GIT_PROJECT_URI


def test_fetch_project():
    """ Test fetching a project to be run locally. """
    with TempDir() as tmp:
        dst_dir = tmp.path()
        mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, version=None, dst_dir=dst_dir,
                                       git_username=None, git_password=None)
        dir_comparison = filecmp.dircmp(TEST_PROJECT_DIR, dst_dir)
        assert len(dir_comparison.left_only) == 0
        assert len(dir_comparison.right_only) == 0
        assert len(dir_comparison.diff_files) == 0
        assert len(dir_comparison.funny_files) == 0
    # Passing `version` raises an exception for local projects
    with TempDir() as dst_dir:
        with pytest.raises(ExecutionException):
            mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, version="some-version",
                                           dst_dir=dst_dir, git_username=None, git_password=None)
    # Passing only one of git_username, git_password results in an error
    for username, password in [(None, "hi"), ("hi", None)]:
        with TempDir() as dst_dir:
            with pytest.raises(ExecutionException):
                mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, version="some-version",
                                               dst_dir=dst_dir, git_username=username,
                                               git_password=password)


def test_run_mode():
    """ Verify that we pick the right run helper given an execution mode """
    with TempDir() as tmp, mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = tmp.path()
        for local_mode in ["local", None]:
            with mock.patch("mlflow.projects._run_local") as run_local_mock:
                mlflow.projects.run(uri=TEST_PROJECT_DIR, mode=local_mode)
                assert run_local_mock.call_count == 1
        with mock.patch("mlflow.projects._run_databricks") as run_databricks_mock:
            mlflow.projects.run(uri=TEST_PROJECT_DIR, mode="databricks")
            assert run_databricks_mock.call_count == 1
        with pytest.raises(ExecutionException):
            mlflow.projects.run(uri=TEST_PROJECT_DIR, mode="some unsupported mode")


def test_use_conda():
    """ Verify that we correctly handle the `use_conda` argument."""
    with TempDir() as tmp, mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = tmp.path()
        for use_conda, expected_call_count in [(True, 1), (False, 0), (None, 0)]:
            with mock.patch("mlflow.projects._maybe_create_conda_env") as conda_env_mock:
                mlflow.projects.run(TEST_PROJECT_DIR, use_conda=use_conda)
                assert conda_env_mock.call_count == expected_call_count
        # Verify we throw an exception when conda is unavailable
        old_path = os.environ["PATH"]
        env.unset_variable("PATH")
        try:
            with pytest.raises(ExecutionException):
                mlflow.projects.run(TEST_PROJECT_DIR, use_conda=True)
        finally:
            os.environ["PATH"] = old_path


def test_log_parameters():
    """ Test that we log provided parameters when running a project. """
    with TempDir() as tmp, mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        tmp_dir = tmp.path()
        get_tracking_uri_mock.return_value = tmp_dir
        mlflow.projects.run(
            TEST_PROJECT_DIR, entry_point="greeter", parameters={"name": "friend"},
            use_conda=False, experiment_id=0)
        store = FileStore(tmp_dir)
        run_uuid = store.list_run_infos(experiment_id=0)[0].run_uuid
        run = store.get_run(run_uuid)
        expected_params = {"name": "friend"}
        assert len(run.data.params) == len(expected_params)
        for param in run.data.params:
            assert param.value == expected_params[param.key]


def test_get_work_dir():
    """ Test that we correctly determine the working directory to use when running a project. """
    for use_temp_cwd, uri in [(True, TEST_PROJECT_DIR), (False, GIT_PROJECT_URI)]:
        work_dir = mlflow.projects._get_work_dir(uri=uri, use_temp_cwd=use_temp_cwd)
        assert work_dir != uri
        assert os.path.exists(work_dir)
    for use_temp_cwd, uri in [(None, TEST_PROJECT_DIR), (False, TEST_PROJECT_DIR)]:
        assert mlflow.projects._get_work_dir(uri=uri, use_temp_cwd=use_temp_cwd) == TEST_PROJECT_DIR


def test_storage_dir():
    """
    Test that we correctly handle the `storage_dir` argument, which specifies where to download
    distributed artifacts passed to arguments of type `path`.
    """
    with TempDir() as tmp_dir:
        assert os.path.dirname(mlflow.projects._get_storage_dir(tmp_dir.path())) == tmp_dir.path()
    assert os.path.dirname(mlflow.projects._get_storage_dir(None)) == tempfile.gettempdir()
