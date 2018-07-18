import os
import filecmp
import tempfile

import mock
import pytest

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.projects import ExecutionException
from mlflow.store.file_store import FileStore
from mlflow.utils.file_utils import TempDir
from mlflow.utils import env

from tests.projects.utils import TEST_PROJECT_DIR, GIT_PROJECT_URI, validate_exit_status


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


def test_invalid_run_mode():
    """ Verify that we raise an exception given an invalid run mode """
    with TempDir() as tmp, mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = tmp.path()
        with pytest.raises(ExecutionException):
            mlflow.projects.run(uri=TEST_PROJECT_DIR, mode="some unsupported mode")


def test_use_conda():
    """ Verify that we correctly handle the `use_conda` argument."""
    with TempDir() as tmp, mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = tmp.path()
        # Verify we throw an exception when conda is unavailable
        old_path = os.environ["PATH"]
        env.unset_variable("PATH")
        try:
            with pytest.raises(ExecutionException):
                mlflow.projects.run(TEST_PROJECT_DIR, use_conda=True)
        finally:
            os.environ["PATH"] = old_path


def test_run():
    for use_start_run in map(str, [0, 1]):
        with TempDir() as tmp, mock.patch("mlflow.tracking.get_tracking_uri")\
                as get_tracking_uri_mock:
            tmp_dir = tmp.path()
            get_tracking_uri_mock.return_value = tmp_dir
            submitted_run = mlflow.projects.run(
                TEST_PROJECT_DIR, entry_point="test_tracking",
                parameters={"use_start_run": use_start_run},
                use_conda=False, experiment_id=0)
            # Blocking runs should be finished when they return
            validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
            # Test that we can call wait() on a synchronous run & that the run has the correct
            # status after calling wait().
            submitted_run.wait()
            validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
            # Validate run contents in the FileStore
            run_uuid = submitted_run.run_id
            store = FileStore(tmp_dir)
            run_infos = store.list_run_infos(experiment_id=0)
            assert len(run_infos) == 1
            store_run_uuid = run_infos[0].run_uuid
            assert run_uuid == store_run_uuid
            run = store.get_run(run_uuid)
            expected_params = {"use_start_run": use_start_run}
            assert run.info.status == RunStatus.FINISHED
            assert len(run.data.params) == len(expected_params)
            for param in run.data.params:
                assert param.value == expected_params[param.key]
            expected_metrics = {"some_key": 3}
            for metric in run.data.metrics:
                assert metric.value == expected_metrics[metric.key]


def test_run_async():
    with TempDir() as tmp, mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        tmp_dir = tmp.path()
        get_tracking_uri_mock.return_value = tmp_dir
        submitted_run0 = mlflow.projects.run(
            TEST_PROJECT_DIR, entry_point="sleep", parameters={"duration": 2},
            use_conda=False, experiment_id=0, block=False)
        validate_exit_status(submitted_run0.get_status(), RunStatus.RUNNING)
        submitted_run0.wait()
        validate_exit_status(submitted_run0.get_status(), RunStatus.FINISHED)
        submitted_run1 = mlflow.projects.run(
            TEST_PROJECT_DIR, entry_point="sleep", parameters={"duration": -1, "invalid-param": 30},
            use_conda=False, experiment_id=0, block=False)
        submitted_run1.wait()
        validate_exit_status(submitted_run1.get_status(), RunStatus.FAILED)


def test_cancel_run():
    with TempDir() as tmp, mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        tmp_dir = tmp.path()
        get_tracking_uri_mock.return_value = tmp_dir
        submitted_run0, submitted_run1 = [mlflow.projects.run(
            TEST_PROJECT_DIR, entry_point="sleep", parameters={"duration": 2},
            use_conda=False, experiment_id=0, block=False) for _ in range(2)]
        submitted_run0.cancel()
        validate_exit_status(submitted_run0.get_status(), RunStatus.FAILED)
        # Sanity check: cancelling one run has no effect on the other
        submitted_run1.wait()
        validate_exit_status(submitted_run1.get_status(), RunStatus.FINISHED)


def test_get_work_dir():
    """ Test that we correctly determine the working directory to use when running a project. """
    for use_temp_cwd, uri in [(True, TEST_PROJECT_DIR), (False, GIT_PROJECT_URI)]:
        work_dir = mlflow.projects._get_work_dir(uri=uri, use_temp_cwd=use_temp_cwd)
        assert work_dir != uri
        assert os.path.exists(work_dir)
    for use_temp_cwd, uri in [(None, TEST_PROJECT_DIR), (False, TEST_PROJECT_DIR)]:
        assert mlflow.projects._get_work_dir(uri=uri, use_temp_cwd=use_temp_cwd) ==\
               os.path.abspath(TEST_PROJECT_DIR)


def test_storage_dir():
    """
    Test that we correctly handle the `storage_dir` argument, which specifies where to download
    distributed artifacts passed to arguments of type `path`.
    """
    with TempDir() as tmp_dir:
        assert os.path.dirname(mlflow.projects._get_storage_dir(tmp_dir.path())) == tmp_dir.path()
    assert os.path.dirname(mlflow.projects._get_storage_dir(None)) == tempfile.gettempdir()
