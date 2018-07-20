import os
import filecmp
import tempfile

import pytest

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.projects import ExecutionException
from mlflow.store.file_store import FileStore
from mlflow.utils import env

from tests.projects.utils import TEST_PROJECT_DIR, GIT_PROJECT_URI, validate_exit_status,\
    tracking_uri_mock  # pylint: disable=unused-import


def _assert_dirs_equal(expected, actual):
    dir_comparison = filecmp.dircmp(actual, expected)
    assert len(dir_comparison.left_only) == 0
    assert len(dir_comparison.right_only) == 0
    assert len(dir_comparison.diff_files) == 0
    assert len(dir_comparison.funny_files) == 0


def test_fetch_project(tmpdir):
    """ Test fetching a project to be run locally. """
    dst_dir = tmpdir.join("dst-dir").strpath
    mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, version=None, dst_dir=dst_dir,
                                   git_username=None, git_password=None)
    _assert_dirs_equal(expected=TEST_PROJECT_DIR, actual=dst_dir)
    # Passing `version` raises an exception for local projects
    with pytest.raises(ExecutionException):
        mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, version="some-version",
                                       dst_dir=tmpdir.join("pass-version").strpath,
                                       git_username=None, git_password=None)
    # Passing only one of git_username, git_password results in an error
    for i, (username, password) in enumerate([(None, "hi"), ("hi", None)]):
        with pytest.raises(ExecutionException):
            mlflow.projects._fetch_project(
                uri=TEST_PROJECT_DIR, version="some-version",
                dst_dir=tmpdir.join("partial-credentials-%s" % i).strpath, git_username=username,
                git_password=password)
    # Fetching a directory containing an "mlruns" folder doesn't remove the "mlruns" folder
    src_dir = tmpdir.mkdir("mlruns-src-dir")
    src_dir.mkdir("mlruns").join("some-file.txt").write("hi")
    dst_dir_path = tmpdir.join("mlruns-work-dir").strpath
    src_dir_path = src_dir.strpath
    mlflow.projects._fetch_project(uri=src_dir_path, version=None, dst_dir=dst_dir_path,
                                   git_username=None, git_password=None)
    _assert_dirs_equal(expected=src_dir_path, actual=dst_dir_path)


def test_invalid_run_mode(tracking_uri_mock):  # pylint: disable=unused-argument
    """ Verify that we raise an exception given an invalid run mode """
    with pytest.raises(ExecutionException):
        mlflow.projects.run(uri=TEST_PROJECT_DIR, mode="some unsupported mode")


def test_use_conda(tracking_uri_mock):  # pylint: disable=unused-argument
    """ Verify that we correctly handle the `use_conda` argument."""
    # Verify we throw an exception when conda is unavailable
    old_path = os.environ["PATH"]
    env.unset_variable("PATH")
    try:
        with pytest.raises(ExecutionException):
            mlflow.projects.run(TEST_PROJECT_DIR, use_conda=True)
    finally:
        os.environ["PATH"] = old_path


def test_log_parameters(tmpdir, tracking_uri_mock):  # pylint: disable=unused-argument
    """ Test that we log provided parameters when running a project. """
    mlflow.projects.run(
        TEST_PROJECT_DIR, entry_point="greeter", parameters={"name": "friend"},
        use_conda=False, experiment_id=0)
    store = FileStore(tmpdir.strpath)
    run_uuid = store.list_run_infos(experiment_id=0)[0].run_uuid
    run = store.get_run(run_uuid)
    expected_params = {"name": "friend"}
    assert len(run.data.params) == len(expected_params)
    for param in run.data.params:
        assert param.value == expected_params[param.key]


@pytest.mark.parametrize("use_start_run", map(str, [0, 1]))
def test_run(tmpdir, tracking_uri_mock, use_start_run):  # pylint: disable=unused-argument
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
    store = FileStore(tmpdir.strpath)
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


def test_run_async(tracking_uri_mock):  # pylint: disable=unused-argument
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


def test_cancel_run(tracking_uri_mock):  # pylint: disable=unused-argument
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


def test_storage_dir(tmpdir):
    """
    Test that we correctly handle the `storage_dir` argument, which specifies where to download
    distributed artifacts passed to arguments of type `path`.
    """
    assert os.path.dirname(mlflow.projects._get_storage_dir(tmpdir.strpath)) == tmpdir.strpath
    assert os.path.dirname(mlflow.projects._get_storage_dir(None)) == tempfile.gettempdir()
