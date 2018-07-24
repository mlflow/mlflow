import os
import filecmp
import git
import tempfile

from distutils import dir_util

import mock
import pytest

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.projects import ExecutionException
from mlflow.store.file_store import FileStore
from mlflow.utils.file_utils import TempDir
from mlflow.utils import env

from tests.projects.utils import TEST_PROJECT_DIR, GIT_PROJECT_URI, TEST_DIR, validate_exit_status


def _assert_dirs_equal(expected, actual):
    dir_comparison = filecmp.dircmp(expected, actual)
    assert len(dir_comparison.left_only) == 0
    assert len(dir_comparison.right_only) == 0
    assert len(dir_comparison.diff_files) == 0
    assert len(dir_comparison.funny_files) == 0


def test_fetch_project(tmpdir):
    # Creating a local Git repo containing a MLproject file.
    local_git = tmpdir.join('git_repo').strpath
    repo = git.Repo.init(local_git)
    dir_util.copy_tree(src=TEST_PROJECT_DIR, dst=local_git)
    repo.git.add(A=True)
    repo.index.commit("test")
    git_repo_uri = "file://" + os.path.abspath(local_git)

    # Creating a local Git repo with a MLproject file in a subdirectory.
    local_git_subdir = tmpdir.join('subdir_git_repo').strpath
    repo = git.Repo.init(local_git_subdir)
    dir_util.copy_tree(src=os.path.join(TEST_DIR, "resources"), dst=local_git_subdir)
    repo.git.add(A=True)
    repo.index.commit("test")
    git_subdir_repo = "file://" + os.path.abspath(local_git_subdir)

    # Test fetching a locally saved project.
    dst_dir = tmpdir.join('local').strpath

    work_dir = mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, subdirectory='', version=None,
                                              dst_dir=dst_dir, git_username=None,
                                              git_password=None)
    _assert_dirs_equal(expected=TEST_PROJECT_DIR, actual=work_dir)

    # Test fetching a project located in a Git repo root directory.
    dst_dir = tmpdir.join('git-root-dir').strpath
    work_dir = mlflow.projects._fetch_project(uri=git_repo_uri, subdirectory='',
                                              version=None, dst_dir=dst_dir, git_username=None,
                                              git_password=None)
    _assert_dirs_equal(expected=local_git, actual=work_dir)

    # Test fetching a project located in a Git repo subdirectory.
    dst_dir = tmpdir.join('git-subdir').strpath
    work_dir = mlflow.projects._fetch_project(uri=git_subdir_repo, subdirectory='example_project',
                                              version=None, dst_dir=dst_dir, git_username=None,
                                              git_password=None)
    _assert_dirs_equal(expected=os.path.join(local_git_subdir, 'example_project'), actual=work_dir)

    # Test passing a subdirectory with `#` works for local directories.
    work_dir = mlflow.projects._fetch_project(uri=TEST_DIR,
                                              subdirectory="resources/example_project",
                                              version=None, dst_dir=dst_dir, git_username=None,
                                              git_password=None)
    _assert_dirs_equal(expected=TEST_PROJECT_DIR, actual=work_dir)

    # Passing `version` raises an exception for local projects
    with pytest.raises(ExecutionException):
        mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, subdirectory='', version="version",
                                       dst_dir=dst_dir, git_username=None, git_password=None)

    # Passing only one of git_username, git_password results in an error
    for username, password in [(None, "hi"), ("hi", None)]:
        with pytest.raises(ExecutionException):
            mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, subdirectory='',
                                           version="some-version", dst_dir=dst_dir,
                                           git_username=username, git_password=password)

    # Verify that runs fail if given incorrect subdirectories via the `#` character.
    # Local test.
    with pytest.raises(ExecutionException):
        mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, subdirectory='fake', version=None,
                                       dst_dir=dst_dir, git_username=None, git_password=None)

    # Tests that an exception is thrown when an invalid subdirectory is given.
    dst_dir = tmpdir.join('git-bad-subdirectory').strpath
    with pytest.raises(ExecutionException):
        mlflow.projects._fetch_project(uri=git_repo_uri, subdirectory='fake', version=None,
                                       dst_dir=dst_dir, git_username=None, git_password=None)


def test_parse_subdirectory():
    # Make sure the parsing works as intended.
    test_uri = "uri#subdirectory"
    parsed_uri, parsed_subdirectory = mlflow.projects._parse_subdirectory(test_uri)
    assert parsed_uri == "uri"
    assert parsed_subdirectory == "subdirectory"

    # Make sure periods are restricted in Git repo subdirectory paths.
    period_fail_uri = GIT_PROJECT_URI + "#.."
    with pytest.raises(ExecutionException):
        mlflow.projects._parse_subdirectory(period_fail_uri)


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


@pytest.mark.parametrize(
    "mock_env,expected",
    [({}, "conda"), ({mlflow.projects.MLFLOW_CONDA: "/some/dir/conda"}, "/some/dir/conda")]
)
def test_conda_path(mock_env, expected):
    """Verify that we correctly determine the path to a conda executable"""
    with mock.patch.dict("os.environ", mock_env):
        assert mlflow.projects._conda_executable() == expected


@pytest.mark.skip(reason="hangs in travis py3.6")
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


@pytest.mark.skip(reason="hangs in travis py3.6")
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


@pytest.mark.skip(reason="hangs in travis py3.6")
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


def test_get_dest_dir():
    """ Test that we correctly determine the dest directory to use when fetching a project. """
    for use_temp_cwd, uri in [(True, TEST_PROJECT_DIR), (False, GIT_PROJECT_URI)]:
        dest_dir = mlflow.projects._get_dest_dir(uri=uri, use_temp_cwd=use_temp_cwd)
        assert dest_dir != uri
        assert os.path.exists(dest_dir)
    for use_temp_cwd, uri in [(None, TEST_PROJECT_DIR), (False, TEST_PROJECT_DIR)]:
        assert mlflow.projects._get_dest_dir(uri=uri, use_temp_cwd=use_temp_cwd) ==\
               os.path.abspath(TEST_PROJECT_DIR)


def test_storage_dir():
    """
    Test that we correctly handle the `storage_dir` argument, which specifies where to download
    distributed artifacts passed to arguments of type `path`.
    """
    with TempDir() as tmp_dir:
        assert os.path.dirname(mlflow.projects._get_storage_dir(tmp_dir.path())) == tmp_dir.path()
    assert os.path.dirname(mlflow.projects._get_storage_dir(None)) == tempfile.gettempdir()
