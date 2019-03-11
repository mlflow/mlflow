import os
import git
import tempfile

from distutils import dir_util

import mock
import pytest

import mlflow
from mlflow.entities import RunStatus, ViewType
from mlflow.exceptions import ExecutionException
from mlflow.utils import env
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_GIT_BRANCH, MLFLOW_GIT_REPO_URL, \
    LEGACY_MLFLOW_GIT_BRANCH_NAME, LEGACY_MLFLOW_GIT_REPO_URL

from tests.projects.utils import TEST_PROJECT_DIR, TEST_PROJECT_NAME, GIT_PROJECT_URI, \
    validate_exit_status, assert_dirs_equal
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


def _build_uri(base_uri, subdirectory):
    if subdirectory != "":
        return "%s#%s" % (base_uri, subdirectory)
    return base_uri


def _get_version_local_git_repo(local_git_repo):
    repo = git.Repo(local_git_repo, search_parent_directories=True)
    return repo.git.rev_parse("HEAD")


@pytest.fixture()
def local_git_repo(tmpdir):
    local_git = tmpdir.join('git_repo').strpath
    repo = git.Repo.init(local_git)
    dir_util.copy_tree(src=TEST_PROJECT_DIR, dst=local_git)
    dir_util.copy_tree(src=os.path.dirname(TEST_PROJECT_DIR), dst=local_git)
    repo.git.add(A=True)
    repo.index.commit("test")
    yield os.path.abspath(local_git)


@pytest.fixture()
def local_git_repo_uri(local_git_repo):
    return "file://%s" % local_git_repo


@pytest.fixture()
def zipped_repo(tmpdir):
    import zipfile
    zip_name = tmpdir.join('%s.zip' % TEST_PROJECT_NAME).strpath
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(TEST_PROJECT_DIR):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                zip_file.write(file_path, file_path[len(TEST_PROJECT_DIR)+len(os.sep):])
    return zip_name


def test_is_zip_uri():
    assert mlflow.projects._is_zip_uri('http://foo.bar/moo.zip')
    assert mlflow.projects._is_zip_uri('https://foo.bar/moo.zip')
    assert mlflow.projects._is_zip_uri('file:///moo.zip')
    assert mlflow.projects._is_zip_uri('file://C:/moo.zip')
    assert mlflow.projects._is_zip_uri('/moo.zip')
    assert mlflow.projects._is_zip_uri('C:/moo.zip')
    assert not mlflow.projects._is_zip_uri('http://foo.bar/moo')
    assert not mlflow.projects._is_zip_uri('https://foo.bar/moo')
    assert not mlflow.projects._is_zip_uri('file:///moo')
    assert not mlflow.projects._is_zip_uri('file://C:/moo')
    assert not mlflow.projects._is_zip_uri('/moo')
    assert not mlflow.projects._is_zip_uri('C:/moo')


def test_fetch_project(local_git_repo, local_git_repo_uri, zipped_repo, httpserver):
    httpserver.serve_content(open(zipped_repo, 'rb').read())
    # The tests are as follows:
    # 1. Fetching a locally saved project.
    # 2. Fetching a project located in a Git repo root directory.
    # 3. Fetching a project located in a Git repo subdirectory.
    # 4. Passing a subdirectory works for local directories.
    # 5. Fetching a remote ZIP file
    # 6. Using a local ZIP file
    # 7. Using a file:// URL to a local ZIP file
    test_list = [
        (TEST_PROJECT_DIR, '', TEST_PROJECT_DIR),
        (local_git_repo_uri, '', local_git_repo),
        (local_git_repo_uri, 'example_project', os.path.join(local_git_repo, 'example_project')),
        (os.path.dirname(TEST_PROJECT_DIR), os.path.basename(TEST_PROJECT_DIR), TEST_PROJECT_DIR),
        (httpserver.url + '/%s.zip' % TEST_PROJECT_NAME, '', TEST_PROJECT_DIR),
        (zipped_repo, '', TEST_PROJECT_DIR),
        ('file://%s' % zipped_repo, '', TEST_PROJECT_DIR),
    ]
    for base_uri, subdirectory, expected in test_list:
        work_dir = mlflow.projects._fetch_project(
            uri=_build_uri(base_uri, subdirectory), force_tempdir=False)
        assert_dirs_equal(expected=expected, actual=work_dir)
    # Test that we correctly determine the dest directory to use when fetching a project.
    for force_tempdir, uri in [(True, TEST_PROJECT_DIR), (False, GIT_PROJECT_URI)]:
        dest_dir = mlflow.projects._fetch_project(uri=uri, force_tempdir=force_tempdir)
        assert os.path.commonprefix([dest_dir, tempfile.gettempdir()]) == tempfile.gettempdir()
        assert os.path.exists(dest_dir)
    for force_tempdir, uri in [(None, TEST_PROJECT_DIR), (False, TEST_PROJECT_DIR)]:
        assert mlflow.projects._fetch_project(uri=uri, force_tempdir=force_tempdir) == \
               os.path.abspath(TEST_PROJECT_DIR)


def test_fetch_project_validations(local_git_repo_uri):
    # Verify that runs fail if given incorrect subdirectories via the `#` character.
    for base_uri in [TEST_PROJECT_DIR, local_git_repo_uri]:
        with pytest.raises(ExecutionException):
            mlflow.projects._fetch_project(uri=_build_uri(base_uri, "fake"), force_tempdir=False)

    # Passing `version` raises an exception for local projects
    with pytest.raises(ExecutionException):
        mlflow.projects._fetch_project(uri=TEST_PROJECT_DIR, force_tempdir=False, version="version")

    # Passing only one of git_username, git_password results in an error
    for username, password in [(None, "hi"), ("hi", None)]:
        with pytest.raises(ExecutionException):
            mlflow.projects._fetch_project(
                local_git_repo_uri, force_tempdir=False, git_username=username,
                git_password=password)


def test_dont_remove_mlruns(tmpdir):
    # Fetching a directory containing an "mlruns" folder doesn't remove the "mlruns" folder
    src_dir = tmpdir.mkdir("mlruns-src-dir")
    src_dir.mkdir("mlruns").join("some-file.txt").write("hi")
    src_dir.join("MLproject").write("dummy MLproject contents")
    dst_dir = mlflow.projects._fetch_project(
        uri=src_dir.strpath, version=None, git_username=None,
        git_password=None, force_tempdir=False)
    assert_dirs_equal(expected=src_dir.strpath, actual=dst_dir)


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


def test_is_valid_branch_name(local_git_repo):
    assert mlflow.projects._is_valid_branch_name(local_git_repo, "master")
    assert not mlflow.projects._is_valid_branch_name(local_git_repo, "dev")


@pytest.mark.parametrize("use_start_run", map(str, [0, 1]))
@pytest.mark.parametrize("version", [None, "master", "git-commit"])
def test_run_local_git_repo(local_git_repo,
                            local_git_repo_uri,
                            tracking_uri_mock,   # pylint: disable=unused-argument
                            use_start_run,
                            version):
    if version is not None:
        uri = local_git_repo_uri + "#" + TEST_PROJECT_NAME
    else:
        uri = os.path.join("%s/" % local_git_repo, TEST_PROJECT_NAME)
    if version == "git-commit":
        version = _get_version_local_git_repo(local_git_repo)
    submitted_run = mlflow.projects.run(
        uri, entry_point="test_tracking", version=version,
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
    mlflow_service = mlflow.tracking.MlflowClient()
    run_infos = mlflow_service.list_run_infos(experiment_id=0, run_view_type=ViewType.ACTIVE_ONLY)
    assert "file:" in run_infos[0].source_name
    assert len(run_infos) == 1
    store_run_uuid = run_infos[0].run_uuid
    assert run_uuid == store_run_uuid
    run = mlflow_service.get_run(run_uuid)

    assert run.info.status == RunStatus.FINISHED

    expected_params = {"use_start_run": use_start_run}
    params = {param.key: param.value for param in run.data.params}
    assert params == expected_params

    expected_metrics = {"some_key": 3}
    metrics = {metric.key: metric.value for metric in run.data.metrics}
    assert metrics == expected_metrics

    # Validate the branch name tag is logged
    if version == "master":
        tags = {tag.key: tag.value for tag in run.data.tags}
        assert tags[MLFLOW_GIT_BRANCH] == "master"
        assert tags[MLFLOW_GIT_REPO_URL] == local_git_repo_uri
        assert tags[LEGACY_MLFLOW_GIT_BRANCH_NAME] == "master"
        assert tags[LEGACY_MLFLOW_GIT_REPO_URL] == local_git_repo_uri


def test_invalid_version_local_git_repo(local_git_repo_uri,
                                        tracking_uri_mock):   # pylint: disable=unused-argument
    # Run project with invalid commit hash
    with pytest.raises(ExecutionException,
                       match=r'Unable to checkout version \'badc0de\''):
        mlflow.projects.run(local_git_repo_uri + "#" + TEST_PROJECT_NAME,
                            entry_point="test_tracking", version="badc0de",
                            use_conda=False, experiment_id=0)


@pytest.mark.parametrize("use_start_run", map(str, [0, 1]))
def test_run(tmpdir, tracking_uri_mock, use_start_run):  # pylint: disable=unused-argument
    submitted_run = mlflow.projects.run(
        TEST_PROJECT_DIR, entry_point="test_tracking",
        parameters={"use_start_run": use_start_run},
        use_conda=False, experiment_id=0)
    assert submitted_run.run_id is not None
    # Blocking runs should be finished when they return
    validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
    # Test that we can call wait() on a synchronous run & that the run has the correct
    # status after calling wait().
    submitted_run.wait()
    validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
    # Validate run contents in the FileStore
    run_uuid = submitted_run.run_id
    mlflow_service = mlflow.tracking.MlflowClient()

    run_infos = mlflow_service.list_run_infos(experiment_id=0, run_view_type=ViewType.ACTIVE_ONLY)
    assert len(run_infos) == 1
    store_run_uuid = run_infos[0].run_uuid
    assert run_uuid == store_run_uuid
    run = mlflow_service.get_run(run_uuid)

    assert run.info.status == RunStatus.FINISHED

    expected_params = {"use_start_run": use_start_run}
    params = {param.key: param.value for param in run.data.params}
    assert params == expected_params

    expected_metrics = {"some_key": 3}
    metrics = {metric.key: metric.value for metric in run.data.metrics}
    assert metrics == expected_metrics


def test_run_with_parent(tmpdir, tracking_uri_mock):  # pylint: disable=unused-argument
    """Verify that if we are in a nested run, mlflow.projects.run() will have a parent_run_id."""
    with mlflow.start_run():
        parent_run_id = mlflow.active_run().info.run_uuid
        submitted_run = mlflow.projects.run(
            TEST_PROJECT_DIR, entry_point="test_tracking",
            parameters={"use_start_run": "1"},
            use_conda=False, experiment_id=0)
    assert submitted_run.run_id is not None
    validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
    run_uuid = submitted_run.run_id
    run = mlflow.tracking.MlflowClient().get_run(run_uuid)
    tags = {tag.key: tag.value for tag in run.data.tags}
    assert tags[MLFLOW_PARENT_RUN_ID] == parent_run_id


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


@pytest.mark.parametrize(
    "mock_env,expected_conda,expected_activate",
    [
        ({}, "conda", "activate"),
        ({mlflow.projects.MLFLOW_CONDA_HOME: "/some/dir/"}, "/some/dir/bin/conda",
         "/some/dir/bin/activate")
     ]
)
def test_conda_path(mock_env, expected_conda, expected_activate):
    """Verify that we correctly determine the path to conda executables"""
    with mock.patch.dict("os.environ", mock_env):
        assert mlflow.projects._get_conda_bin_executable("conda") == expected_conda
        assert mlflow.projects._get_conda_bin_executable("activate") == expected_activate


def test_cancel_run(tracking_uri_mock):  # pylint: disable=unused-argument
    submitted_run0, submitted_run1 = [mlflow.projects.run(
        TEST_PROJECT_DIR, entry_point="sleep", parameters={"duration": 2},
        use_conda=False, experiment_id=0, block=False) for _ in range(2)]
    submitted_run0.cancel()
    validate_exit_status(submitted_run0.get_status(), RunStatus.FAILED)
    # Sanity check: cancelling one run has no effect on the other
    assert submitted_run1.wait()
    validate_exit_status(submitted_run1.get_status(), RunStatus.FINISHED)
    # Try cancelling after calling wait()
    submitted_run1.cancel()
    validate_exit_status(submitted_run1.get_status(), RunStatus.FINISHED)


def test_storage_dir(tmpdir):
    """
    Test that we correctly handle the `storage_dir` argument, which specifies where to download
    distributed artifacts passed to arguments of type `path`.
    """
    assert os.path.dirname(mlflow.projects._get_storage_dir(tmpdir.strpath)) == tmpdir.strpath
    assert os.path.dirname(mlflow.projects._get_storage_dir(None)) == tempfile.gettempdir()
