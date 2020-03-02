import mock
import os
import tempfile
import pytest

from mlflow import tracking
from mlflow.projects import utils
from mlflow.entities import run, run_status
from mlflow.exceptions import ExecutionException
from tests.projects.utils import TEST_PROJECT_DIR, TEST_PROJECT_NAME, GIT_PROJECT_URI, \
    assert_dirs_equal


def test_gen_envvars_from_run():
    run_info = run.RunInfo(run_uuid="my_run", experiment_id=23, user_id="j.doe",
                           status=run_status.RunStatus.RUNNING, start_time=0,
                           end_time=1, lifecycle_stage=None)
    current_run = run.Run(run_info=run_info, run_data=None)
    with mock.patch('mlflow.tracking.get_tracking_uri', return_value="https://my_mlflow:5000"):
        assert {
            tracking._RUN_ID_ENV_VAR: "my_run",
            tracking._TRACKING_URI_ENV_VAR: "https://my_mlflow:5000",
            tracking._EXPERIMENT_ID_ENV_VAR: "23",
        } == utils.generate_env_vars_to_attach_to_run(current_run)


def _build_uri(base_uri, subdirectory):
    if subdirectory != "":
        return "%s#%s" % (base_uri, subdirectory)
    return base_uri


def test_dont_remove_mlruns(tmpdir):
    # Fetching a directory containing an "mlruns" folder doesn't remove the "mlruns" folder
    src_dir = tmpdir.mkdir("mlruns-src-dir")
    src_dir.mkdir("mlruns").join("some-file.txt").write("hi")
    src_dir.join("MLproject").write("dummy MLproject contents")
    dst_dir = utils.fetch_project(uri=src_dir.strpath, version=None,
                                  force_tempdir=False)
    assert_dirs_equal(expected=src_dir.strpath, actual=dst_dir)


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
        work_dir = utils.fetch_project(
            uri=_build_uri(base_uri, subdirectory), force_tempdir=False)
        assert_dirs_equal(expected=expected, actual=work_dir)
    # Test that we correctly determine the dest directory to use when fetching a project.
    for force_tempdir, uri in [(True, TEST_PROJECT_DIR), (False, GIT_PROJECT_URI)]:
        dest_dir = utils.fetch_project(uri=uri, force_tempdir=force_tempdir)
        assert os.path.commonprefix([dest_dir, tempfile.gettempdir()]) == tempfile.gettempdir()
        assert os.path.exists(dest_dir)
    for force_tempdir, uri in [(None, TEST_PROJECT_DIR), (False, TEST_PROJECT_DIR)]:
        assert utils.fetch_project(uri=uri, force_tempdir=force_tempdir) == \
            os.path.abspath(TEST_PROJECT_DIR)


def test_fetch_project_validations(local_git_repo_uri):
    # Verify that runs fail if given incorrect subdirectories via the `#` character.
    for base_uri in [TEST_PROJECT_DIR, local_git_repo_uri]:
        with pytest.raises(ExecutionException):
            utils.fetch_project(uri=_build_uri(base_uri, "fake"), force_tempdir=False)

    # Passing `version` raises an exception for local projects
    with pytest.raises(ExecutionException):
        utils.fetch_project(uri=TEST_PROJECT_DIR, force_tempdir=False, version="version")


def test_parse_subdirectory():
    # Make sure the parsing works as intended.
    test_uri = "uri#subdirectory"
    parsed_uri, parsed_subdirectory = utils._parse_subdirectory(test_uri)
    assert parsed_uri == "uri"
    assert parsed_subdirectory == "subdirectory"

    # Make sure periods are restricted in Git repo subdirectory paths.
    period_fail_uri = GIT_PROJECT_URI + "#.."
    with pytest.raises(ExecutionException):
        utils._parse_subdirectory(period_fail_uri)


def test_is_zip_uri():
    assert utils._is_zip_uri('http://foo.bar/moo.zip')
    assert utils._is_zip_uri('https://foo.bar/moo.zip')
    assert utils._is_zip_uri('file:///moo.zip')
    assert utils._is_zip_uri('file://C:/moo.zip')
    assert utils._is_zip_uri('/moo.zip')
    assert utils._is_zip_uri('C:/moo.zip')
    assert not utils._is_zip_uri('http://foo.bar/moo')
    assert not utils._is_zip_uri('https://foo.bar/moo')
    assert not utils._is_zip_uri('file:///moo')
    assert not utils._is_zip_uri('file://C:/moo')
    assert not utils._is_zip_uri('/moo')
    assert not utils._is_zip_uri('C:/moo')


@pytest.fixture
def zipped_repo(tmpdir):
    import zipfile
    zip_name = tmpdir.join('%s.zip' % TEST_PROJECT_NAME).strpath
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(TEST_PROJECT_DIR):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                zip_file.write(file_path, file_path[len(TEST_PROJECT_DIR) + len(os.sep):])
    return zip_name
