import os
import tempfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Generator
from unittest import mock

import git
import pytest

import mlflow
from mlflow.exceptions import ExecutionException
from mlflow.projects import _project_spec
from mlflow.projects.utils import (
    _fetch_git_repo,
    _fetch_project,
    _get_storage_dir,
    _is_valid_branch_name,
    _is_zip_uri,
    _parse_subdirectory,
    fetch_and_validate_project,
    get_or_create_run,
    load_project,
)
from mlflow.utils.mlflow_tags import MLFLOW_PROJECT_ENTRY_POINT, MLFLOW_SOURCE_NAME

from tests.helper_functions import get_safe_port
from tests.projects.utils import (
    GIT_PROJECT_BRANCH,
    GIT_PROJECT_URI,
    TEST_PROJECT_DIR,
    TEST_PROJECT_NAME,
    assert_dirs_equal,
)


class _SimpleHTTPServer(HTTPServer):
    def __init__(self, port: int) -> None:
        super().__init__(("127.0.0.1", port), self.RequestHandler)
        self.content = b""
        self._thread = None

    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.server.content)

    def serve_content(self, content: bytes) -> None:
        self.content = content

    @property
    def url(self) -> str:
        return f"http://{self.server_address[0]}:{self.server_address[1]}"

    def __enter__(self) -> "_SimpleHTTPServer":
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()
        self.server_close()
        if self._thread:
            self._thread.join(timeout=1)


@pytest.fixture
def httpserver() -> Generator[_SimpleHTTPServer, None, None]:
    with _SimpleHTTPServer(get_safe_port()) as server:
        yield server


def _build_uri(base_uri, subdirectory):
    if subdirectory != "":
        return f"{base_uri}#{subdirectory}"
    return base_uri


@pytest.fixture
def zipped_repo(tmp_path):
    zip_name = tmp_path.joinpath(f"{TEST_PROJECT_NAME}.zip")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(TEST_PROJECT_DIR):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                zip_file.write(file_path, file_path[len(TEST_PROJECT_DIR) + len(os.sep) :])
    return str(zip_name)


def test_is_zip_uri():
    assert _is_zip_uri("http://foo.bar/moo.zip")
    assert _is_zip_uri("https://foo.bar/moo.zip")
    assert _is_zip_uri("file:///moo.zip")
    assert _is_zip_uri("file://C:/moo.zip")
    assert _is_zip_uri("/moo.zip")
    assert _is_zip_uri("C:/moo.zip")
    assert not _is_zip_uri("http://foo.bar/moo")
    assert not _is_zip_uri("https://foo.bar/moo")
    assert not _is_zip_uri("file:///moo")
    assert not _is_zip_uri("file://C:/moo")
    assert not _is_zip_uri("/moo")
    assert not _is_zip_uri("C:/moo")


def test__fetch_project(local_git_repo, local_git_repo_uri, zipped_repo, httpserver):
    with open(zipped_repo, "rb") as f:
        httpserver.serve_content(f.read())
    # The tests are as follows:
    # 1. Fetching a locally saved project.
    # 2. Fetching a project located in a Git repo root directory.
    # 3. Fetching a project located in a Git repo subdirectory.
    # 4. Passing a subdirectory works for local directories.
    # 5. Fetching a remote ZIP file
    # 6. Using a local ZIP file
    # 7. Using a file:// URL to a local ZIP file
    test_list = [
        (TEST_PROJECT_DIR, "", TEST_PROJECT_DIR),
        (local_git_repo_uri, "", local_git_repo),
        (local_git_repo_uri, "example_project", os.path.join(local_git_repo, "example_project")),
        (os.path.dirname(TEST_PROJECT_DIR), os.path.basename(TEST_PROJECT_DIR), TEST_PROJECT_DIR),
        (httpserver.url + f"/{TEST_PROJECT_NAME}.zip", "", TEST_PROJECT_DIR),
        (zipped_repo, "", TEST_PROJECT_DIR),
        (f"file://{zipped_repo}", "", TEST_PROJECT_DIR),
    ]
    for base_uri, subdirectory, expected in test_list:
        work_dir = _fetch_project(uri=_build_uri(base_uri, subdirectory))
        assert_dirs_equal(expected=expected, actual=work_dir)
    # Test that we correctly determine the dest directory to use when fetching a project.
    fetched_local_project = _fetch_project(uri=TEST_PROJECT_DIR)
    assert os.path.abspath(fetched_local_project) == os.path.abspath(TEST_PROJECT_DIR)
    fetched_git_project = _fetch_project(GIT_PROJECT_URI)
    assert (
        os.path.commonprefix([fetched_git_project, tempfile.gettempdir()]) == tempfile.gettempdir()
    )
    assert os.path.exists(fetched_git_project)


@pytest.mark.parametrize(
    ("version", "expected_version"), [(None, "master"), (GIT_PROJECT_BRANCH, GIT_PROJECT_BRANCH)]
)
def test__fetch_git_repo(local_git_repo, local_git_repo_uri, version, expected_version):
    # Verify that the correct branch is checked out
    _fetch_git_repo(local_git_repo_uri, version, local_git_repo)
    repo = git.Repo(local_git_repo)
    assert repo.active_branch.name == expected_version


@pytest.mark.parametrize(
    "commit",
    ["0651d1c962aa35e4dd02608c51a7b0efc2412407", "3c0711f8868232f17a9adbb69fb1186ec8a3c0c7"],
)
def test_fetch_git_repo_commit(tmp_path, commit):
    _fetch_git_repo("https://github.com/mlflow/mlflow-example.git", commit, tmp_path)
    repo = git.Repo(tmp_path)
    assert repo.commit().hexsha == commit


def test_fetching_non_existing_version_fails(local_git_repo, local_git_repo_uri):
    with pytest.raises(ExecutionException, match="Unable to checkout"):
        _fetch_git_repo(local_git_repo_uri, "non-version", local_git_repo)


def test_fetch_project_validations(local_git_repo_uri):
    # Verify that runs fail if given incorrect subdirectories via the `#` character.
    for base_uri in [TEST_PROJECT_DIR, local_git_repo_uri]:
        with pytest.raises(ExecutionException, match="Could not find subdirectory fake"):
            _fetch_project(uri=_build_uri(base_uri, "fake"))

    # Passing `version` raises an exception for local projects
    with pytest.raises(ExecutionException, match="Setting a version is only supported"):
        _fetch_project(uri=TEST_PROJECT_DIR, version="version")


def test_dont_remove_mlruns(tmp_path):
    # Fetching a directory containing an "mlruns" folder doesn't remove the "mlruns" folder
    src_dir = tmp_path.joinpath("mlruns-src-dir")
    src_dir.mkdir()
    mlruns = src_dir.joinpath("mlruns")
    mlruns.mkdir()
    mlruns.joinpath("some-file.txt").write_text("hi")
    src_dir.joinpath("MLproject").write_text("dummy MLproject contents")
    dst_dir = _fetch_project(uri=str(src_dir), version=None)
    assert_dirs_equal(expected=str(src_dir), actual=dst_dir)


def test_parse_subdirectory():
    # Make sure the parsing works as intended.
    test_uri = "uri#subdirectory"
    parsed_uri, parsed_subdirectory = _parse_subdirectory(test_uri)
    assert parsed_uri == "uri"
    assert parsed_subdirectory == "subdirectory"

    # Make sure the parsing works with quotes.
    test_uri = "'uri#subdirectory'"
    parsed_uri, parsed_subdirectory = _parse_subdirectory(test_uri)
    assert parsed_uri == "uri"
    assert parsed_subdirectory == "subdirectory"

    # Make sure periods are restricted in Git repo subdirectory paths.
    period_fail_uri = GIT_PROJECT_URI + "#.."
    with pytest.raises(ExecutionException, match=r"'\.' is not allowed"):
        _parse_subdirectory(period_fail_uri)


def test_storage_dir(tmp_path):
    """
    Test that we correctly handle the `storage_dir` argument, which specifies where to download
    distributed artifacts passed to arguments of type `path`.
    """
    assert os.path.dirname(_get_storage_dir(tmp_path)) == str(tmp_path)
    assert os.path.dirname(_get_storage_dir(None)) == tempfile.gettempdir()


def test_is_valid_branch_name(local_git_repo):
    assert _is_valid_branch_name(local_git_repo, "master")
    assert not _is_valid_branch_name(local_git_repo, "dev")


def test_fetch_create_and_log(tmp_path):
    entry_point_name = "entry_point"
    parameters = {
        "method_name": "string",
    }
    entry_point = _project_spec.EntryPoint(entry_point_name, parameters, "run_model.sh")
    mock_fetched_project = _project_spec.Project(
        env_type="local",
        env_config_path=None,
        entry_points={entry_point_name: entry_point},
        docker_env=None,
        name="my_project",
    )
    experiment_id = mlflow.create_experiment("test_fetch_project")
    expected_dir = str(tmp_path)
    project_uri = "http://someuri/myproject.git"
    user_param = {"method_name": "newton"}
    with mock.patch("mlflow.projects.utils._fetch_project", return_value=expected_dir):
        with mock.patch(
            "mlflow.projects._project_spec.load_project", return_value=mock_fetched_project
        ):
            work_dir = fetch_and_validate_project("", "", entry_point_name, user_param)
            project = load_project(work_dir)
            assert mock_fetched_project == project
            assert expected_dir == work_dir
            # Create a run
            active_run = get_or_create_run(
                run_id=None,
                uri=project_uri,
                experiment_id=experiment_id,
                work_dir=work_dir,
                version=None,
                entry_point=entry_point_name,
                parameters=user_param,
            )

            # check tags
            run = mlflow.get_run(active_run.info.run_id)
            assert MLFLOW_PROJECT_ENTRY_POINT in run.data.tags
            assert MLFLOW_SOURCE_NAME in run.data.tags
            assert entry_point_name == run.data.tags[MLFLOW_PROJECT_ENTRY_POINT]
            assert project_uri == run.data.tags[MLFLOW_SOURCE_NAME]
            assert user_param == run.data.params
