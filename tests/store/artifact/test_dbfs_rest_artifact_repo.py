import json
import os
from unittest import mock
from unittest.mock import Mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.dbfs_artifact_repo import (
    DbfsRestArtifactRepository,
    _get_host_creds_from_default_store,
)
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds, http_request


@pytest.fixture
def dbfs_artifact_repo():
    with mock.patch(
        "mlflow.store.artifact.dbfs_artifact_repo._get_host_creds_from_default_store",
        return_value=lambda: MlflowHostCreds("http://host"),
    ):
        return get_artifact_repository("dbfs:/test/")


TEST_FILE_1_CONTENT = "Hello 🍆🍔".encode()
TEST_FILE_2_CONTENT = "World 🍆🍔🍆".encode()
TEST_FILE_3_CONTENT = "¡🍆🍆🍔🍆🍆!".encode()

DBFS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.dbfs_artifact_repo"
DBFS_ARTIFACT_REPOSITORY = DBFS_ARTIFACT_REPOSITORY_PACKAGE + ".DbfsRestArtifactRepository"


@pytest.fixture
def test_file(tmp_path):
    p = tmp_path.joinpath("test.txt")
    p.write_bytes(TEST_FILE_1_CONTENT)
    return str(p)


@pytest.fixture
def test_dir(tmp_path):
    subdir = tmp_path.joinpath("subdir")
    subdir.mkdir()
    subdir.joinpath("test.txt").write_bytes(TEST_FILE_2_CONTENT)
    tmp_path.joinpath("test.txt").write_bytes(bytes(TEST_FILE_3_CONTENT))
    tmp_path.joinpath("empty-file").touch()
    return str(tmp_path)


LIST_ARTIFACTS_RESPONSE = {
    "files": [
        {"path": "/test/a.txt", "is_dir": False, "file_size": 100},
        {"path": "/test/dir", "is_dir": True, "file_size": 0},
    ]
}

LIST_ARTIFACTS_SINGLE_FILE_RESPONSE = {
    "files": [{"path": "/test/a.txt", "is_dir": False, "file_size": 0}]
}

MOCK_SUCCESS_RESPONSE = Mock(status_code=200, text="{}")
MOCK_REQUEST_KWARGS = {
    "allow_redirects": False,
    "verify": True,
    "headers": mock.ANY,
    "timeout": mock.ANY,
    "data": mock.ANY,
}


def test_init_validation_and_cleaning():
    with mock.patch(
        DBFS_ARTIFACT_REPOSITORY_PACKAGE + "._get_host_creds_from_default_store",
        return_value=lambda: MlflowHostCreds("http://host"),
    ):
        repo = get_artifact_repository("dbfs:/test/")
        assert repo.artifact_uri == "dbfs:/test"
        match = "DBFS URI must be of the form dbfs:/<path>"
        with pytest.raises(MlflowException, match=match):
            DbfsRestArtifactRepository("s3://test")
        with pytest.raises(MlflowException, match=match):
            DbfsRestArtifactRepository("dbfs://profile@notdatabricks/test/")


def test_init_get_host_creds_with_databricks_profile_uri():
    databricks_host = "https://something.databricks.com"
    default_host = "http://host"
    with (
        mock.patch(
            DBFS_ARTIFACT_REPOSITORY_PACKAGE + "._get_host_creds_from_default_store",
            return_value=lambda: MlflowHostCreds(default_host),
        ),
        mock.patch(
            DBFS_ARTIFACT_REPOSITORY_PACKAGE + ".get_databricks_host_creds",
            return_value=MlflowHostCreds(databricks_host),
        ),
    ):
        repo = DbfsRestArtifactRepository("dbfs://profile@databricks/test/")
        assert repo.artifact_uri == "dbfs:/test/"
        creds = repo.get_host_creds()
        assert creds.host == databricks_host
        # no databricks_profile_uri given
        repo = DbfsRestArtifactRepository("dbfs:/test/")
        creds = repo.get_host_creds()
        assert creds.host == default_host


@pytest.mark.parametrize(
    ("artifact_path", "expected_endpoint"),
    [(None, "/dbfs/test/test.txt"), ("output", "/dbfs/test/output/test.txt")],
)
def test_log_artifact(dbfs_artifact_repo, test_file, artifact_path, expected_endpoint):
    with (
        mock.patch("mlflow.utils.rest_utils.http_request") as http_request_mock,
        mock.patch(
            "requests.Session.request", return_value=MOCK_SUCCESS_RESPONSE
        ) as mock_base_request,
    ):
        endpoints = []
        data = []

        def my_http_request(host_creds, **kwargs):
            endpoints.append(kwargs["endpoint"])
            data.append(kwargs["data"].read())
            return http_request(host_creds, **kwargs)

        http_request_mock.side_effect = my_http_request
        dbfs_artifact_repo.log_artifact(test_file, artifact_path)
        assert endpoints == [expected_endpoint]
        assert data == [TEST_FILE_1_CONTENT]
        mock_base_request.assert_called_once_with(
            "POST", f"http://host{expected_endpoint}", **MOCK_REQUEST_KWARGS
        )


def test_log_artifact_empty_file(dbfs_artifact_repo, test_dir):
    with (
        mock.patch("mlflow.utils.rest_utils.http_request") as http_request_mock,
        mock.patch(
            "requests.Session.request", return_value=MOCK_SUCCESS_RESPONSE
        ) as mock_base_request,
    ):

        def my_http_request(host_creds, **kwargs):
            assert kwargs["endpoint"] == "/dbfs/test/empty-file"
            assert kwargs["data"] == ""
            return http_request(host_creds, **kwargs)

        http_request_mock.side_effect = my_http_request
        dbfs_artifact_repo.log_artifact(os.path.join(test_dir, "empty-file"))
        mock_base_request.assert_called_once_with(
            "POST", "http://host/dbfs/test/empty-file", **MOCK_REQUEST_KWARGS
        )


def test_log_artifact_empty_artifact_path(dbfs_artifact_repo, test_file):
    with (
        mock.patch("mlflow.utils.rest_utils.http_request") as http_request_mock,
        mock.patch(
            "requests.Session.request", return_value=MOCK_SUCCESS_RESPONSE
        ) as mock_base_request,
    ):

        def my_http_request(host_creds, **kwargs):
            assert kwargs["endpoint"] == "/dbfs/test/test.txt"
            assert kwargs["data"].read() == TEST_FILE_1_CONTENT
            return http_request(host_creds, **kwargs)

        http_request_mock.side_effect = my_http_request
        dbfs_artifact_repo.log_artifact(test_file, "")
        mock_base_request.assert_called_once_with(
            "POST", "http://host/dbfs/test/test.txt", **MOCK_REQUEST_KWARGS
        )


def test_log_artifact_error(dbfs_artifact_repo, test_file):
    with mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=Mock(status_code=409, text="")
    ):
        with pytest.raises(MlflowException, match=r"API request to endpoint .+ failed"):
            dbfs_artifact_repo.log_artifact(test_file)


@pytest.mark.parametrize(
    "artifact_path",
    [
        None,
        "",  # should behave like '/' and exclude base name of logged_dir
        # We should add '.',
    ],
)
def test_log_artifacts(dbfs_artifact_repo, test_dir, artifact_path):
    with (
        mock.patch("mlflow.utils.rest_utils.http_request") as http_request_mock,
        mock.patch(
            "requests.Session.request", return_value=MOCK_SUCCESS_RESPONSE
        ) as mock_base_request,
    ):
        endpoints = []
        data = []

        def my_http_request(host_creds, **kwargs):
            endpoints.append(kwargs["endpoint"])
            if kwargs["endpoint"] == "/dbfs/test/empty-file":
                data.append(kwargs["data"])
            else:
                data.append(kwargs["data"].read())
            return http_request(host_creds, **kwargs)

        http_request_mock.side_effect = my_http_request
        dbfs_artifact_repo.log_artifacts(test_dir, artifact_path)
        assert set(endpoints) == {
            "/dbfs/test/subdir/test.txt",
            "/dbfs/test/test.txt",
            "/dbfs/test/empty-file",
        }
        assert set(data) == {
            TEST_FILE_2_CONTENT,
            TEST_FILE_3_CONTENT,
            "",
        }

        mock_calls = [
            mock.call("POST", "http://host/dbfs/test/empty-file", **MOCK_REQUEST_KWARGS),
            mock.call("POST", "http://host/dbfs/test/test.txt", **MOCK_REQUEST_KWARGS),
            mock.call("POST", "http://host/dbfs/test/subdir/test.txt", **MOCK_REQUEST_KWARGS),
        ]
        mock_base_request.assert_has_calls(mock_calls, any_order=True)


def test_log_artifacts_error(dbfs_artifact_repo, test_dir):
    with mock.patch(
        "mlflow.utils.rest_utils.http_request", return_value=Mock(status_code=409, text="")
    ):
        with pytest.raises(MlflowException, match=r"API request to endpoint .+ failed"):
            dbfs_artifact_repo.log_artifacts(test_dir)


@pytest.mark.parametrize(
    ("artifact_path", "expected_endpoints"),
    [
        (
            "a",
            {
                "/dbfs/test/a/subdir/test.txt",
                "/dbfs/test/a/test.txt",
                "/dbfs/test/a/empty-file",
            },
        ),
        (
            "a/",
            {
                "/dbfs/test/a/subdir/test.txt",
                "/dbfs/test/a/test.txt",
                "/dbfs/test/a/empty-file",
            },
        ),
        ("/", {"/dbfs/test/subdir/test.txt", "/dbfs/test/test.txt", "/dbfs/test/empty-file"}),
    ],
)
def test_log_artifacts_with_artifact_path(
    dbfs_artifact_repo, test_dir, artifact_path, expected_endpoints
):
    with (
        mock.patch("mlflow.utils.rest_utils.http_request") as http_request_mock,
        mock.patch(
            "requests.Session.request", return_value=MOCK_SUCCESS_RESPONSE
        ) as mock_base_request,
    ):
        endpoints = []

        def my_http_request(host_creds, **kwargs):
            endpoints.append(kwargs["endpoint"])
            return http_request(host_creds, **kwargs)

        http_request_mock.side_effect = my_http_request
        dbfs_artifact_repo.log_artifacts(test_dir, artifact_path)
        assert set(endpoints) == expected_endpoints
        mock_calls = [
            mock.call("POST", f"http://host{endpoint}", **MOCK_REQUEST_KWARGS)
            for endpoint in expected_endpoints
        ]
        mock_base_request.assert_has_calls(mock_calls, any_order=True)


def test_list_artifacts(dbfs_artifact_repo):
    with mock.patch("requests.Session.request") as mock_base_request:
        mock_base_request.return_value = Mock(text=json.dumps(LIST_ARTIFACTS_RESPONSE))
        artifacts = dbfs_artifact_repo.list_artifacts()
        assert len(artifacts) == 2
        assert artifacts[0].path == "a.txt"
        assert artifacts[0].is_dir is False
        assert artifacts[0].file_size == 100
        assert artifacts[1].path == "dir"
        assert artifacts[1].is_dir is True
        assert artifacts[1].file_size is None
        mock_base_request.assert_called_with(
            "GET",
            "http://host/api/2.0/dbfs/list",
            allow_redirects=True,
            headers=mock.ANY,
            verify=True,
            timeout=mock.ANY,
            params={"path": "/test/"},
        )
        # Calling list_artifacts() on a path that's a file should return an empty list
        mock_base_request.return_value.text = json.dumps(LIST_ARTIFACTS_SINGLE_FILE_RESPONSE)
        list_on_file = dbfs_artifact_repo.list_artifacts("a.txt")
        assert len(list_on_file) == 0
        mock_base_request.assert_called_with(
            "GET",
            "http://host/api/2.0/dbfs/list",
            allow_redirects=True,
            headers=mock.ANY,
            verify=True,
            timeout=mock.ANY,
            params={"path": "/test/a.txt"},
        )


def test_download_artifacts(dbfs_artifact_repo):
    with (
        mock.patch(DBFS_ARTIFACT_REPOSITORY + "._dbfs_is_dir") as is_dir_mock,
        mock.patch(DBFS_ARTIFACT_REPOSITORY + "._dbfs_list_api") as list_mock,
        mock.patch(DBFS_ARTIFACT_REPOSITORY + "._dbfs_download") as download_mock,
    ):
        is_dir_mock.side_effect = [
            True,
            False,
            True,
        ]
        list_mock.side_effect = [
            Mock(text=json.dumps(LIST_ARTIFACTS_RESPONSE)),
            Mock(text="{}"),  # this call is for listing `/dir`.
            Mock(text="{}"),  # this call is for listing `/dir/a.txt`.
        ]
        dbfs_artifact_repo.download_artifacts("/")
        assert list_mock.call_count == 2
        assert download_mock.call_count == 1
        chronological_download_calls = list(download_mock.call_args_list)
        # Calls are in reverse chronological order by default
        chronological_download_calls.reverse()
        _, kwargs_call = chronological_download_calls[0]
        assert kwargs_call["endpoint"] == "/dbfs/test/a.txt"


def test_get_host_creds_from_default_store_file_store():
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=FileStore()):
        with pytest.raises(MlflowException, match="Failed to get credentials for DBFS"):
            _get_host_creds_from_default_store()


def test_get_host_creds_from_default_store_rest_store():
    with mock.patch(
        "mlflow.tracking._tracking_service.utils._get_store",
        return_value=RestStore(lambda: MlflowHostCreds("http://host")),
    ):
        assert isinstance(_get_host_creds_from_default_store()(), MlflowHostCreds)
