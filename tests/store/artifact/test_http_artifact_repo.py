import os
import posixpath
from unittest import mock

import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import (
    _TRACKING_CLIENT_CERT_PATH_ENV_VAR,
    _TRACKING_INSECURE_TLS_ENV_VAR,
    _TRACKING_PASSWORD_ENV_VAR,
    _TRACKING_SERVER_CERT_PATH_ENV_VAR,
    _TRACKING_TOKEN_ENV_VAR,
    _TRACKING_USERNAME_ENV_VAR,
    _get_default_host_creds,
)
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_artifact_uri_factory(scheme):
    repo = get_artifact_repository(f"{scheme}://test.com")
    assert isinstance(repo, HttpArtifactRepository)


class MockResponse:
    def __init__(self, data, status_code):
        self.data = data
        self.status_code = status_code

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("request failed")


class MockStreamResponse(MockResponse):
    def iter_content(self, chunk_size):  # pylint: disable=unused-argument
        yield self.data.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class FileObjectMatcher:
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode

    def __eq__(self, other):
        return self.name == other.name and self.mode == other.mode


@pytest.fixture
def http_artifact_repo():
    artifact_uri = "http://test.com/api/2.0/mlflow-artifacts/artifacts"
    return HttpArtifactRepository(artifact_uri)


@pytest.mark.parametrize(
    ("filename", "expected_mime_type"),
    [
        ("c.txt", "text/plain"),
        ("c.pkl", "application/octet-stream"),
        ("MLmodel", "text/plain"),
    ],
)
@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifact(http_artifact_repo, tmpdir, artifact_path, filename, expected_mime_type):
    tmp_path = tmpdir.join(filename)
    tmp_path.write("0")
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 200),
    ) as mock_put:
        http_artifact_repo.log_artifact(tmp_path, artifact_path)
        paths = (artifact_path, tmp_path.basename) if artifact_path else (tmp_path.basename,)
        mock_put.assert_called_once_with(
            http_artifact_repo._host_creds,
            posixpath.join("/", *paths),
            "PUT",
            data=FileObjectMatcher(tmp_path, "rb"),
            extra_headers={"Content-Type": expected_mime_type},
        )

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 400),
    ):
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo.log_artifact(tmp_path, artifact_path)


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifacts(http_artifact_repo, tmpdir, artifact_path):
    tmp_path_a = tmpdir.join("a.txt")
    tmp_path_b = tmpdir.mkdir("dir").join("b.txt")
    tmp_path_a.write("0")
    tmp_path_b.write("1")

    with mock.patch.object(http_artifact_repo, "log_artifact") as mock_log_artifact:
        http_artifact_repo.log_artifacts(tmpdir, artifact_path)
        mock_log_artifact.assert_has_calls(
            [
                mock.call(tmp_path_a.strpath, artifact_path),
                mock.call(
                    tmp_path_b.strpath,
                    posixpath.join(artifact_path, "dir") if artifact_path else "dir",
                ),
            ],
        )

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 400),
    ):
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo.log_artifacts(tmpdir, artifact_path)


def test_list_artifacts(http_artifact_repo):
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 200),
    ) as mock_get:
        assert http_artifact_repo.list_artifacts() == []
        endpoint = "/mlflow-artifacts/artifacts"
        url, _ = http_artifact_repo.artifact_uri.split(endpoint, maxsplit=1)
        mock_get.assert_called_once_with(
            _get_default_host_creds(url),
            endpoint,
            "GET",
            params={"path": ""},
        )

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse(
            {
                "files": [
                    {"path": "1.txt", "is_dir": False, "file_size": 1},
                    {"path": "dir", "is_dir": True},
                ]
            },
            200,
        ),
    ):
        assert [a.path for a in http_artifact_repo.list_artifacts()] == ["1.txt", "dir"]

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse(
            {
                "files": [
                    {"path": "1.txt", "is_dir": False, "file_size": 1},
                    {"path": "dir", "is_dir": True},
                ]
            },
            200,
        ),
    ):
        assert [a.path for a in http_artifact_repo.list_artifacts(path="path")] == [
            "path/1.txt",
            "path/dir",
        ]

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 400),
    ):
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo.list_artifacts()


def read_file(path):
    with open(path) as f:
        return f.read()


@pytest.mark.parametrize("remote_file_path", ["a.txt", "dir/b.xtx"])
def test_download_file(http_artifact_repo, tmpdir, remote_file_path):
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockStreamResponse("data", 200),
    ) as mock_get:
        tmp_path = tmpdir.join(posixpath.basename(remote_file_path))
        http_artifact_repo._download_file(remote_file_path, tmp_path)
        mock_get.assert_called_once_with(
            http_artifact_repo._host_creds,
            posixpath.join("/", remote_file_path),
            "GET",
            stream=True,
        )
        with open(tmp_path) as f:
            assert f.read() == "data"

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockStreamResponse("data", 400),
    ):
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo._download_file(remote_file_path, tmp_path)


def test_download_artifacts(http_artifact_repo, tmpdir):
    # This test simulates downloading artifacts in the following structure:
    # ---------
    # - a.txt
    # - dir
    #   - b.txt
    # ---------
    def http_request(_host_creds, endpoint, _method, **kwargs):
        # Responses for list_artifacts
        params = kwargs.get("params")
        if params:
            if params.get("path") == "":
                return MockResponse(
                    {
                        "files": [
                            {"path": "a.txt", "is_dir": False, "file_size": 1},
                            {"path": "dir", "is_dir": True},
                        ]
                    },
                    200,
                )
            elif params.get("path") == "dir":
                return MockResponse(
                    {
                        "files": [
                            {"path": "b.txt", "is_dir": False, "file_size": 1},
                        ]
                    },
                    200,
                )
            else:
                Exception("Unreachable")

        # Responses for _download_file
        if endpoint == "/a.txt":
            return MockStreamResponse("data_a", 200)
        elif endpoint == "/dir/b.txt":
            return MockStreamResponse("data_b", 200)
        else:
            raise Exception("Unreachable")

    with mock.patch("mlflow.store.artifact.http_artifact_repo.http_request", http_request):
        http_artifact_repo.download_artifacts("", tmpdir)
        paths = [os.path.join(root, f) for root, _, files in os.walk(tmpdir) for f in files]
        assert [os.path.relpath(p, tmpdir) for p in paths] == [
            "a.txt",
            os.path.join("dir", "b.txt"),
        ]
        assert read_file(paths[0]) == "data_a"
        assert read_file(paths[1]) == "data_b"


def test_default_host_creds():
    artifact_uri = "https://test.com"
    username = "user"
    password = "pass"
    token = "token"
    ignore_tls_verification = False
    client_cert_path = "client_cert_path"
    server_cert_path = "server_cert_path"

    expected_host_creds = MlflowHostCreds(
        host=artifact_uri,
        username=username,
        password=password,
        token=token,
        ignore_tls_verification=ignore_tls_verification,
        client_cert_path=client_cert_path,
        server_cert_path=server_cert_path,
    )

    repo = HttpArtifactRepository(artifact_uri)

    with mock.patch.dict(
        "mlflow.tracking._tracking_service.utils.os.environ",
        {
            _TRACKING_USERNAME_ENV_VAR: username,
            _TRACKING_PASSWORD_ENV_VAR: password,
            _TRACKING_TOKEN_ENV_VAR: token,
            _TRACKING_INSECURE_TLS_ENV_VAR: str(ignore_tls_verification),
            _TRACKING_CLIENT_CERT_PATH_ENV_VAR: client_cert_path,
            _TRACKING_SERVER_CERT_PATH_ENV_VAR: server_cert_path,
        },
    ):
        assert repo._host_creds == expected_host_creds


@pytest.mark.parametrize("remote_file_path", ["a.txt", "dir/b.txt", None])
def test_delete_artifacts(http_artifact_repo, remote_file_path):
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockStreamResponse("data", 200),
    ) as mock_get:
        http_artifact_repo.delete_artifacts(remote_file_path)
        mock_get.assert_called_once_with(
            http_artifact_repo._host_creds,
            posixpath.join("/", remote_file_path if remote_file_path else ""),
            "DELETE",
            stream=True,
        )
