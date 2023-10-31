import os
import posixpath
from unittest import mock

import pytest
from requests import HTTPError

from mlflow.entities.multipart_upload import (
    CreateMultipartUploadResponse,
    MultipartUploadCredential,
    MultipartUploadPart,
)
from mlflow.environment_variables import (
    MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE,
    MLFLOW_TRACKING_CLIENT_CERT_PATH,
    MLFLOW_TRACKING_INSECURE_TLS,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_SERVER_CERT_PATH,
    MLFLOW_TRACKING_TOKEN,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import _get_default_host_creds
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
def test_log_artifact(http_artifact_repo, tmp_path, artifact_path, filename, expected_mime_type):
    file_path = tmp_path.joinpath(filename)
    file_path.write_text("0")

    def assert_called_log_artifact(mock_http_request):
        paths = (artifact_path, file_path.name) if artifact_path else (file_path.name,)
        mock_http_request.assert_called_once_with(
            http_artifact_repo._host_creds,
            posixpath.join("/", *paths),
            "PUT",
            data=FileObjectMatcher(str(file_path), "rb"),
            extra_headers={"Content-Type": expected_mime_type},
        )

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 200),
    ) as mock_put:
        http_artifact_repo.log_artifact(file_path, artifact_path)
        assert_called_log_artifact(mock_put)

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 400),
    ):
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo.log_artifact(file_path, artifact_path)

    # assert mpu is triggered when file size is larger than minimum file size
    file_path.write_text("0" * MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get())
    with mock.patch.object(
        http_artifact_repo, "_try_multipart_upload", return_value=200
    ) as mock_mpu:
        http_artifact_repo.log_artifact(file_path, artifact_path)
        mock_mpu.assert_called_once()

    # assert reverted to normal upload when mpu is not supported
    # mock that create_multipart_upload will returns a 400 error with appropriate message
    with mock.patch.object(
        http_artifact_repo,
        "create_multipart_upload",
        side_effect=HTTPError(
            response=MockResponse(
                data={
                    "message": "Multipart upload is not supported for the current "
                    "artifact repository"
                },
                status_code=501,
            )
        ),
    ), mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 200),
    ) as mock_put:
        http_artifact_repo.log_artifact(file_path, artifact_path)
        assert_called_log_artifact(mock_put)

    # assert if mpu is triggered but the uploads failed, mpu is aborted and exception is raised
    with mock.patch("requests.put", side_effect=Exception("MPU_UPLOAD_FAILS")), mock.patch.object(
        http_artifact_repo,
        "create_multipart_upload",
        return_value=CreateMultipartUploadResponse(
            upload_id="upload_id",
            credentials=[MultipartUploadCredential(url="url", part_number=1, headers={})],
        ),
    ), mock.patch.object(
        http_artifact_repo,
        "abort_multipart_upload",
        return_value=None,
    ) as mock_abort:
        with pytest.raises(Exception, match="MPU_UPLOAD_FAILS"):
            http_artifact_repo.log_artifact(file_path, artifact_path)
        mock_abort.assert_called_once()


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifacts(http_artifact_repo, tmp_path, artifact_path):
    tmp_path_a = tmp_path.joinpath("a.txt")
    d = tmp_path.joinpath("dir")
    d.mkdir()
    tmp_path_b = d.joinpath("b.txt")
    tmp_path_a.write_text("0")
    tmp_path_b.write_text("1")

    with mock.patch.object(http_artifact_repo, "log_artifact") as mock_log_artifact:
        http_artifact_repo.log_artifacts(tmp_path, artifact_path)
        mock_log_artifact.assert_has_calls(
            [
                mock.call(str(tmp_path_a), artifact_path),
                mock.call(
                    str(tmp_path_b),
                    posixpath.join(artifact_path, "dir") if artifact_path else "dir",
                ),
            ],
        )

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 400),
    ):
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo.log_artifacts(tmp_path, artifact_path)


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
def test_download_file(http_artifact_repo, tmp_path, remote_file_path):
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockStreamResponse("data", 200),
    ) as mock_get:
        file_path = tmp_path.joinpath(posixpath.basename(remote_file_path))
        http_artifact_repo._download_file(remote_file_path, file_path)
        mock_get.assert_called_once_with(
            http_artifact_repo._host_creds,
            posixpath.join("/", remote_file_path),
            "GET",
            stream=True,
        )
        assert file_path.read_text() == "data"

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockStreamResponse("data", 400),
    ):
        with pytest.raises(Exception, match="request failed"):
            http_artifact_repo._download_file(remote_file_path, tmp_path)


def test_download_artifacts(http_artifact_repo, tmp_path):
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
        http_artifact_repo.download_artifacts("", tmp_path)
        paths = [os.path.join(root, f) for root, _, files in os.walk(tmp_path) for f in files]
        assert [os.path.relpath(p, tmp_path) for p in paths] == [
            "a.txt",
            os.path.join("dir", "b.txt"),
        ]
        assert read_file(paths[0]) == "data_a"
        assert read_file(paths[1]) == "data_b"


def test_default_host_creds(monkeypatch):
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

    monkeypatch.setenvs(
        {
            MLFLOW_TRACKING_USERNAME.name: username,
            MLFLOW_TRACKING_PASSWORD.name: password,
            MLFLOW_TRACKING_TOKEN.name: token,
            MLFLOW_TRACKING_INSECURE_TLS.name: str(ignore_tls_verification),
            MLFLOW_TRACKING_CLIENT_CERT_PATH.name: client_cert_path,
            MLFLOW_TRACKING_SERVER_CERT_PATH.name: server_cert_path,
        }
    )
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


def test_create_multipart_upload(http_artifact_repo):
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse(
            {
                "upload_id": "upload_id",
                "credentials": [
                    {
                        "url": "/some/url",
                        "part_number": 1,
                        "headers": {},
                    }
                ],
            },
            200,
        ),
    ):
        response = http_artifact_repo.create_multipart_upload("", 1)
        assert response.upload_id == "upload_id"
        assert len(response.credentials) == 1
        assert response.credentials[0].url == "/some/url"


def test_complete_multipart_upload(http_artifact_repo):
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 200),
    ) as mock_post:
        http_artifact_repo.complete_multipart_upload(
            local_file="local_file",
            upload_id="upload_id",
            parts=[
                MultipartUploadPart(part_number=1, etag="etag1"),
                MultipartUploadPart(part_number=2, etag="etag2"),
            ],
            artifact_path="artifact/path",
        )
        endpoint = "/mlflow-artifacts"
        url, _ = http_artifact_repo.artifact_uri.split(endpoint, maxsplit=1)
        mock_post.assert_called_once_with(
            _get_default_host_creds(url),
            "/mlflow-artifacts/mpu/complete/artifact/path",
            "POST",
            json={
                "path": "local_file",
                "upload_id": "upload_id",
                "parts": [
                    {"part_number": 1, "etag": "etag1"},
                    {"part_number": 2, "etag": "etag2"},
                ],
            },
        )


def test_abort_multipart_upload(http_artifact_repo):
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({}, 200),
    ) as mock_post:
        http_artifact_repo.abort_multipart_upload(
            local_file="local_file",
            upload_id="upload_id",
            artifact_path="artifact/path",
        )
        endpoint = "/mlflow-artifacts"
        url, _ = http_artifact_repo.artifact_uri.split(endpoint, maxsplit=1)
        mock_post.assert_called_once_with(
            _get_default_host_creds(url),
            "/mlflow-artifacts/mpu/abort/artifact/path",
            "POST",
            json={
                "path": "local_file",
                "upload_id": "upload_id",
            },
        )
