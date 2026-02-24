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
from mlflow.entities.presigned_download import PresignedDownloadUrlResponse
from mlflow.environment_variables import (
    MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE,
    MLFLOW_TRACKING_CLIENT_CERT_PATH,
    MLFLOW_TRACKING_INSECURE_TLS,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_SERVER_CERT_PATH,
    MLFLOW_TRACKING_TOKEN,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.exceptions import MlflowException, _UnsupportedMultipartUploadException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository
from mlflow.utils.credentials import get_default_host_creds
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
    def iter_content(self, chunk_size):
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
    repo = HttpArtifactRepository(artifact_uri)
    # Pre-seed the cached property to avoid unexpected server-info HTTP calls in tests.
    repo.__dict__["_server_enforcement"] = {
        "enforce_proxy_multipart_upload": False,
        "enforce_proxy_multipart_download": False,
    }
    return repo


@pytest.fixture
def mlflow_artifact_repo_for_download():
    """MlflowArtifactsRepository with same effective URI as http_artifact_repo.

    For multipart download tests.
    """
    repo = MlflowArtifactsRepository(
        artifact_uri="mlflow-artifacts:/",
        tracking_uri="http://test.com",
    )
    repo.__dict__["_server_enforcement"] = {
        "enforce_proxy_multipart_upload": False,
        "enforce_proxy_multipart_download": False,
    }
    return repo


@pytest.mark.parametrize(
    ("filename", "expected_mime_type"),
    [
        ("c.txt", "text/plain"),
        ("c.pkl", "application/octet-stream"),
        ("MLmodel", "text/plain"),
    ],
)
@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifact(
    http_artifact_repo,
    tmp_path,
    artifact_path,
    filename,
    expected_mime_type,
    monkeypatch,
):
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

    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
    # assert mpu is triggered when file size is larger than minimum file size
    file_path.write_text("0" * MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get())
    with mock.patch.object(
        http_artifact_repo, "_try_multipart_upload", return_value=200
    ) as mock_mpu:
        http_artifact_repo.log_artifact(file_path, artifact_path)
        mock_mpu.assert_called_once()

    # assert unsupported multipart raises when multipart is enabled (no fallback)
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
    ):
        with pytest.raises(_UnsupportedMultipartUploadException):
            http_artifact_repo.log_artifact(file_path, artifact_path)

    # assert if mpu is triggered but the uploads failed, mpu is aborted and exception is raised
    with (
        mock.patch("requests.put", side_effect=Exception("MPU_UPLOAD_FAILS")),
        mock.patch.object(
            http_artifact_repo,
            "create_multipart_upload",
            return_value=CreateMultipartUploadResponse(
                upload_id="upload_id",
                credentials=[MultipartUploadCredential(url="url", part_number=1, headers={})],
            ),
        ),
        mock.patch.object(
            http_artifact_repo,
            "abort_multipart_upload",
            return_value=None,
        ) as mock_abort,
    ):
        with pytest.raises(Exception, match="MPU_UPLOAD_FAILS"):
            http_artifact_repo.log_artifact(file_path, artifact_path)
        mock_abort.assert_called_once()


def test_log_artifact_small_file_uses_multipart_when_enabled(
    http_artifact_repo, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")

    file_path = tmp_path / "tiny.txt"
    file_path.write_text("x")  # 1 byte — well below minimum file size threshold

    with mock.patch.object(
        http_artifact_repo, "_try_multipart_upload"
    ) as mock_mpu:
        http_artifact_repo.log_artifact(file_path)
        mock_mpu.assert_called_once()


def test_log_artifact_no_fallback_when_multipart_enabled(
    http_artifact_repo, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")

    file_path = tmp_path / "file.bin"
    file_path.write_bytes(b"x" * MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get())

    with (
        mock.patch.object(
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
        ),
        mock.patch(
            "mlflow.store.artifact.http_artifact_repo.http_request",
        ) as mock_http_request,
    ):
        with pytest.raises(_UnsupportedMultipartUploadException):
            http_artifact_repo.log_artifact(file_path)

        mock_http_request.assert_not_called()


def test_log_artifact_empty_file_uses_multipart_when_enabled(
    http_artifact_repo, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")

    file_path = tmp_path / "empty.txt"
    file_path.write_bytes(b"")

    with mock.patch.object(
        http_artifact_repo, "_try_multipart_upload"
    ) as mock_mpu:
        http_artifact_repo.log_artifact(file_path)
        mock_mpu.assert_called_once()


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
            get_default_host_creds(url),
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


@pytest.mark.parametrize("path", ["/tmp/path", "../../path", "%2E%2E%2Fpath"])
def test_list_artifacts_malicious_path(http_artifact_repo, path):
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse(
            {
                "files": [
                    {"path": path, "is_dir": False, "file_size": 1},
                ]
            },
            200,
        ),
    ):
        with pytest.raises(MlflowException, match="Invalid path"):
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
        if params := kwargs.get("params"):
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

    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, username)
    monkeypatch.setenv(MLFLOW_TRACKING_PASSWORD.name, password)
    monkeypatch.setenv(MLFLOW_TRACKING_TOKEN.name, token)
    monkeypatch.setenv(MLFLOW_TRACKING_INSECURE_TLS.name, str(ignore_tls_verification))
    monkeypatch.setenv(MLFLOW_TRACKING_CLIENT_CERT_PATH.name, client_cert_path)
    monkeypatch.setenv(MLFLOW_TRACKING_SERVER_CERT_PATH.name, server_cert_path)
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
            posixpath.join("/", remote_file_path or ""),
            "DELETE",
            stream=True,
        )


def test_create_multipart_upload(http_artifact_repo, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
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


def test_complete_multipart_upload(http_artifact_repo, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
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
            get_default_host_creds(url),
            "/mlflow-artifacts/mpu/complete/artifact/path",
            "POST",
            json={
                "path": "local_file",
                "upload_id": "upload_id",
                "parts": [
                    {"part_number": 1, "etag": "etag1", "url": None},
                    {"part_number": 2, "etag": "etag2", "url": None},
                ],
            },
        )


def test_abort_multipart_upload(http_artifact_repo, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
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
            get_default_host_creds(url),
            "/mlflow-artifacts/mpu/abort/artifact/path",
            "POST",
            json={
                "path": "local_file",
                "upload_id": "upload_id",
            },
        )


# Tests for multipart download functionality


def test_download_file_multipart_for_large_files(
    mlflow_artifact_repo_for_download, tmp_path, monkeypatch
):
    """Test that multipart download is used when presigned URL is available.

    Uses MlflowArtifactsRepository.
    """
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "true")

    remote_file_path = "large_file.bin"
    presigned_url = "https://s3.amazonaws.com/bucket/large_file.bin?signature=abc"
    file_size = 2000

    with (
        mock.patch.object(
            mlflow_artifact_repo_for_download,
            "_get_presigned_download_url",
            return_value=PresignedDownloadUrlResponse(
                url=presigned_url, headers={}, file_size=file_size
            ),
        ) as mock_get_presigned,
        mock.patch.object(
            mlflow_artifact_repo_for_download, "_multipart_download"
        ) as mock_multipart_download,
    ):
        file_path = tmp_path / "large_file.bin"
        mlflow_artifact_repo_for_download._download_file(remote_file_path, str(file_path))

        mock_get_presigned.assert_called_once_with(remote_file_path)
        mock_multipart_download.assert_called_once()
        call_args = mock_multipart_download.call_args
        assert call_args.kwargs["presigned_response"].url == presigned_url
        assert call_args.kwargs["file_size"] == file_size


def test_download_file_small_file_uses_multipart(
    mlflow_artifact_repo_for_download, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "true")

    remote_file_path = "small_file.txt"
    presigned_url = "https://s3.amazonaws.com/bucket/small_file.txt?signature=abc"
    file_size = 100

    with (
        mock.patch.object(
            mlflow_artifact_repo_for_download,
            "_get_presigned_download_url",
            return_value=PresignedDownloadUrlResponse(
                url=presigned_url, headers={}, file_size=file_size
            ),
        ),
        mock.patch.object(
            mlflow_artifact_repo_for_download, "_multipart_download"
        ) as mock_multipart_download,
    ):
        file_path = tmp_path / "small_file.txt"
        mlflow_artifact_repo_for_download._download_file(remote_file_path, str(file_path))

        mock_multipart_download.assert_called_once()
        assert mock_multipart_download.call_args.kwargs["file_size"] == file_size


def test_download_file_multipart_success_does_not_call_proxy_download(
    mlflow_artifact_repo_for_download, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "true")

    remote_file_path = "file.bin"
    presigned_url = "https://s3.amazonaws.com/bucket/file.bin?signature=abc"
    file_size = 500

    with (
        mock.patch.object(
            mlflow_artifact_repo_for_download,
            "_get_presigned_download_url",
            return_value=PresignedDownloadUrlResponse(
                url=presigned_url, headers={}, file_size=file_size
            ),
        ),
        mock.patch.object(mlflow_artifact_repo_for_download, "_multipart_download"),
        mock.patch.object(HttpArtifactRepository, "_download_file") as mock_parent_download,
    ):
        file_path = tmp_path / "file.bin"
        mlflow_artifact_repo_for_download._download_file(remote_file_path, str(file_path))

        mock_parent_download.assert_not_called()


def test_download_file_raises_when_presigned_not_supported(
    mlflow_artifact_repo_for_download, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "true")

    remote_file_path = "test.txt"

    mock_response = mock.MagicMock()
    mock_response.status_code = 501
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "message": "Presigned URL download is not supported for the current artifact repository"
    }

    with mock.patch.object(
        mlflow_artifact_repo_for_download,
        "_get_presigned_download_url",
        side_effect=HTTPError(response=mock_response),
    ):
        file_path = tmp_path / "test.txt"
        with pytest.raises(HTTPError):
            mlflow_artifact_repo_for_download._download_file(
                remote_file_path, str(file_path)
            )


def test_download_file_no_fallback_when_multipart_enabled(
    mlflow_artifact_repo_for_download, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "true")
    remote_file_path = "test.txt"
    mock_response = mock.MagicMock()
    mock_response.status_code = 501
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "message": "Presigned URL download is not supported"
    }
    with (
        mock.patch.object(
            mlflow_artifact_repo_for_download,
            "_get_presigned_download_url",
            side_effect=HTTPError(response=mock_response),
        ),
        mock.patch(
            "mlflow.store.artifact.http_artifact_repo.http_request",
        ) as mock_http_request,
    ):
        file_path = tmp_path / "test.txt"
        with pytest.raises(HTTPError):
            mlflow_artifact_repo_for_download._download_file(
                remote_file_path, str(file_path)
            )
        mock_http_request.assert_not_called()


def test_download_file_multipart_disabled_uses_proxy(
    mlflow_artifact_repo_for_download, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")

    remote_file_path = "test.txt"

    with (
        mock.patch.object(
            mlflow_artifact_repo_for_download, "_get_presigned_download_url"
        ) as mock_get_presigned,
        mock.patch(
            "mlflow.store.artifact.http_artifact_repo.http_request",
            return_value=MockStreamResponse("data", 200),
        ) as mock_http_request,
    ):
        file_path = tmp_path / "test.txt"
        mlflow_artifact_repo_for_download._download_file(remote_file_path, str(file_path))

        mock_get_presigned.assert_not_called()
        mock_http_request.assert_called_once()


def test_multipart_download_creates_chunks(http_artifact_repo, tmp_path, monkeypatch):
    chunk_size = 100
    monkeypatch.setenv("MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE", str(chunk_size))

    remote_file_path = "large_file.bin"
    presigned_url = "https://s3.amazonaws.com/bucket/large_file.bin"
    file_size = 250  # Will create 3 chunks: 0-99, 100-199, 200-249
    headers = {"x-amz-header": "value"}

    presigned_response = PresignedDownloadUrlResponse(
        url=presigned_url, headers=headers, file_size=file_size
    )

    download_chunk_calls = []

    def mock_download_chunk(**kwargs):
        download_chunk_calls.append((kwargs["range_start"], kwargs["range_end"]))

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.download_chunk",
        side_effect=mock_download_chunk,
    ):
        file_path = tmp_path / "large_file.bin"
        http_artifact_repo._multipart_download(
            presigned_response=presigned_response,
            remote_file_path=remote_file_path,
            local_path=str(file_path),
            file_size=file_size,
            chunk_size=chunk_size,
        )

    # Should have downloaded 3 chunks
    assert len(download_chunk_calls) == 3
    # Sort by range_start to verify ranges
    sorted_calls = sorted(download_chunk_calls, key=lambda x: x[0])
    assert sorted_calls[0] == (0, 99)
    assert sorted_calls[1] == (100, 199)
    assert sorted_calls[2] == (200, 249)


def test_get_presigned_download_url(http_artifact_repo):
    remote_file_path = "artifacts/model.pkl"
    expected_url = "https://s3.amazonaws.com/bucket/model.pkl?signature=abc"
    expected_headers = {"x-amz-header": "value"}
    expected_file_size = 12345

    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse(
            {
                "url": expected_url,
                "headers": expected_headers,
                "file_size": expected_file_size,
            },
            200,
        ),
    ) as mock_request:
        response = http_artifact_repo._get_presigned_download_url(remote_file_path)

        assert response.url == expected_url
        assert response.headers == expected_headers
        assert response.file_size == expected_file_size

        # Verify correct endpoint was called (presigned is at API 2.0)
        endpoint = "/mlflow-artifacts"
        url, _ = http_artifact_repo.artifact_uri.split(endpoint, maxsplit=1)
        mock_request.assert_called_once_with(
            get_default_host_creds(url),
            f"/mlflow-artifacts/presigned/{remote_file_path}",
            "GET",
        )


# Tests for server enforcement auto-enable


def test_should_use_multipart_upload_true_when_env_var_set(http_artifact_repo, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
    assert http_artifact_repo._should_use_multipart_upload() is True


def test_should_use_multipart_upload_true_when_server_enforces(http_artifact_repo):
    http_artifact_repo.__dict__["_server_enforcement"] = {
        "enforce_proxy_multipart_upload": True,
        "enforce_proxy_multipart_download": False,
    }
    assert http_artifact_repo._should_use_multipart_upload() is True


def test_should_use_multipart_upload_false_by_default(http_artifact_repo):
    http_artifact_repo.__dict__["_server_enforcement"] = {
        "enforce_proxy_multipart_upload": False,
        "enforce_proxy_multipart_download": False,
    }
    assert http_artifact_repo._should_use_multipart_upload() is False


def test_server_enforcement_fetches_from_server_info(http_artifact_repo):
    # Clear pre-seeded cache to test actual fetch behavior.
    del http_artifact_repo.__dict__["_server_enforcement"]
    response_data = {
        "enforce_proxy_multipart_upload": True,
        "enforce_proxy_multipart_download": True,
    }
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse(response_data, 200),
    ) as mock_request:
        result = http_artifact_repo._server_enforcement
        assert result["enforce_proxy_multipart_upload"] is True
        assert result["enforce_proxy_multipart_download"] is True
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0].host == "http://test.com"
        assert call_args[0][1] == "/api/3.0/mlflow/server-info"
        assert call_args[0][2] == "GET"
        assert call_args[1]["timeout"] == 3


def test_server_enforcement_defaults_on_failure(http_artifact_repo):
    del http_artifact_repo.__dict__["_server_enforcement"]
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        side_effect=Exception("connection failed"),
    ) as mock_request:
        result = http_artifact_repo._server_enforcement
        assert result["enforce_proxy_multipart_upload"] is False
        assert result["enforce_proxy_multipart_download"] is False
        mock_request.assert_called_once()


def test_server_enforcement_defaults_on_old_server(http_artifact_repo):
    del http_artifact_repo.__dict__["_server_enforcement"]
    with mock.patch(
        "mlflow.store.artifact.http_artifact_repo.http_request",
        return_value=MockResponse({"store_type": "SqlStore"}, 200),
    ) as mock_request:
        result = http_artifact_repo._server_enforcement
        assert result["enforce_proxy_multipart_upload"] is False
        assert result["enforce_proxy_multipart_download"] is False
        mock_request.assert_called_once()


# Tests for download auto-enable via server enforcement


def test_download_file_auto_enables_multipart_when_server_enforces(monkeypatch):
    monkeypatch.delenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", raising=False)

    repo = MlflowArtifactsRepository(
        artifact_uri="mlflow-artifacts:/",
        tracking_uri="http://localhost:5000",
    )
    repo.__dict__["_server_enforcement"] = {
        "enforce_proxy_multipart_upload": False,
        "enforce_proxy_multipart_download": True,
    }

    presigned_response = PresignedDownloadUrlResponse(
        url="https://s3.amazonaws.com/bucket/key?sig=abc",
        headers={},
        file_size=1000,
    )

    with (
        mock.patch.object(
            repo, "_get_presigned_download_url", return_value=presigned_response
        ) as mock_presigned,
        mock.patch.object(repo, "_multipart_download") as mock_multipart,
        mock.patch(
            "mlflow.store.artifact.http_artifact_repo.http_request",
            return_value=MockStreamResponse("data", 200),
        ),
    ):
        repo._download_file("artifact.bin", "/tmp/artifact.bin")
        mock_presigned.assert_called_once_with("artifact.bin")
        mock_multipart.assert_called_once()


def test_download_file_no_multipart_when_not_enforced_and_env_var_off(monkeypatch):
    monkeypatch.delenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", raising=False)

    repo = MlflowArtifactsRepository(
        artifact_uri="mlflow-artifacts:/",
        tracking_uri="http://localhost:5000",
    )
    repo.__dict__["_server_enforcement"] = {
        "enforce_proxy_multipart_upload": False,
        "enforce_proxy_multipart_download": False,
    }

    with (
        mock.patch.object(repo, "_get_presigned_download_url") as mock_presigned,
        mock.patch(
            "mlflow.store.artifact.http_artifact_repo.http_request",
            return_value=MockStreamResponse("data", 200),
        ),
    ):
        repo._download_file("artifact.bin", "/tmp/artifact.bin")
        mock_presigned.assert_not_called()
