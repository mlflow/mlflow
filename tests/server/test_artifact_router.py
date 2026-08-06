from unittest import mock

import pytest
from starlette.testclient import TestClient

from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR
from mlflow.server import app as flask_app
from mlflow.server.fastapi_app import create_fastapi_app
from mlflow.store.artifact.artifact_repo import ARTIFACT_STREAM_CHUNK_SIZE, ArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, "/tmp/mlflow-artifacts-test")
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    return TestClient(create_fastapi_app())


@pytest.fixture
def client_no_serve(monkeypatch):
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "false")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, "/tmp/mlflow-artifacts-test")
    monkeypatch.setenv("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")
    return TestClient(create_fastapi_app())


def test_download_local_path_returns_file_response(client, tmp_path):
    test_data = b"local artifact content"
    test_file = tmp_path / "model.pkl"
    test_file.write_bytes(test_data)

    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock()
        mock_repo.get_local_path.return_value = str(test_file)
        mock_get_repo.return_value = mock_repo

        resp = client.get("/api/2.0/mlflow-artifacts/artifacts/test/model.pkl")

    assert resp.status_code == 200
    assert resp.content == test_data
    assert "attachment" in resp.headers["Content-Disposition"]
    assert "model.pkl" in resp.headers["Content-Disposition"]
    mock_repo.get_local_path.assert_called_once()
    mock_repo.download_artifacts.assert_not_called()


def test_download_remote_path_returns_streaming_response(client, tmp_path):
    test_data = b"remote artifact content"
    test_file = tmp_path / "model.pkl"
    test_file.write_bytes(test_data)

    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock()
        mock_repo.get_local_path.return_value = None
        mock_repo.download_artifacts.return_value = str(test_file)
        mock_get_repo.return_value = mock_repo

        resp = client.get("/api/2.0/mlflow-artifacts/artifacts/nested/model.pkl")

    assert resp.status_code == 200
    assert resp.content == test_data
    assert "attachment" in resp.headers["Content-Disposition"]
    mock_repo.download_artifacts.assert_called_once()


def test_download_ajax_prefix_also_works(client, tmp_path):
    test_file = tmp_path / "data.bin"
    test_file.write_bytes(b"data")

    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock()
        mock_repo.get_local_path.return_value = str(test_file)
        mock_get_repo.return_value = mock_repo

        resp = client.get("/ajax-api/2.0/mlflow-artifacts/artifacts/data.bin")

    assert resp.status_code == 200
    assert resp.content == b"data"


def test_download_directory_path_returns_400(client, tmp_path):
    dir_path = tmp_path / "model_dir"
    dir_path.mkdir()

    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock()
        mock_repo.get_local_path.return_value = str(dir_path)
        mock_get_repo.return_value = mock_repo

        resp = client.get("/api/2.0/mlflow-artifacts/artifacts/model_dir")

    assert resp.status_code == 400


def test_download_disabled_returns_503(client_no_serve):
    resp = client_no_serve.get("/api/2.0/mlflow-artifacts/artifacts/model.pkl")
    assert resp.status_code == 503


def test_upload_with_stream_upload_mixin(client):
    test_data = b"uploaded artifact"

    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock(spec=LocalArtifactRepository)
        mock_get_repo.return_value = mock_repo

        resp = client.put(
            "/api/2.0/mlflow-artifacts/artifacts/nested/model.pkl",
            content=test_data,
        )

    assert resp.status_code == 200
    mock_repo.log_artifact_from_async_stream.assert_awaited_once()
    args, kwargs = mock_repo.log_artifact_from_async_stream.call_args
    assert args[1] == "model.pkl"
    assert kwargs["artifact_path"] == "nested"


def test_upload_without_stream_mixin_uses_log_artifact(client):
    test_data = b"uploaded artifact"

    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock(spec=ArtifactRepository)
        mock_get_repo.return_value = mock_repo

        resp = client.put(
            "/api/2.0/mlflow-artifacts/artifacts/nested/model.pkl",
            content=test_data,
        )

    assert resp.status_code == 200
    mock_repo.log_artifact.assert_called_once()
    args, kwargs = mock_repo.log_artifact.call_args
    assert args[0].endswith("model.pkl")
    assert kwargs["artifact_path"] == "nested"


def test_upload_preserves_content(client, tmp_path):
    test_data = b"x" * (ARTIFACT_STREAM_CHUNK_SIZE * 2 + 500)
    uploaded_content = None

    async def capture_stream(chunks, filename, artifact_path=None):
        nonlocal uploaded_content
        content = bytearray()
        async for chunk in chunks:
            content.extend(chunk)
        uploaded_content = bytes(content)

    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock(spec=LocalArtifactRepository)
        mock_repo.log_artifact_from_async_stream.side_effect = capture_stream
        mock_get_repo.return_value = mock_repo

        resp = client.put(
            "/api/2.0/mlflow-artifacts/artifacts/big_model.bin",
            content=test_data,
        )

    assert resp.status_code == 200
    assert uploaded_content == test_data


def test_upload_ajax_prefix(client):
    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock(spec=ArtifactRepository)
        mock_get_repo.return_value = mock_repo

        resp = client.put(
            "/ajax-api/2.0/mlflow-artifacts/artifacts/model.pkl",
            content=b"data",
        )

    assert resp.status_code == 200


def test_upload_disabled_returns_503(client_no_serve):
    resp = client_no_serve.put(
        "/api/2.0/mlflow-artifacts/artifacts/model.pkl",
        content=b"data",
    )
    assert resp.status_code == 503


def test_download_does_not_hit_flask_handler(client, tmp_path):
    test_file = tmp_path / "model.pkl"
    test_file.write_bytes(b"test")

    with (
        mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo,
        mock.patch("mlflow.server.handlers._download_artifact") as mock_flask_handler,
    ):
        mock_repo = mock.MagicMock()
        mock_repo.get_local_path.return_value = str(test_file)
        mock_get_repo.return_value = mock_repo

        resp = client.get("/api/2.0/mlflow-artifacts/artifacts/model.pkl")

    assert resp.status_code == 200
    mock_flask_handler.assert_not_called()


def test_upload_does_not_hit_flask_handler(client):
    with (
        mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo,
        mock.patch("mlflow.server.handlers._upload_artifact") as mock_flask_handler,
    ):
        mock_repo = mock.MagicMock(spec=ArtifactRepository)
        mock_get_repo.return_value = mock_repo

        resp = client.put(
            "/api/2.0/mlflow-artifacts/artifacts/model.pkl",
            content=b"test",
        )

    assert resp.status_code == 200
    mock_flask_handler.assert_not_called()


def test_path_traversal_rejected(client):
    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock()
        mock_get_repo.return_value = mock_repo

        resp = client.get("/api/2.0/mlflow-artifacts/artifacts/../../../etc/passwd")

    assert resp.status_code != 200


def test_upload_trailing_slash_rejected(client):
    with mock.patch("mlflow.server.artifact_router._get_artifact_repo") as mock_get_repo:
        mock_repo = mock.MagicMock(spec=ArtifactRepository)
        mock_get_repo.return_value = mock_repo

        resp = client.put(
            "/api/2.0/mlflow-artifacts/artifacts/nested/",
            content=b"data",
        )

    assert resp.status_code == 400
    assert "filename" in resp.json()["detail"].lower()


@pytest.fixture
def flask_client(monkeypatch):
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, "/tmp/mlflow-artifacts-test")
    return flask_app.test_client()


def test_flask_fallback_download(flask_client, tmp_path):
    test_data = b"flask fallback content"
    test_file = tmp_path / "model.pkl"
    test_file.write_bytes(test_data)

    with mock.patch("mlflow.server.handlers._get_artifact_repo_mlflow_artifacts") as mock_get_repo:
        mock_repo = mock.MagicMock()
        mock_repo.get_local_path.return_value = str(test_file)
        mock_get_repo.return_value = mock_repo

        resp = flask_client.get("/api/2.0/mlflow-artifacts/artifacts/test/model.pkl")

    assert resp.status_code == 200
    assert resp.data == test_data


def test_flask_fallback_upload(flask_client, tmp_path):
    test_data = b"flask upload content"

    with mock.patch("mlflow.server.handlers._get_artifact_repo_mlflow_artifacts") as mock_get_repo:
        mock_repo = mock.MagicMock(spec=LocalArtifactRepository)
        mock_get_repo.return_value = mock_repo

        resp = flask_client.put(
            "/api/2.0/mlflow-artifacts/artifacts/nested/model.pkl",
            data=test_data,
        )

    assert resp.status_code == 200
    mock_repo.log_artifact_from_stream.assert_called_once()
