"""Tests for native FastAPI artifact upload/download endpoints (artifact_router.py).

Verifies that:
1. Download uses FileResponse for local artifacts (get_local_path fast path)
2. Download uses StreamingResponse for remote artifacts
3. Upload streams request body to disk without full-body buffering
4. Upload uses log_artifact_from_stream when StreamUploadMixin is available
5. Upload falls back to log_artifact for non-StreamUploadMixin repos
6. Endpoints return 503 when serve-artifacts is disabled
7. Routing explicitly serves these paths via FastAPI (not WSGI bridge)
"""

from unittest import mock

import pytest
from starlette.testclient import TestClient

from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR
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


class TestDownloadArtifact:
    def test_local_path_returns_file_response(self, client, tmp_path):
        test_data = b"local artifact content"
        test_file = tmp_path / "model.pkl"
        test_file.write_bytes(test_data)

        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
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

    def test_remote_path_returns_streaming_response(self, client, tmp_path):
        test_data = b"remote artifact content"
        test_file = tmp_path / "model.pkl"
        test_file.write_bytes(test_data)

        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
            mock_repo = mock.MagicMock()
            mock_repo.get_local_path.return_value = None
            mock_repo.download_artifacts.return_value = str(test_file)
            mock_get_repo.return_value = mock_repo

            resp = client.get("/api/2.0/mlflow-artifacts/artifacts/nested/model.pkl")

        assert resp.status_code == 200
        assert resp.content == test_data
        assert "attachment" in resp.headers["Content-Disposition"]
        mock_repo.download_artifacts.assert_called_once()

    def test_ajax_prefix_also_works(self, client, tmp_path):
        test_file = tmp_path / "data.bin"
        test_file.write_bytes(b"data")

        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
            mock_repo = mock.MagicMock()
            mock_repo.get_local_path.return_value = str(test_file)
            mock_get_repo.return_value = mock_repo

            resp = client.get("/ajax-api/2.0/mlflow-artifacts/artifacts/data.bin")

        assert resp.status_code == 200
        assert resp.content == b"data"

    def test_directory_path_returns_400(self, client, tmp_path):
        dir_path = tmp_path / "model_dir"
        dir_path.mkdir()

        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
            mock_repo = mock.MagicMock()
            mock_repo.get_local_path.return_value = str(dir_path)
            mock_get_repo.return_value = mock_repo

            resp = client.get("/api/2.0/mlflow-artifacts/artifacts/model_dir")

        assert resp.status_code == 400

    def test_disabled_returns_503(self, client_no_serve):
        resp = client_no_serve.get("/api/2.0/mlflow-artifacts/artifacts/model.pkl")
        assert resp.status_code == 503


class TestUploadArtifact:
    def test_upload_with_stream_upload_mixin(self, client):
        test_data = b"uploaded artifact"

        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
            mock_repo = mock.MagicMock(spec=LocalArtifactRepository)
            mock_get_repo.return_value = mock_repo

            resp = client.put(
                "/api/2.0/mlflow-artifacts/artifacts/nested/model.pkl",
                content=test_data,
            )

        assert resp.status_code == 200
        mock_repo.log_artifact_from_stream.assert_called_once()
        args, kwargs = mock_repo.log_artifact_from_stream.call_args
        assert args[1] == "model.pkl"
        assert kwargs["artifact_path"] == "nested"

    def test_upload_without_stream_mixin_uses_log_artifact(self, client):
        test_data = b"uploaded artifact"

        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
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

    def test_upload_preserves_content(self, client, tmp_path):
        test_data = b"x" * (ARTIFACT_STREAM_CHUNK_SIZE * 2 + 500)
        uploaded_content = None

        def capture_stream(stream, filename, artifact_path=None):
            nonlocal uploaded_content
            uploaded_content = stream.read()

        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
            mock_repo = mock.MagicMock(spec=LocalArtifactRepository)
            mock_repo.log_artifact_from_stream.side_effect = capture_stream
            mock_get_repo.return_value = mock_repo

            resp = client.put(
                "/api/2.0/mlflow-artifacts/artifacts/big_model.bin",
                content=test_data,
            )

        assert resp.status_code == 200
        assert uploaded_content == test_data

    def test_upload_ajax_prefix(self, client):
        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
            mock_repo = mock.MagicMock(spec=ArtifactRepository)
            mock_get_repo.return_value = mock_repo

            resp = client.put(
                "/ajax-api/2.0/mlflow-artifacts/artifacts/model.pkl",
                content=b"data",
            )

        assert resp.status_code == 200

    def test_disabled_returns_503(self, client_no_serve):
        resp = client_no_serve.put(
            "/api/2.0/mlflow-artifacts/artifacts/model.pkl",
            content=b"data",
        )
        assert resp.status_code == 503


class TestRoutingExplicitness:
    """Verify that under FastAPI/ASGI, the native router handles artifact requests
    instead of falling through to the WSGI bridge (Flask handlers).
    """

    def test_download_does_not_hit_flask_handler(self, client, tmp_path):
        test_file = tmp_path / "model.pkl"
        test_file.write_bytes(b"test")

        with (
            mock.patch(
                "mlflow.server.artifact_router._get_artifact_repo"
            ) as mock_get_repo,
            mock.patch("mlflow.server.handlers._download_artifact") as mock_flask_handler,
        ):
            mock_repo = mock.MagicMock()
            mock_repo.get_local_path.return_value = str(test_file)
            mock_get_repo.return_value = mock_repo

            resp = client.get("/api/2.0/mlflow-artifacts/artifacts/model.pkl")

        assert resp.status_code == 200
        mock_flask_handler.assert_not_called()

    def test_upload_does_not_hit_flask_handler(self, client):
        with (
            mock.patch(
                "mlflow.server.artifact_router._get_artifact_repo"
            ) as mock_get_repo,
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


class TestPathSafety:
    def test_path_traversal_rejected(self, client):
        with mock.patch(
            "mlflow.server.artifact_router._get_artifact_repo"
        ) as mock_get_repo:
            mock_repo = mock.MagicMock()
            mock_get_repo.return_value = mock_repo

            resp = client.get("/api/2.0/mlflow-artifacts/artifacts/../../../etc/passwd")

        # Path traversal is blocked — either by Starlette URL normalization (404)
        # or by validate_path_is_safe (400/500). The key assertion is that it does
        # NOT return 200 with file content.
        assert resp.status_code != 200
