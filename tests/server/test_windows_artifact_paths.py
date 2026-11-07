from unittest import mock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.server import (
    ARTIFACTS_DESTINATION_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
    artifact_router,
    handlers,
)
from mlflow.server import app as flask_app
from mlflow.store.artifact.artifact_repo import ArtifactRepository, StreamUploadMixin
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository


class _RemoteArtifactRepository(ArtifactRepository, StreamUploadMixin):
    pass


@pytest.mark.parametrize(
    ("framework", "operation"),
    [
        ("flask", "download"),
        ("flask", "upload"),
        ("flask", "list"),
        ("flask", "delete"),
        ("fastapi", "download"),
        ("fastapi", "upload"),
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        "folder./file.txt",
        "folder /file.txt",
        "FOLDER~1/file.txt",
        "folder/file.txt.",
        "folder/file:stream",
    ],
)
@pytest.mark.parametrize("local", [True, False])
def test_windows_artifact_routes_enforce_filesystem_policy(
    monkeypatch, tmp_path, framework, operation, path, local
):
    monkeypatch.setattr("mlflow.utils.uri.is_windows", lambda: True)
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, str(tmp_path) if local else "s3://bucket")
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    monkeypatch.delenv("MLFLOW_ARTIFACTS_PRESIGNED_ONLY", raising=False)
    if local:
        repo = LocalArtifactRepository(str(tmp_path))
    else:
        repo = mock.MagicMock(spec=_RemoteArtifactRepository)
        repo.get_local_path.return_value = None
        downloaded = tmp_path / "downloaded.txt"
        downloaded.write_text("artifact")
        repo.download_artifacts.return_value = str(downloaded)
        repo.list_artifacts.return_value = []
    monkeypatch.setattr(handlers, "_get_artifact_repo_mlflow_artifacts", lambda: repo)
    monkeypatch.setattr(artifact_router, "_get_artifact_repo", lambda: repo)
    if framework == "fastapi":
        app = FastAPI()
        app.include_router(artifact_router.artifact_router)
        with TestClient(app) as client:
            url = f"/api/2.0/mlflow-artifacts/artifacts/{path}"
            response = client.get(url) if operation == "download" else client.put(url, content=b"x")
    else:
        method = {"download": "GET", "upload": "PUT", "list": "GET", "delete": "DELETE"}[operation]
        with flask_app.test_request_context(
            "/api/2.0/mlflow-artifacts/artifacts",
            method=method,
            query_string={"path": path},
            data=b"x" if operation == "upload" else None,
        ):
            response = {
                "download": lambda: handlers._download_artifact(path),
                "upload": lambda: handlers._upload_artifact(path),
                "list": handlers._list_artifacts_mlflow_artifacts,
                "delete": lambda: handlers._delete_artifact_mlflow_artifacts(path),
            }[operation]()
    assert response.status_code == (400 if local else 200)
    if not local:
        if operation == "download":
            repo.get_local_path.assert_called_once_with(path)
        elif operation == "list":
            repo.list_artifacts.assert_called_once_with(path)
        elif operation == "delete":
            repo.delete_artifacts.assert_called_once_with(path)
        elif framework == "flask":
            assert (
                repo.log_artifact_from_stream.call_args.kwargs["artifact_path"]
                == path.rsplit("/", 1)[0]
            )
        else:
            assert (
                repo.log_artifact_from_async_stream.call_args.kwargs["artifact_path"]
                == path.rsplit("/", 1)[0]
            )
    if framework == "flask":
        response.close()
