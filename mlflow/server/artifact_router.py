"""
Native FastAPI artifact upload and download endpoints.

When MLflow runs under uvicorn/FastAPI (the default), these endpoints handle
artifact upload and download requests directly via ASGI, bypassing the WSGI
bridge and avoiding full-body buffering. This enables true streaming for large
artifact transfers.

When MLflow runs under gunicorn/waitress (Flask-only), the Flask handlers in
handlers.py serve these same URL patterns as the natural fallback.
"""

import asyncio
import json
import logging
import os
import posixpath
import tempfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ARTIFACT_STREAM_CHUNK_SIZE, StreamUploadMixin
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.uri import validate_path_is_safe

_logger = logging.getLogger(__name__)

artifact_router = APIRouter()


def _is_serving_proxied_artifacts() -> bool:
    from mlflow.server import SERVE_ARTIFACTS_ENV_VAR

    return os.environ.get(SERVE_ARTIFACTS_ENV_VAR, "false") == "true"


def _get_artifact_repo():
    from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR
    from mlflow.server.handlers import _get_artifact_repo_mlflow_artifacts

    if ARTIFACTS_DESTINATION_ENV_VAR not in os.environ:
        raise HTTPException(status_code=503, detail="Artifact serving is not configured.")
    return _get_artifact_repo_mlflow_artifacts()


def _get_workspace_scoped_path(artifact_path: str) -> str:
    from mlflow.server.handlers import _get_workspace_scoped_repo_path_if_enabled

    return _get_workspace_scoped_repo_path_if_enabled(artifact_path)


def _content_disposition(filename: str) -> str:
    from mlflow.server.handlers import _content_disposition_attachment

    return _content_disposition_attachment(filename)


@artifact_router.get("/api/2.0/mlflow-artifacts/artifacts/{artifact_path:path}")
@artifact_router.get("/ajax-api/2.0/mlflow-artifacts/artifacts/{artifact_path:path}")
async def download_artifact(artifact_path: str):
    if not _is_serving_proxied_artifacts():
        raise HTTPException(
            status_code=503,
            detail=(
                "Artifact serving is disabled. "
                "Run `mlflow server` with `--serve-artifacts` to enable."
            ),
        )

    try:
        artifact_path = validate_path_is_safe(artifact_path)
        artifact_path = _get_workspace_scoped_path(artifact_path)
        artifact_repo = _get_artifact_repo()

        if (local_path := artifact_repo.get_local_path(artifact_path)) is not None:
            abs_path = os.path.abspath(local_path)
            if os.path.isdir(abs_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Artifact path refers to a directory, not a file: '{artifact_path}'",
                )
            filename = os.path.basename(artifact_path)
            return FileResponse(
                abs_path,
                media_type=_guess_mime_type(abs_path),
                headers={
                    "Content-Disposition": _content_disposition(filename),
                    "X-Content-Type-Options": "nosniff",
                },
            )

        tmp_dir = tempfile.TemporaryDirectory()
        try:
            dst = os.path.abspath(
                await asyncio.to_thread(
                    artifact_repo.download_artifacts, artifact_path, tmp_dir.name
                )
            )
        except Exception:
            tmp_dir.cleanup()
            raise

        mime_type = _guess_mime_type(dst)
        filename = os.path.basename(artifact_path)

        def _stream_file():
            """Sync generator — Starlette iterates it in a threadpool automatically."""
            try:
                with open(dst, "rb") as f:
                    while chunk := f.read(ARTIFACT_STREAM_CHUNK_SIZE):
                        yield chunk
            finally:
                tmp_dir.cleanup()

        return StreamingResponse(
            _stream_file(),
            media_type=mime_type,
            headers={
                "Content-Disposition": _content_disposition(filename),
                "X-Content-Type-Options": "nosniff",
            },
        )
    except HTTPException:
        raise
    except MlflowException as e:
        return JSONResponse(
            status_code=e.get_http_status_code(),
            content=json.loads(e.serialize_as_json()),
        )


@artifact_router.put("/api/2.0/mlflow-artifacts/artifacts/{artifact_path:path}")
@artifact_router.put("/ajax-api/2.0/mlflow-artifacts/artifacts/{artifact_path:path}")
async def upload_artifact(artifact_path: str, request: Request):
    if not _is_serving_proxied_artifacts():
        raise HTTPException(
            status_code=503,
            detail=(
                "Artifact serving is disabled. "
                "Run `mlflow server` with `--serve-artifacts` to enable."
            ),
        )

    try:
        artifact_path = validate_path_is_safe(artifact_path)
        artifact_path = _get_workspace_scoped_path(artifact_path)
        head, tail = posixpath.split(artifact_path)

        if not tail:
            raise HTTPException(
                status_code=400,
                detail="Artifact path must include a filename (cannot end with '/').",
            )

        artifact_repo = _get_artifact_repo()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, tail)
            with open(tmp_path, "wb") as f:
                async for chunk in request.stream():
                    f.write(chunk)

            if isinstance(artifact_repo, StreamUploadMixin):
                with open(tmp_path, "rb") as stream_file:
                    await asyncio.to_thread(
                        artifact_repo.log_artifact_from_stream,
                        stream_file,
                        tail,
                        artifact_path=head or None,
                    )
            else:
                await asyncio.to_thread(
                    artifact_repo.log_artifact, tmp_path, artifact_path=head or None
                )

        return Response(status_code=200)
    except MlflowException as e:
        return JSONResponse(
            status_code=e.get_http_status_code(),
            content=json.loads(e.serialize_as_json()),
        )
