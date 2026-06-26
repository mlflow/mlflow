"""Framework-agnostic response builders for MLflow server handlers.

During the migration period these return Flask Response objects. Once
Flask is removed they will produce Starlette/FastAPI responses instead,
requiring zero changes in handler code.
"""

from __future__ import annotations

import json
from typing import Any

from flask import Response, send_file


def json_response(data: str, status: int = 200, headers: dict[str, str] | None = None) -> Response:
    """Return a JSON response from an already-serialized string."""
    response = Response(data, status=status, mimetype="application/json")
    if headers:
        for k, v in headers.items():
            response.headers[k] = v
    return response


def jsonify_response(obj: Any, status: int = 200) -> Response:
    """Return a JSON response from a Python object (replaces ``flask.jsonify``)."""
    return json_response(json.dumps(obj), status=status)


def text_response(text: str, status: int = 200) -> Response:
    """Return a plain-text response."""
    return Response(text, status=status, mimetype="text/plain")


def empty_response(status: int = 204) -> Response:
    return Response(status=status)


def file_response(
    path_or_file,
    *,
    mimetype: str | None = None,
    as_attachment: bool = False,
    download_name: str | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    """Serve a file or file-like object (replaces ``flask.send_file``)."""
    kwargs: dict[str, Any] = {"mimetype": mimetype}
    if as_attachment:
        kwargs["as_attachment"] = True
    if download_name:
        kwargs["download_name"] = download_name
    resp = send_file(path_or_file, **kwargs)
    if headers:
        for k, v in headers.items():
            resp.headers[k] = v
    return resp


def streaming_response(generator, headers: dict[str, str] | None = None) -> Response:
    """Return a streaming response from a generator."""
    resp = Response(generator)
    if headers:
        for k, v in headers.items():
            resp.headers[k] = v
    return resp
