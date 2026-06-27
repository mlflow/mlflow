"""Framework-agnostic request context for MLflow server handlers.

Provides a thin request shim backed by a ``contextvars.ContextVar`` so that
handler code can read the current HTTP request without importing Flask. The
shim exposes only the surface area actually used by ``handlers.py`` and
``validation.py``.

During the migration period both Flask and FastAPI populate the contextvar
via their respective middleware/hooks. Once Flask is removed only the FastAPI
middleware remains.
"""

from __future__ import annotations

import base64
import io
import json as _json_module
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs


class _CIHeaders(dict):
    """Case-insensitive dict for HTTP headers."""

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __contains__(self, key):
        return super().__contains__(key.lower())

    def get(self, key, default=None):
        return super().get(key.lower(), default)

    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)


@dataclass
class Authorization:
    username: str | None = None
    password: str | None = None
    auth_type: str | None = None
    _data: dict[str, Any] | None = None

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        auth_type: str | None = None,
        data: dict[str, Any] | None = None,
    ):
        self.auth_type = auth_type
        self._data = data or {}
        self.username = username if username is not None else self._data.get("username")
        self.password = password if password is not None else self._data.get("password")


@dataclass
class _Args:
    """Minimal multi-dict over query parameters."""

    _data: dict[str, list[str]] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        if values := self._data.get(key):
            return values[0]
        return default

    def getlist(self, key: str) -> list[str]:
        return list(self._data.get(key, []))

    def to_dict(self, flat: bool = True) -> dict[str, Any]:
        if flat:
            return {k: v[0] for k, v in self._data.items() if v}
        return dict(self._data)

    def __getitem__(self, key: str) -> str:
        values = self._data.get(key)
        if not values:
            raise KeyError(key)
        return values[0]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return ((k, v[0]) for k, v in self._data.items() if v)

    def __bool__(self) -> bool:
        return bool(self._data)

    def __or__(self, other: dict[str, Any]) -> dict[str, Any]:
        return self.to_dict() | other


_SENTINEL = object()


class RequestShim:
    """Sync-accessible request object matching the Flask surface used by handlers.

    Fields are parsed lazily on first access to avoid paying for JSON parsing,
    auth decoding, header copying, etc. on requests that never touch them.
    """

    __slots__ = (
        "method",
        "path",
        "url_rule",
        "content_type",
        "content_length",
        "view_args",
        "state",
        "_raw_body",
        "_raw_starlette",
        "_args",
        "_authorization",
        "_json",
        "_stream",
        "_headers",
        "_form",
    )

    def __init__(
        self,
        *,
        method: str = "GET",
        path: str = "",
        url_rule: str | None = None,
        content_type: str | None = None,
        content_length: int | None = None,
        view_args: dict[str, str] | None = None,
        state: dict[str, Any] | None = None,
        raw_body: bytes = b"",
        raw_starlette=None,
        # Eager overrides (used by from_flask_request where values are already parsed)
        args: _Args | None = None,
        authorization: Authorization | None = _SENTINEL,
        json_body: Any = _SENTINEL,
        headers: dict[str, str] | _CIHeaders | None = None,
        form: dict[str, str] | None = None,
    ):
        self.method = method
        self.path = path
        self.url_rule = url_rule
        self.content_type = content_type
        self.content_length = content_length
        self.view_args = view_args or {}
        self.state = state or {}
        self._raw_body = raw_body
        self._raw_starlette = raw_starlette
        self._args = args
        self._authorization = authorization
        self._json = json_body
        self._stream = None
        self._headers = headers
        self._form = form

    @property
    def args(self) -> _Args:
        if self._args is None:
            if self._raw_starlette is not None:
                data: dict[str, list[str]] = {}
                for key, value in self._raw_starlette.query_params.multi_items():
                    data.setdefault(key, []).append(value)
                self._args = _Args(_data=data)
            else:
                self._args = _Args()
        return self._args

    @args.setter
    def args(self, value: _Args) -> None:
        self._args = value

    @property
    def authorization(self) -> Authorization | None:
        if self._authorization is _SENTINEL:
            self._authorization = None
            if self._raw_starlette is not None:
                auth_header = self._raw_starlette.headers.get("authorization", "")
                if auth_header.lower().startswith("basic "):
                    try:
                        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                        username, _, password = decoded.partition(":")
                        self._authorization = Authorization(username=username, password=password)
                    except Exception:
                        pass
        return self._authorization

    @authorization.setter
    def authorization(self, value: Authorization | None) -> None:
        self._authorization = value

    def get_json(self, force: bool = False, silent: bool = False) -> Any:
        return self.json

    @property
    def json(self) -> Any:
        if self._json is _SENTINEL:
            self._json = None
            if self._raw_body:
                try:
                    self._json = _json_module.loads(self._raw_body)
                except (ValueError, TypeError):
                    pass
        return self._json

    @json.setter
    def json(self, value: Any) -> None:
        self._json = value

    @property
    def is_json(self) -> bool:
        ct = self.content_type or ""
        return "json" in ct.lower()

    @property
    def data(self) -> bytes:
        return self._raw_body

    @property
    def stream(self) -> io.BytesIO:
        if self._stream is None:
            self._stream = io.BytesIO(self._raw_body)
        return self._stream

    @property
    def headers(self) -> _CIHeaders:
        if self._headers is None:
            if self._raw_starlette is not None:
                self._headers = _CIHeaders(dict(self._raw_starlette.headers))
            else:
                self._headers = _CIHeaders()
        elif not isinstance(self._headers, _CIHeaders):
            self._headers = _CIHeaders(self._headers)
        return self._headers

    @headers.setter
    def headers(self, value) -> None:
        self._headers = value

    @property
    def form(self) -> dict[str, str]:
        if self._form is None:
            self._form = {}
            ct = self.content_type
            if ct and "application/x-www-form-urlencoded" in ct and self._raw_body:
                for k, vs in parse_qs(self._raw_body.decode("utf-8")).items():
                    if vs:
                        self._form[k] = vs[0]
        return self._form

    @form.setter
    def form(self, value: dict[str, str]) -> None:
        self._form = value


_current_request: ContextVar[RequestShim | None] = ContextVar("mlflow_request", default=None)


def get_request() -> RequestShim:
    r = _current_request.get()
    if r is None:
        raise RuntimeError("No active request context")
    return r


def set_request(shim: RequestShim) -> None:
    _current_request.set(shim)


def clear_request() -> None:
    _current_request.set(None)


# ---------------------------------------------------------------------------
# flask.g replacement
# ---------------------------------------------------------------------------
_request_state: ContextVar[dict[str, Any]] = ContextVar("mlflow_request_state", default=None)


class _GProxy:
    """Drop-in replacement for ``flask.g`` backed by a contextvar dict."""

    def _store(self) -> dict[str, Any]:
        d = _request_state.get()
        if d is None:
            d = {}
            _request_state.set(d)
        return d

    def __getattr__(self, name: str) -> Any:
        try:
            return self._store()[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self._store()[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self._store()[name]
        except KeyError:
            raise AttributeError(name) from None


g = _GProxy()


def clear_g() -> None:
    _request_state.set(None)


# ---------------------------------------------------------------------------
# Builders: populate the shim from Flask or Starlette request objects
# ---------------------------------------------------------------------------
def from_flask_request(flask_request) -> RequestShim:
    """Build a ``RequestShim`` from a Flask/Werkzeug request object."""
    args_data: dict[str, list[str]] = {}
    for key in flask_request.args:
        args_data[key] = flask_request.args.getlist(key)

    auth = None
    if flask_request.authorization:
        auth = Authorization(
            username=flask_request.authorization.username,
            password=flask_request.authorization.password,
        )

    headers = dict(flask_request.headers)
    form = dict(flask_request.form) if hasattr(flask_request, "form") else {}

    return RequestShim(
        method=flask_request.method,
        path=flask_request.path,
        url_rule=str(flask_request.url_rule) if flask_request.url_rule else None,
        content_type=flask_request.content_type,
        content_length=flask_request.content_length,
        view_args=dict(flask_request.view_args or {}),
        raw_body=flask_request.get_data(),
        args=_Args(_data=args_data),
        authorization=auth,
        json_body=flask_request.get_json(force=True, silent=True),
        headers=headers,
        form=form,
    )


async def from_starlette_request(starlette_request) -> RequestShim:
    """Build a ``RequestShim`` from a Starlette/FastAPI request object.

    Only the request body is read eagerly (it requires ``await``). All other
    fields (JSON, auth, headers, query params, form data) are parsed lazily on
    first access.
    """
    body = await starlette_request.body()

    ct = starlette_request.headers.get("content-type")
    cl_raw = starlette_request.headers.get("content-length")
    request_path = starlette_request.scope.get("path", "")
    path_params = dict(getattr(starlette_request, "path_params", None) or {})

    return RequestShim(
        method=starlette_request.method,
        path=request_path,
        url_rule=request_path,
        content_type=ct,
        content_length=int(cl_raw) if cl_raw else None,
        view_args=path_params,
        raw_body=body,
        raw_starlette=starlette_request,
    )
