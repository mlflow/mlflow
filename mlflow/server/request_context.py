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

import io
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Authorization:
    username: str | None = None
    password: str | None = None


@dataclass
class _Args:
    """Minimal multi-dict over query parameters."""

    _data: dict[str, list[str]] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        values = self._data.get(key)
        if values:
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

    def __bool__(self) -> bool:
        return bool(self._data)


@dataclass
class RequestShim:
    """Sync-accessible request object matching the Flask surface used by handlers."""

    method: str = "GET"
    args: _Args = field(default_factory=_Args)
    content_type: str | None = None
    content_length: int | None = None
    url_rule: str | None = None
    authorization: Authorization | None = None
    _json: Any = None
    _data: bytes = b""
    _stream: io.BytesIO | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def get_json(self, force: bool = False, silent: bool = False) -> Any:
        return self._json

    @property
    def json(self) -> Any:
        return self._json

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def stream(self) -> io.BytesIO:
        if self._stream is None:
            self._stream = io.BytesIO(self._data)
        return self._stream


_current_request: ContextVar[RequestShim | None] = ContextVar(
    "mlflow_request", default=None
)


def get_request() -> RequestShim:
    r = _current_request.get()
    if r is None:
        # Fall back to Flask's request context during the migration period.
        # Tests using app.test_request_context() don't trigger before_request
        # hooks, so the shim may not be populated.
        try:
            import flask

            if flask.has_request_context():
                return from_flask_request(flask.request)
        except ImportError:
            pass
        raise RuntimeError("No active request context")
    return r


def set_request(shim: RequestShim) -> None:
    _current_request.set(shim)


def clear_request() -> None:
    _current_request.set(None)


# ---------------------------------------------------------------------------
# flask.g replacement
# ---------------------------------------------------------------------------
_request_state: ContextVar[dict[str, Any]] = ContextVar(
    "mlflow_request_state", default=None
)


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

    return RequestShim(
        method=flask_request.method,
        args=_Args(_data=args_data),
        content_type=flask_request.content_type,
        content_length=flask_request.content_length,
        url_rule=str(flask_request.url_rule) if flask_request.url_rule else None,
        authorization=auth,
        _json=flask_request.get_json(force=True, silent=True),
        _data=flask_request.get_data(),
        _stream=None,
    )
