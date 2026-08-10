from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from mlflow.utils.rest_utils import MlflowHostCreds, http_request

SERVER_INFO_ENDPOINT = "/api/3.0/mlflow/server-info"

SERVER_INFO_STORE_TYPE = "store_type"
SERVER_INFO_WORKSPACES_ENABLED = "workspaces_enabled"
SERVER_INFO_TRACE_ARCHIVAL_ENABLED = "trace_archival_enabled"
SERVER_INFO_MULTIPART_UPLOADS_ENABLED = "multipart_uploads_enabled"
SERVER_INFO_MULTIPART_DOWNLOADS_ENABLED = "multipart_downloads_enabled"

_SERVER_INFO_CACHE_MAXSIZE = 16
_SERVER_INFO_CACHE_TTL_SECONDS = 60.0


@dataclass(frozen=True)
class ServerInfoResponse:
    status_code: int
    data: dict[str, Any] | None = None
    text: str | None = None


class ServerInfoRequestError(Exception):
    pass


@dataclass(frozen=True)
class _ServerInfoCacheEntry:
    response: ServerInfoResponse
    expires_at_monotonic: float


@dataclass
class _InFlightRequest:
    event: threading.Event = field(default_factory=threading.Event)
    response: ServerInfoResponse | None = None
    error: BaseException | None = None


_SERVER_INFO_CACHE: OrderedDict[str, _ServerInfoCacheEntry] = OrderedDict()
_SERVER_INFO_CACHE_LOCK = threading.Lock()
_SERVER_INFO_IN_FLIGHT: dict[str, _InFlightRequest] = {}


def _reset_server_info_state_after_fork() -> None:
    global _SERVER_INFO_CACHE, _SERVER_INFO_CACHE_LOCK, _SERVER_INFO_IN_FLIGHT

    # A child cannot safely reuse locks or in-flight requests owned by threads
    # that only exist in the parent process.
    _SERVER_INFO_CACHE = OrderedDict()
    _SERVER_INFO_CACHE_LOCK = threading.Lock()
    _SERVER_INFO_IN_FLIGHT = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_server_info_state_after_fork)


def _normalize_server_info_cache_key(host: str) -> str:
    return host.removesuffix("/")


def _cache_success_locked(key: str, response: ServerInfoResponse) -> None:
    _SERVER_INFO_CACHE[key] = _ServerInfoCacheEntry(
        response=response,
        expires_at_monotonic=time.monotonic() + _SERVER_INFO_CACHE_TTL_SECONDS,
    )
    _SERVER_INFO_CACHE.move_to_end(key)
    if len(_SERVER_INFO_CACHE) > _SERVER_INFO_CACHE_MAXSIZE:
        _SERVER_INFO_CACHE.popitem(last=False)


def _fetch_server_info_uncached(host_creds: MlflowHostCreds) -> ServerInfoResponse:
    try:
        response = http_request(
            host_creds=host_creds,
            endpoint=SERVER_INFO_ENDPOINT,
            method="GET",
            timeout=3,
            max_retries=0,
            raise_on_status=False,
        )
    except Exception as exc:  # pragma: no cover - behavior validated by callers
        raise ServerInfoRequestError(str(exc)) from exc

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError as exc:
            raise ServerInfoRequestError(
                f"Invalid JSON returned by {SERVER_INFO_ENDPOINT}"
            ) from exc

        if not isinstance(data, dict):
            raise ServerInfoRequestError(
                f"Expected a JSON object from {SERVER_INFO_ENDPOINT}, got {type(data).__name__}"
            )

        return ServerInfoResponse(status_code=200, data=data)

    return ServerInfoResponse(status_code=response.status_code, text=response.text)


def fetch_server_info(host_creds: MlflowHostCreds) -> ServerInfoResponse:
    key = _normalize_server_info_cache_key(host_creds.host)
    with _SERVER_INFO_CACHE_LOCK:
        if cache_entry := _SERVER_INFO_CACHE.get(key):
            if time.monotonic() < cache_entry.expires_at_monotonic:
                _SERVER_INFO_CACHE.move_to_end(key)
                return cache_entry.response
            _SERVER_INFO_CACHE.pop(key)

        if in_flight := _SERVER_INFO_IN_FLIGHT.get(key):
            leader = False
        else:
            in_flight = _InFlightRequest()
            _SERVER_INFO_IN_FLIGHT[key] = in_flight
            leader = True

    if not leader:
        in_flight.event.wait()
        if in_flight.error is not None:
            raise in_flight.error
        return in_flight.response

    try:
        response = _fetch_server_info_uncached(host_creds)
        with _SERVER_INFO_CACHE_LOCK:
            if response.status_code == 200:
                _cache_success_locked(key, response)
            in_flight.response = response
    except BaseException as exc:
        in_flight.error = exc
        raise
    finally:
        with _SERVER_INFO_CACHE_LOCK:
            _SERVER_INFO_IN_FLIGHT.pop(key, None)
            in_flight.event.set()

    return response


def _clear_server_info_cache() -> None:
    with _SERVER_INFO_CACHE_LOCK:
        _SERVER_INFO_CACHE.clear()
