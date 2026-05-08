from __future__ import annotations

import asyncio
import email.message
import io
import json
import urllib.error
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pypi
import pytest
from packaging.version import Version

_URLOPEN = "pypi._client.urllib.request.urlopen"


def _payload(versions: list[str]) -> dict[str, Any]:
    return {
        "info": {"name": "demo", "summary": "demo"},
        "releases": {v: [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}] for v in versions},
    }


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example/", code=code, msg="boom", hdrs=email.message.Message(), fp=None
    )


def _success_response(payload: dict[str, Any]) -> mock.MagicMock:
    body = io.BytesIO(json.dumps(payload).encode())
    return mock.MagicMock(
        __enter__=mock.MagicMock(return_value=body),
        __exit__=mock.MagicMock(return_value=False),
    )


def _urlopen_returning(payloads: list[dict[str, Any]]) -> Any:
    iterator: Iterator[dict[str, Any]] = iter(payloads)

    def _open(url: str, timeout: float) -> mock.MagicMock:
        return _success_response(next(iterator))

    return _open


def test_get_package_returns_typed_package() -> None:
    payload = _payload(["1.0.0", "2.0.0"])
    with mock.patch(_URLOPEN, side_effect=_urlopen_returning([payload])) as mock_open:
        pkg = pypi.get_package("demo")
        mock_open.assert_called_once()

    assert isinstance(pkg, pypi.Package)
    assert pkg.latest_version == Version("2.0.0")


def test_get_package_uses_in_memory_cache() -> None:
    payload = _payload(["1.0.0"])
    with mock.patch(_URLOPEN, side_effect=_urlopen_returning([payload])) as mock_open:
        pypi.get_package("demo")
        pypi.get_package("demo")
        mock_open.assert_called_once()


def test_clear_cache_forces_refetch() -> None:
    payload = _payload(["1.0.0"])
    with mock.patch(_URLOPEN, side_effect=_urlopen_returning([payload, payload])) as mock_open:
        pypi.get_package("demo")
        pypi.clear_cache()
        pypi.get_package("demo")
        assert mock_open.call_count == 2


def test_pypi_url_env_var_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYPI_URL", "https://internal-proxy.example/")
    seen: list[str] = []

    def _open(url: str, timeout: float) -> mock.MagicMock:
        seen.append(url)
        return _success_response(_payload(["1.0.0"]))

    with mock.patch(_URLOPEN, side_effect=_open):
        pypi.get_package("demo")

    assert seen == ["https://internal-proxy.example/pypi/demo/json"]


def test_404_raises_pypi_error_without_retry() -> None:
    with mock.patch(_URLOPEN, side_effect=_http_error(404)) as mock_open:
        with pytest.raises(pypi.PyPIError, match="Package not found"):
            pypi.get_package("does-not-exist")
        mock_open.assert_called_once()


def test_non_retryable_http_error_raises_immediately() -> None:
    with mock.patch(_URLOPEN, side_effect=_http_error(401)) as mock_open:
        with pytest.raises(pypi.PyPIError, match="HTTP 401"):
            pypi.get_package("demo")
        mock_open.assert_called_once()


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retries_transient_http_errors_then_succeeds(status: int) -> None:
    payload = _payload(["1.0.0"])
    side_effects = [_http_error(status), _http_error(status), _success_response(payload)]
    with mock.patch(_URLOPEN, side_effect=side_effects) as mock_open:
        pkg = pypi.get_package("demo")
        assert mock_open.call_count == 3
    assert pkg.latest_version == Version("1.0.0")


def test_retries_url_errors_then_succeeds() -> None:
    payload = _payload(["1.0.0"])
    side_effects: list[Any] = [
        urllib.error.URLError("dns fail"),
        ConnectionResetError("reset"),
        _success_response(payload),
    ]
    with mock.patch(_URLOPEN, side_effect=side_effects) as mock_open:
        pkg = pypi.get_package("demo")
        assert mock_open.call_count == 3
    assert pkg.latest_version == Version("1.0.0")


def test_retry_exhaustion_raises_pypi_error() -> None:
    err = urllib.error.URLError("dns fail")
    with mock.patch(_URLOPEN, side_effect=[err, err, err]) as mock_open:
        with pytest.raises(pypi.PyPIError, match="after 3 attempts"):
            pypi.get_package("demo")
        assert mock_open.call_count == 3


def test_unexpected_payload_shape_raises() -> None:
    list_payload_response = mock.MagicMock(
        __enter__=mock.MagicMock(return_value=io.BytesIO(json.dumps([1, 2, 3]).encode())),
        __exit__=mock.MagicMock(return_value=False),
    )
    with mock.patch(_URLOPEN, return_value=list_payload_response):
        with pytest.raises(pypi.PyPIError, match="Unexpected response"):
            pypi.get_package("demo")


def test_aget_package_returns_package() -> None:
    payload = _payload(["1.0.0"])
    with mock.patch(_URLOPEN, side_effect=_urlopen_returning([payload])):
        pkg = asyncio.run(pypi.aget_package("demo"))

    assert isinstance(pkg, pypi.Package)
    assert pkg.latest_version == Version("1.0.0")


def test_aget_packages_preserves_order_and_fetches_each() -> None:
    names = ["alpha", "beta", "gamma"]
    payloads = {n: _payload([f"{i + 1}.0.0"]) for i, n in enumerate(names)}
    seen: list[str] = []

    def _open(url: str, timeout: float) -> mock.MagicMock:
        for n in names:
            if url.endswith(f"/pypi/{n}/json"):
                seen.append(n)
                return _success_response(payloads[n])
        raise AssertionError(f"unexpected URL: {url}")

    with mock.patch(_URLOPEN, side_effect=_open):
        pkgs = asyncio.run(pypi.aget_packages(names))

    expected = [Version("1.0.0"), Version("2.0.0"), Version("3.0.0")]
    assert [p.latest_version for p in pkgs] == expected
    assert sorted(seen) == sorted(names)
