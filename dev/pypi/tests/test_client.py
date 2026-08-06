from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import mock

import pypi
import pytest
from packaging.version import Version


def _payload(versions: list[str]) -> dict[str, Any]:
    return {
        "info": {"name": "demo", "summary": "demo"},
        "releases": {v: [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}] for v in versions},
    }


def _patch_fetch(side_effect: Any) -> Any:
    return mock.patch("pypi._client._fetch_json", side_effect=side_effect)


def test_get_package_returns_typed_package() -> None:
    payload = _payload(["1.0.0", "2.0.0"])

    async def _f(session: object, url: str) -> dict[str, Any]:
        return payload

    with _patch_fetch(_f) as mock_fetch:
        pkg = asyncio.run(pypi.get_package("demo"))
        mock_fetch.assert_called_once()

    assert isinstance(pkg, pypi.Package)
    assert pkg.latest_version == Version("2.0.0")


def test_get_package_uses_in_memory_cache() -> None:
    payload = _payload(["1.0.0"])

    async def _f(session: object, url: str) -> dict[str, Any]:
        return payload

    async def go() -> None:
        await pypi.get_package("demo")
        await pypi.get_package("demo")

    with _patch_fetch(_f) as mock_fetch:
        asyncio.run(go())
        mock_fetch.assert_called_once()


def test_clear_cache_forces_refetch() -> None:
    payload = _payload(["1.0.0"])

    async def _f(session: object, url: str) -> dict[str, Any]:
        return payload

    with _patch_fetch(_f) as mock_fetch:
        asyncio.run(pypi.get_package("demo"))
        pypi.clear_cache()
        asyncio.run(pypi.get_package("demo"))
        assert mock_fetch.call_count == 2


def test_pypi_url_env_var_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYPI_URL", "https://internal-proxy.example/")
    seen: list[str] = []

    async def _f(session: object, url: str) -> dict[str, Any]:
        seen.append(url)
        return _payload(["1.0.0"])

    with _patch_fetch(_f):
        asyncio.run(pypi.get_package("demo"))

    assert seen == ["https://internal-proxy.example/pypi/demo/json"]


def test_get_packages_preserves_order_and_fetches_each() -> None:
    names = ["alpha", "beta", "gamma"]
    payloads = {n: _payload([f"{i + 1}.0.0"]) for i, n in enumerate(names)}
    seen: list[str] = []

    async def _f(session: object, url: str) -> dict[str, Any]:
        for n in names:
            if url.endswith(f"/pypi/{n}/json"):
                seen.append(n)
                return payloads[n]
        raise AssertionError(f"unexpected URL: {url}")

    with _patch_fetch(_f):
        pkgs = asyncio.run(pypi.get_packages(names))

    expected = [Version("1.0.0"), Version("2.0.0"), Version("3.0.0")]
    assert [p.latest_version for p in pkgs] == expected
    assert sorted(seen) == sorted(names)


@contextmanager
def _serve(body: str, content_type: str, requests: list[str] | None = None) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if requests is not None:
                requests.append(self.path)
            encoded = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_fetch_json_accepts_non_json_mimetype(monkeypatch: pytest.MonkeyPatch) -> None:
    # Some PyPI mirrors serve the JSON API as `text/html`
    with _serve(json.dumps(_payload(["1.0.0"])), "text/html") as url:
        monkeypatch.setenv("PYPI_URL", url)
        pkg = asyncio.run(pypi.get_package("demo"))

    assert pkg.latest_version == Version("1.0.0")


def test_fetch_json_retries_then_rejects_non_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pypi._client, "_BACKOFF_BASE", 0)
    requests: list[str] = []

    with _serve("<html>nope</html>", "text/html", requests) as url:
        monkeypatch.setenv("PYPI_URL", url)
        with pytest.raises(pypi.PyPIError, match="invalid JSON"):
            asyncio.run(pypi.get_package("demo"))

    assert len(requests) == pypi._client._RETRIES


def test_pypi_error_propagates() -> None:
    async def _f(session: object, url: str) -> dict[str, Any]:
        raise pypi.PyPIError(f"Package not found: {url}")

    with _patch_fetch(_f):
        with pytest.raises(pypi.PyPIError, match="Package not found"):
            asyncio.run(pypi.get_package("does-not-exist"))
