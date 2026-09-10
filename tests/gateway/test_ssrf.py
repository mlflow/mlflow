import socket
from unittest import mock

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import HTTPException

from mlflow.gateway.providers.utils import _aiohttp_post, send_request
from mlflow.gateway.ssrf import (
    GatewaySSRFProtectionError,
    SSRFGuardedResolver,
    assert_public_upstream_url,
    build_ssrf_guarded_connector,
    upstream_ssrf_protection,
)


@pytest.fixture(autouse=True)
def protected_request():
    # Simulates running inside a tracking-server request, where the middleware turns the
    # upstream guard on.
    token = upstream_ssrf_protection.set(True)
    try:
        yield
    finally:
        upstream_ssrf_protection.reset(token)


def _resolve_result(ip: str, port: int = 0):
    return {
        "hostname": "upstream.example",
        "host": ip,
        "port": port,
        "family": socket.AF_INET6 if ":" in ip else socket.AF_INET,
        "proto": 0,
        "flags": 0,
    }


def _inner_resolver(*ips: str):
    resolver = mock.AsyncMock()
    resolver.resolve.return_value = [_resolve_result(ip) for ip in ips]
    return resolver


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "::1",
        "fc00::1",
        "64:ff9b::169.254.169.254",
        "::ffff:10.0.0.1",
    ],
)
@pytest.mark.asyncio
async def test_guarded_resolver_rejects_private_addresses(ip):
    resolver = SSRFGuardedResolver(_inner_resolver(ip))
    with pytest.raises(GatewaySSRFProtectionError, match="not a public IP address"):
        await resolver.resolve("upstream.example", 443)


@pytest.mark.asyncio
async def test_guarded_resolver_rejects_if_any_address_is_private():
    resolver = SSRFGuardedResolver(_inner_resolver("8.8.8.8", "10.0.0.1"))
    with pytest.raises(GatewaySSRFProtectionError, match="10.0.0.1"):
        await resolver.resolve("upstream.example", 443)


@pytest.mark.asyncio
async def test_guarded_resolver_returns_public_addresses():
    inner = _inner_resolver("8.8.8.8", "2001:4860:4860::8888")
    resolver = SSRFGuardedResolver(inner)
    results = await resolver.resolve("upstream.example", 443, family=socket.AF_UNSPEC)
    assert [r["host"] for r in results] == ["8.8.8.8", "2001:4860:4860::8888"]
    inner.resolve.assert_awaited_once_with("upstream.example", 443, family=socket.AF_UNSPEC)
    await resolver.close()
    inner.close.assert_awaited_once()


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/v1",
        "http://169.254.169.254/latest/meta-data/",
        "https://[::1]:8443/v1",
        "https://[64:ff9b::169.254.169.254]/v1",
        "https://192.168.1.10:11434/v1",
    ],
)
def test_assert_public_upstream_url_rejects_private_ip_literals(url):
    with pytest.raises(GatewaySSRFProtectionError, match="not a public IP address"):
        assert_public_upstream_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "https://8.8.8.8/v1",
        "https://[2001:4860:4860::8888]/v1",
        # Hostnames are deferred to the resolver, so no DNS lookup happens here.
        "https://localhost:11434/v1",
    ],
)
def test_assert_public_upstream_url_accepts_public_literals_and_hostnames(url):
    with mock.patch("socket.getaddrinfo") as getaddrinfo:
        assert_public_upstream_url(url)
    getaddrinfo.assert_not_called()


def test_assert_public_upstream_url_rejects_missing_hostname():
    with pytest.raises(GatewaySSRFProtectionError, match="must include a hostname"):
        assert_public_upstream_url("https:///v1")


def test_assert_public_upstream_url_skipped_when_private_ips_allowed(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS", "true")
    assert_public_upstream_url("http://169.254.169.254/latest/meta-data/")


@pytest.mark.asyncio
async def test_build_connector_uses_guarded_resolver_by_default():
    connector = build_ssrf_guarded_connector()
    try:
        assert isinstance(connector._resolver, SSRFGuardedResolver)
    finally:
        await connector.close()


def test_build_connector_disabled_when_private_ips_allowed(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS", "true")
    assert build_ssrf_guarded_connector() is None


def test_protection_is_off_outside_a_protected_request():
    # The standalone gateway and direct provider use are not tracking-server requests, so
    # they keep aiohttp's default resolver and may target private upstreams.
    token = upstream_ssrf_protection.set(False)
    try:
        assert build_ssrf_guarded_connector() is None
        assert_public_upstream_url("http://127.0.0.1:11434/v1")
    finally:
        upstream_ssrf_protection.reset(token)


# -- End-to-end through the gateway's egress choke point against a real local upstream --


@pytest_asyncio.fixture
async def upstream():
    hits = []

    async def handler(request):
        hits.append(request.path)
        return web.json_response({"ok": True, "path": request.path})

    async def redirect(request):
        hits.append(request.path)
        raise web.HTTPFound("/internal/secret")

    app = web.Application()
    app.router.add_post("/redirect", redirect)
    app.router.add_post("/{tail:.*}", handler)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    server.hits = hits
    try:
        yield server
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_send_request_blocks_private_ip_literal_upstream(upstream):
    with pytest.raises(HTTPException, match="not a public IP address") as exc:
        await send_request({}, str(upstream.make_url("")), "/v1/chat", {"model": "x"})
    assert exc.value.status_code == 502
    assert "not a public IP address" in exc.value.detail
    assert upstream.hits == []


@pytest.mark.asyncio
async def test_send_request_blocks_hostname_that_resolves_to_private_ip(upstream):
    # Simulates DNS rebinding: whatever the hostname resolved to when the secret was
    # written, the address handed to the connector at request time is private.
    rebinding = _inner_resolver("127.0.0.1")
    rebinding.resolve.return_value = [_resolve_result("127.0.0.1", upstream.port)]
    with mock.patch("mlflow.gateway.ssrf.DefaultResolver", return_value=rebinding):
        with pytest.raises(HTTPException, match="rebind.example") as exc:
            await send_request(
                {}, f"http://rebind.example:{upstream.port}", "/v1/chat", {"model": "x"}
            )
    assert exc.value.status_code == 502
    assert "'rebind.example' resolves to 127.0.0.1" in exc.value.detail
    rebinding.resolve.assert_awaited()
    assert upstream.hits == []


@pytest.mark.asyncio
async def test_send_request_reaches_private_upstream_when_allowed(upstream, monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS", "true")
    result = await send_request({}, str(upstream.make_url("")), "/v1/chat", {"model": "x"})
    assert result == {"ok": True, "path": "/v1/chat"}
    assert upstream.hits == ["/v1/chat"]


@pytest.mark.asyncio
async def test_aiohttp_post_does_not_follow_upstream_redirects(upstream, monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS", "true")
    async with _aiohttp_post({}, str(upstream.make_url("")), "/redirect", {}) as response:
        assert response.status == 302
        assert response.headers["Location"] == "/internal/secret"
        assert response.history == ()
    assert upstream.hits == ["/redirect"]


@pytest.mark.asyncio
async def test_send_request_explicit_bypass_reaches_private_upstream(upstream):
    # Used for the server calling itself (guardrail sanitization), never for provider hosts.
    result = await send_request(
        {}, str(upstream.make_url("")), "/v1/chat", {"model": "x"}, ssrf_protect=False
    )
    assert result == {"ok": True, "path": "/v1/chat"}
    assert upstream.hits == ["/v1/chat"]
