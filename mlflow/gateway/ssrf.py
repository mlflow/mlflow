"""Connect-time SSRF protection for AI Gateway upstream calls.

Write-time validation of a secret's ``api_base`` cannot stop DNS rebinding, redirects to
internal hosts, or rows stored before it existed. This module enforces the same policy at
the egress point: ``SSRFGuardedResolver`` rejects non-public addresses before aiohttp dials
them, ``assert_public_upstream_url`` covers IP literals (which aiohttp never resolves), and
``assert_public_upstream_host`` is the pre-call check for LiteLLM's own HTTP client.
``_aiohttp_post`` never follows redirects.

Enforcement is per request via ``upstream_ssrf_protection`` (set in
``mlflow.server.gateway_api._enable_upstream_ssrf_protection``) and is skipped entirely
when ``MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS`` is true.
"""

import asyncio
import ipaddress
import socket
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlparse

import aiohttp
from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

from mlflow.environment_variables import MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS
from mlflow.utils.validation import _is_ip_literal_like, _is_public_ip


class GatewaySSRFProtectionError(Exception):
    """Raised when an upstream connection would target a non-public IP address.

    Not an ``OSError`` subclass, so aiohttp propagates it unchanged instead of wrapping it.
    """


# Request-scoped: set by mlflow.server.gateway_api once the endpoint config is known, so
# every provider call for that request, including streamed bodies, observes it.
upstream_ssrf_protection: ContextVar[bool] = ContextVar(
    "gateway_upstream_ssrf_protection", default=False
)


def _is_protection_enabled() -> bool:
    return upstream_ssrf_protection.get() and not MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS.get()


def _assert_public_ip(ip_str: str, host: str) -> None:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError as e:
        raise GatewaySSRFProtectionError(
            f"Gateway upstream {host!r} resolved to an invalid IP address: {ip_str!r}"
        ) from e
    if not _is_public_ip(ip):
        raise GatewaySSRFProtectionError(
            f"Gateway upstream connection blocked: {host!r} resolves to {ip}, which is not a "
            "public IP address. Set MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS=true to allow "
            "private upstreams."
        )


def _parse_upstream_hostname(url: str) -> str:
    try:
        hostname = urlparse(url).hostname
    except ValueError as e:
        raise GatewaySSRFProtectionError(f"Invalid gateway upstream URL {url!r}: {e}") from e
    if not hostname:
        raise GatewaySSRFProtectionError(f"Gateway upstream URL must include a hostname: {url!r}")
    return hostname


def assert_public_upstream_url(url: str) -> None:
    """Reject an upstream URL whose host is a non-public or non-canonical IP literal.

    Hostnames are left to ``SSRFGuardedResolver`` at connect time. Anything aiohttp would
    dial as a literal must parse canonically, since aiohttp skips the resolver for it and
    ``socket`` would silently map a legacy spelling such as ``127.1`` onto an address.
    """
    if not _is_protection_enabled():
        return
    hostname = _parse_upstream_hostname(url)
    if not _is_ip_literal_like(hostname):
        return
    try:
        ipaddress.ip_address(hostname)
    except ValueError as e:
        raise GatewaySSRFProtectionError(
            f"Gateway upstream host {hostname!r} is not a canonical IP address literal."
        ) from e
    _assert_public_ip(hostname, hostname)


async def _getaddrinfo(hostname: str) -> list[Any]:
    return await asyncio.get_running_loop().getaddrinfo(hostname, None)


async def assert_public_upstream_host(url: str) -> None:
    """Resolve an upstream URL's host and reject it unless every address is public.

    For clients that bypass ``_aiohttp_post`` (LiteLLM). The lookup is separate from the
    client's own, so a DNS-rebinding window remains between the two.
    """
    if not _is_protection_enabled():
        return
    assert_public_upstream_url(url)
    hostname = _parse_upstream_hostname(url)
    if _is_ip_literal_like(hostname):
        return
    try:
        addr_infos = await _getaddrinfo(hostname)
    except (OSError, UnicodeError, ValueError) as e:
        raise GatewaySSRFProtectionError(
            f"Cannot resolve gateway upstream host {hostname!r}: {e}"
        ) from e
    for addr_info in addr_infos:
        _assert_public_ip(addr_info[4][0], hostname)


class SSRFGuardedResolver(AbstractResolver):
    """aiohttp resolver that only returns public addresses.

    aiohttp dials exactly what the resolver returns, so this checks the real connection target.
    """

    def __init__(self, resolver: AbstractResolver | None = None) -> None:
        self._resolver = resolver or DefaultResolver()

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[Any]:
        results = await self._resolver.resolve(host, port, family=family)
        for result in results:
            _assert_public_ip(result["host"], host)
        return results

    async def close(self) -> None:
        await self._resolver.close()


def build_ssrf_guarded_connector() -> aiohttp.TCPConnector | None:
    """Return a guarded connector, or ``None`` for aiohttp's default when protection is off."""
    if not _is_protection_enabled():
        return None
    return aiohttp.TCPConnector(resolver=SSRFGuardedResolver())
