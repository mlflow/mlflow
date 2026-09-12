"""Connection-time SSRF protection for AI Gateway upstream calls.

Gateway secrets can carry a user-supplied ``api_base`` that the gateway sends requests to,
and the raw proxy route additionally appends a caller-supplied path. Write-time validation
of ``api_base`` (``mlflow.utils.validation._validate_gateway_api_base``) resolves the
hostname once and discards the result, so on its own it cannot stop:

- a DNS-rebinding attacker who returns a public IP during validation and a private or
  link-local IP (e.g. ``169.254.169.254``) at request time;
- an upstream that answers with a redirect to an internal address;
- rows that were written before write-time validation existed.

This module closes those gaps at the egress point instead:

- ``SSRFGuardedResolver`` wraps aiohttp's default resolver and rejects any resolved address
  that is not public. aiohttp connects to exactly the addresses the resolver returns, so
  there is no second lookup between the check and the connection.
- ``assert_public_upstream_url`` covers IP-literal hosts, which aiohttp dials without
  consulting the resolver, and rejects non-canonical numeric spellings (``2130706433``,
  ``127.1``, ``0177.0.0.1``) that aiohttp also treats as literals but ``ipaddress`` cannot
  parse, so they would otherwise slip past both checks.
- ``_aiohttp_post`` disables redirect following, so a redirect can never introduce a host
  that bypasses the checks above.
- ``assert_public_upstream_host`` resolves and checks a host for providers that use their
  own HTTP client instead of ``_aiohttp_post`` (LiteLLM). That lookup is separate from the
  client's own, so it cannot pin the connection the way the resolver above does; it still
  blocks plain private hosts, non-canonical literals and rows stored before write-time
  validation existed.

Enforcement is scoped to requests served by the tracking server: its middleware sets
``upstream_ssrf_protection`` for every request it handles, because there the upstream host
comes from user-created gateway secrets. The standalone ``mlflow gateway`` server reads its
provider configuration from an operator-owned file and is not affected. Every check is also
skipped when ``MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS`` is true, which is required for
deployments that legitimately talk to private upstreams (self-hosted Ollama, vLLM, and
similar).
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
    """Raised when an upstream gateway connection would target a non-public IP address.

    Deliberately not an ``OSError`` subclass: aiohttp only wraps ``OSError`` from the
    resolver, so this propagates unchanged and the request fails closed.
    """


# True while handling a request whose upstream targets may be user-controlled. Set by the
# tracking server's FastAPI middleware (see mlflow.server.fastapi_app); the handler task
# inherits a copy of the middleware's context, so provider calls made while serving the
# request, including streamed response bodies, observe it.
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

    Hostnames are intentionally not resolved here; they are checked by
    ``SSRFGuardedResolver`` at connection time so there is no window between the check and
    the connection. Anything aiohttp would dial as a literal must parse as a canonical
    address, since aiohttp skips the resolver for such hosts and ``socket`` would map a
    legacy numeric spelling onto an address unseen. No-op when private upstreams are allowed.
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

    For provider clients that do not go through ``_aiohttp_post`` (LiteLLM uses its own HTTP
    client), so ``SSRFGuardedResolver`` cannot sit on the connection. The lookup here is
    separate from the client's own, which leaves a DNS-rebinding attacker a narrow window
    between the two; it still blocks plain private hosts, non-canonical literals and rows
    stored before write-time validation existed. No-op when private upstreams are allowed.
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

    aiohttp dials the addresses this resolver returns, so validating them here is a check on
    the actual connection target rather than on an earlier, separate lookup.
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
    """Return a connector whose DNS results are checked, or ``None`` to use aiohttp's default
    when private upstreams are allowed.
    """
    if not _is_protection_enabled():
        return None
    return aiohttp.TCPConnector(resolver=SSRFGuardedResolver())
