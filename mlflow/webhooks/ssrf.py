"""Connection-time SSRF protection for outbound webhook delivery.

``_validate_webhook_url`` resolves the webhook hostname and checks that every
resolved IP is public, but it discards the resolved IP and the subsequent
``requests.post`` re-resolves the hostname independently. That TOCTOU gap lets a
DNS-rebinding attacker return a public IP during validation and a private or
link-local IP (e.g. ``169.254.169.254``) at request time, reaching cloud
metadata or internal services.

The pieces below close the gap by validating the peer IP of the *actual
connected socket* immediately after ``connect()`` returns, before any TLS
handshake or HTTP data is exchanged. There is no second DNS resolution between
the check and the request, so rebinding has nothing to exploit. Because the TLS
handshake still runs against the original hostname, certificate validation is
unaffected (unlike pinning the URL to a resolved IP).

Everything here is scoped to the webhook delivery session via
``SSRFProtectedHTTPAdapter`` so it never alters urllib3 behavior process-wide.
"""

import ipaddress
import socket

from requests.adapters import DEFAULT_POOLBLOCK, HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.poolmanager import PoolManager, ProxyManager

from mlflow.environment_variables import _MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS


class SSRFProtectionError(Exception):
    """Raised when a webhook connection resolves to a non-public IP address.

    Not a urllib3 exception type, so it is never retried and propagates
    immediately: the delivery fails closed.
    """


def _assert_public_peer(sock: socket.socket) -> None:
    if _MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS.get():
        return

    # getpeername() returns (host, port, ...) for both IPv4 and IPv6 sockets.
    # It can raise OSError (e.g. ENOTCONN) on an unconnected socket; fail closed.
    try:
        peer_ip = sock.getpeername()[0]
    except OSError as e:
        sock.close()
        raise SSRFProtectionError(
            f"Could not determine webhook connection peer address: {e}"
        ) from e
    try:
        ip = ipaddress.ip_address(peer_ip)
    except ValueError as e:
        sock.close()
        raise SSRFProtectionError(
            f"Webhook connection resolved to an invalid IP address: {peer_ip!r}"
        ) from e

    if not ip.is_global:
        sock.close()
        raise SSRFProtectionError(
            f"Webhook connection blocked: {ip} is not a public IP address. "
            "This may indicate a DNS rebinding attempt."
        )


class _SSRFProtectedHTTPConnection(HTTPConnection):
    def _new_conn(self) -> socket.socket:
        sock = super()._new_conn()
        _assert_public_peer(sock)
        return sock


class _SSRFProtectedHTTPSConnection(HTTPSConnection):
    def _new_conn(self) -> socket.socket:
        # HTTPSConnection inherits _new_conn from HTTPConnection: it returns the
        # raw TCP socket before the TLS handshake, so the IP check runs pre-TLS.
        sock = super()._new_conn()
        _assert_public_peer(sock)
        return sock


class _SSRFProtectedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _SSRFProtectedHTTPConnection


class _SSRFProtectedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _SSRFProtectedHTTPSConnection


# Instance-local override installed on any (Pool|Proxy)Manager below, so only
# pools created by these managers get the SSRF-protected connection classes.
_SSRF_POOL_CLASSES = {
    "http": _SSRFProtectedHTTPConnectionPool,
    "https": _SSRFProtectedHTTPSConnectionPool,
}


class _SSRFProtectedPoolManager(PoolManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool_classes_by_scheme = _SSRF_POOL_CLASSES


class _SSRFProtectedProxyManager(ProxyManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool_classes_by_scheme = _SSRF_POOL_CLASSES


class SSRFProtectedHTTPAdapter(HTTPAdapter):
    """``HTTPAdapter`` that validates each connection's peer IP is public.

    Both the direct pool manager and the proxy manager use SSRF-protected pools,
    so the peer-IP check runs whether or not a proxy is configured.
    """

    def init_poolmanager(
        self, connections, maxsize, block=DEFAULT_POOLBLOCK, **pool_kwargs
    ) -> None:
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block
        self.poolmanager = _SSRFProtectedPoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        if proxy not in self.proxy_manager:
            proxy_headers = self.proxy_headers(proxy)
            self.proxy_manager[proxy] = _SSRFProtectedProxyManager(
                proxy,
                proxy_headers=proxy_headers,
                num_pools=self._pool_connections,
                maxsize=self._pool_maxsize,
                block=self._pool_block,
                **proxy_kwargs,
            )
        return self.proxy_manager[proxy]
