import http.server
import socket
import threading
from collections.abc import Iterator
from unittest import mock

import pytest

from mlflow.webhooks.delivery import _create_webhook_session
from mlflow.webhooks.ssrf import SSRFProtectionError


@pytest.fixture
def loopback_server() -> Iterator[int]:
    """A real HTTP server on loopback, standing in for an internal/metadata endpoint."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"SECRET")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="ssrf-test-loopback", daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()


def _rebind_getaddrinfo(rebind_host: str, target_host: str, target_port: int):
    """getaddrinfo that resolves rebind_host to (target_host, target_port)."""
    orig = socket.getaddrinfo

    def resolver(host, port, *args, **kwargs):
        if host == rebind_host:
            return orig(target_host, target_port, *args, **kwargs)
        return orig(host, port, *args, **kwargs)

    return resolver


def test_webhook_session_uses_ssrf_protected_adapter():
    session = _create_webhook_session()
    for scheme in ("http://", "https://"):
        adapter = session.get_adapter(f"{scheme}example.com")
        assert type(adapter).__name__ == "SSRFProtectedHTTPAdapter"


def test_webhook_session_ignores_env_proxy(loopback_server: int, monkeypatch: pytest.MonkeyPatch):
    # With trust_env left on, an HTTP_PROXY env var would route the request
    # through the proxy, making the validated peer the proxy (not the webhook
    # destination) and bypassing the SSRF check entirely. The session disables
    # env proxy handling, so the loopback "proxy" must never be reached.
    session = _create_webhook_session()
    assert session.trust_env is False
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{loopback_server}")
    resolver = _rebind_getaddrinfo("dest.example.com", "127.0.0.1", loopback_server)
    with mock.patch.object(socket, "getaddrinfo", side_effect=resolver):
        # The real destination still resolves to loopback here, so the peer
        # check fires on the destination — proving the proxy was bypassed and
        # the destination IP is what gets validated.
        with pytest.raises(SSRFProtectionError, match="not a public IP"):
            session.post("http://dest.example.com/hook", timeout=10)


def test_dns_rebinding_to_loopback_is_blocked(loopback_server: int):
    session = _create_webhook_session()
    resolver = _rebind_getaddrinfo("evil.example.com", "127.0.0.1", loopback_server)
    with mock.patch.object(socket, "getaddrinfo", side_effect=resolver):
        with pytest.raises(SSRFProtectionError, match="not a public IP"):
            session.get(f"http://evil.example.com:{loopback_server}/latest/meta-data/", timeout=10)


def test_private_ip_connection_is_blocked_by_default(loopback_server: int):
    session = _create_webhook_session()
    with pytest.raises(SSRFProtectionError, match="not a public IP"):
        session.get(f"http://127.0.0.1:{loopback_server}/", timeout=10)


def test_private_ip_allowed_when_env_var_set(loopback_server: int, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS", "true")
    session = _create_webhook_session()
    response = session.get(f"http://127.0.0.1:{loopback_server}/", timeout=10)
    assert response.status_code == 200


def test_ssrf_error_is_not_retried(loopback_server: int):
    # SSRFProtectionError is not a urllib3 exception, so it propagates on the
    # first connection attempt rather than being retried. Use POST to exercise
    # the production retry config (allowed_methods=["POST"]).
    session = _create_webhook_session()
    with mock.patch("mlflow.webhooks.ssrf.ipaddress.ip_address") as mock_ip:
        mock_ip.return_value.is_global = False
        with pytest.raises(SSRFProtectionError, match="not a public IP"):
            session.post(f"http://127.0.0.1:{loopback_server}/", timeout=10)
        # One check per connection attempt; no retry loop.
        mock_ip.assert_called_once()
