import pytest
from starlette.requests import Request

from mlflow.server.asgi_utils import get_server_base_url, get_server_self_url


def _request(server, host_header=b"10.attacker.example:8080"):
    return Request({
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/ajax-api/3.0/mlflow/assistant/sessions/abc/stream",
        "query_string": b"",
        "headers": [(b"host", host_header)],
        "server": server,
        "client": ("10.0.0.9", 50000),
    })


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        (("127.0.0.1", 5000), "http://127.0.0.1:5000"),
        (("0.0.0.0", 5000), "http://127.0.0.1:5000"),
        (("10.0.0.5", 5000), "http://10.0.0.5:5000"),
        (("::1", 5000), "http://[::1]:5000"),
        (("::", 5000), "http://[::1]:5000"),
        (("fd00::5", 5000), "http://[fd00::5]:5000"),
        (("::ffff:10.0.0.5", 5000), "http://10.0.0.5:5000"),
        (("testserver", 80), "http://testserver:80"),
    ],
)
def test_get_server_self_url_uses_the_server_address_not_the_host_header(server, expected):
    """A self-call carries server-internal credentials, so a client-chosen Host header must not
    be able to redirect it (GHSA-98p3-crmr-mxcx); the client-facing base URL keeps following it.
    """
    request = _request(server)

    assert get_server_base_url(request) == "http://10.attacker.example:8080"
    assert get_server_self_url(request) == expected


def test_get_server_self_url_appends_static_prefix(monkeypatch):
    monkeypatch.setenv("_MLFLOW_STATIC_PREFIX", "/mlflow/")

    assert get_server_self_url(_request(("127.0.0.1", 5000))) == "http://127.0.0.1:5000/mlflow"


def test_get_server_self_url_uses_https_when_the_listener_terminates_tls(monkeypatch):
    monkeypatch.setenv("_MLFLOW_SERVER_TLS_LISTENER", "true")

    assert get_server_self_url(_request(("127.0.0.1", 5000))) == "https://127.0.0.1:5000"


@pytest.mark.parametrize("server", [None, ("/tmp/mlflow.sock", None)])
def test_get_server_self_url_is_none_without_a_server_address(server):
    assert get_server_self_url(_request(server)) is None
