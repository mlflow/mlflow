import socket
import subprocess
import sys
from unittest import mock

import pytest
from requests.adapters import HTTPAdapter

from mlflow.utils import request_utils
from mlflow.utils.request_utils import (
    TCPKeepAliveHTTPAdapter,
    _build_socket_options,
)


def test_request_utils_does_not_import_mlflow(tmp_path):
    file_content = f"""
import importlib.util
import os
import sys

file_path = r"{request_utils.__file__}"
module_name = "mlflow.utils.request_utils"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

assert "mlflow" not in sys.modules
assert "mlflow.utils.request_utils" in sys.modules
"""
    test_file = tmp_path.joinpath("test_request_utils_does_not_import_mlflow.py")
    test_file.write_text(file_content)

    subprocess.check_call([sys.executable, str(test_file)])


class IncompleteResponse:
    def __init__(self):
        self.headers = {"Content-Length": "100"}
        raw = mock.MagicMock()
        raw.tell.return_value = 50
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_download_chunk_incomplete_read(tmp_path):
    with mock.patch.object(
        request_utils, "cloud_storage_http_request", return_value=IncompleteResponse()
    ):
        download_path = tmp_path / "chunk"
        download_path.touch()
        with pytest.raises(IOError, match="Incomplete read"):
            request_utils.download_chunk(
                range_start=0,
                range_end=999,
                headers={},
                download_path=download_path,
                http_uri="https://example.com",
            )


@pytest.mark.parametrize("env_value", ["0", "false", "False", "FALSE"])
def test_redirects_disabled_if_env_var_set(monkeypatch, env_value):
    monkeypatch.setenv("MLFLOW_ALLOW_HTTP_REDIRECTS", env_value)

    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value.status_code = 302
        mock_request.return_value.text = "mock response"

        response = request_utils.cloud_storage_http_request("GET", "http://localhost:5000")

        assert response.text == "mock response"
        mock_request.assert_called_once_with(
            "GET",
            "http://localhost:5000",
            allow_redirects=False,
            timeout=None,
        )


@pytest.mark.parametrize("env_value", ["1", "true", "True", "TRUE"])
def test_redirects_enabled_if_env_var_set(monkeypatch, env_value):
    monkeypatch.setenv("MLFLOW_ALLOW_HTTP_REDIRECTS", env_value)

    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value.status_code = 302
        mock_request.return_value.text = "mock response"

        response = request_utils.cloud_storage_http_request(
            "GET",
            "http://localhost:5000",
        )

        assert response.text == "mock response"
        mock_request.assert_called_once_with(
            "GET",
            "http://localhost:5000",
            allow_redirects=True,
            timeout=None,
        )


@pytest.mark.parametrize("env_value", ["0", "false", "False", "FALSE"])
def test_redirect_kwarg_overrides_env_value_false(monkeypatch, env_value):
    monkeypatch.setenv("MLFLOW_ALLOW_HTTP_REDIRECTS", env_value)

    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value.status_code = 302
        mock_request.return_value.text = "mock response"

        response = request_utils.cloud_storage_http_request(
            "GET", "http://localhost:5000", allow_redirects=True
        )

        assert response.text == "mock response"
        mock_request.assert_called_once_with(
            "GET",
            "http://localhost:5000",
            allow_redirects=True,
            timeout=None,
        )


@pytest.mark.parametrize("env_value", ["1", "true", "True", "TRUE"])
def test_redirect_kwarg_overrides_env_value_true(monkeypatch, env_value):
    monkeypatch.setenv("MLFLOW_ALLOW_HTTP_REDIRECTS", env_value)

    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value.status_code = 302
        mock_request.return_value.text = "mock response"

        response = request_utils.cloud_storage_http_request(
            "GET", "http://localhost:5000", allow_redirects=False
        )

        assert response.text == "mock response"
        mock_request.assert_called_once_with(
            "GET",
            "http://localhost:5000",
            allow_redirects=False,
            timeout=None,
        )


def test_redirects_enabled_by_default():
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value.status_code = 302
        mock_request.return_value.text = "mock response"

        response = request_utils.cloud_storage_http_request(
            "GET",
            "http://localhost:5000",
        )

        assert response.text == "mock response"
        mock_request.assert_called_once_with(
            "GET",
            "http://localhost:5000",
            allow_redirects=True,
            timeout=None,
        )


# --- TCP Keepalive tests ---


def test_build_socket_options_includes_keepalive():
    options = _build_socket_options()
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options


def test_build_socket_options_platform_specific():
    options = _build_socket_options()
    if hasattr(socket, "TCP_KEEPIDLE"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30) in options
    elif hasattr(socket, "TCP_KEEPALIVE"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30) in options
    if hasattr(socket, "TCP_KEEPINTVL"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10) in options
    if hasattr(socket, "TCP_KEEPCNT"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3) in options


def test_build_socket_options_disabled_via_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_TCP_KEEPALIVE", "false")
    options = _build_socket_options()
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) not in options


def test_build_socket_options_custom_values_via_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_TCP_KEEPALIVE_IDLE", "60")
    monkeypatch.setenv("MLFLOW_HTTP_TCP_KEEPALIVE_INTERVAL", "20")
    monkeypatch.setenv("MLFLOW_HTTP_TCP_KEEPALIVE_COUNT", "5")
    options = _build_socket_options()
    if hasattr(socket, "TCP_KEEPIDLE"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60) in options
    elif hasattr(socket, "TCP_KEEPALIVE"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 60) in options
    if hasattr(socket, "TCP_KEEPINTVL"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 20) in options
    if hasattr(socket, "TCP_KEEPCNT"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5) in options


def test_tcp_keepalive_adapter_init_poolmanager():
    adapter = TCPKeepAliveHTTPAdapter()
    with mock.patch.object(HTTPAdapter, "init_poolmanager") as mock_init:
        adapter.init_poolmanager(1, 1)
        mock_init.assert_called_once()
        _, kwargs = mock_init.call_args
        assert "socket_options" in kwargs
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in kwargs["socket_options"]


def test_tcp_keepalive_adapter_proxy_manager_for():
    adapter = TCPKeepAliveHTTPAdapter()
    with mock.patch.object(HTTPAdapter, "proxy_manager_for") as mock_proxy:
        adapter.proxy_manager_for("http://proxy:8080")
        mock_proxy.assert_called_once()
        _, kwargs = mock_proxy.call_args
        assert "socket_options" in kwargs
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in kwargs["socket_options"]


def test_tcp_keepalive_adapter_proxy_respects_explicit_options():
    adapter = TCPKeepAliveHTTPAdapter()
    custom_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 0)]
    with mock.patch.object(HTTPAdapter, "proxy_manager_for") as mock_proxy:
        adapter.proxy_manager_for("http://proxy:8080", socket_options=custom_options)
        _, kwargs = mock_proxy.call_args
        assert kwargs["socket_options"] == custom_options


def test_session_uses_tcp_keepalive_adapter():
    request_utils._cached_get_request_session.cache_clear()
    session = request_utils._get_request_session(
        max_retries=3,
        backoff_factor=1,
        backoff_jitter=0.5,
        retry_codes=(500,),
        raise_on_status=True,
        respect_retry_after_header=True,
    )
    assert isinstance(session.get_adapter("https://example.com"), TCPKeepAliveHTTPAdapter)
    assert isinstance(session.get_adapter("http://example.com"), TCPKeepAliveHTTPAdapter)
    request_utils._cached_get_request_session.cache_clear()
