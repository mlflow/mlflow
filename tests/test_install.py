import importlib.util
import sys
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

import pytest

_INSTALL_PATH = Path(__file__).parents[1] / "bin" / "install.py"
_SPEC = importlib.util.spec_from_file_location("install", _INSTALL_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
install = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = install
_SPEC.loader.exec_module(install)


def _http_error(status_code: int) -> HTTPError:
    return HTTPError("https://example.com", status_code, "error", {}, None)


def test_urlopen_with_retry_retries_http_500():
    response = mock.Mock()
    with (
        mock.patch.object(
            install.urllib.request,
            "urlopen",
            side_effect=[_http_error(500), response],
        ) as urlopen,
        mock.patch.object(install.time, "sleep") as sleep,
    ):
        assert install.urlopen_with_retry("https://example.com", base_delay=0) is response

    assert urlopen.call_count == 2
    sleep.assert_called_once_with(0)


def test_urlopen_with_retry_raises_after_http_500_retries_are_exhausted():
    with (
        mock.patch.object(
            install.urllib.request, "urlopen", side_effect=_http_error(500)
        ) as urlopen,
        mock.patch.object(install.time, "sleep") as sleep,
        pytest.raises(HTTPError, match="500"),
    ):
        install.urlopen_with_retry("https://example.com", max_retries=3, base_delay=0)

    assert urlopen.call_count == 3
    assert sleep.call_count == 2
