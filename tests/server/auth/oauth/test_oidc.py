import hashlib
from base64 import urlsafe_b64encode
from unittest import mock

import pytest

from mlflow.server.auth.oauth.oidc import _fetch_discovery, _generate_pkce


def test_pkce_generate_format():
    verifier, challenge = _generate_pkce()
    # Verifier should be url-safe base64
    assert len(verifier) > 40
    # Challenge should be S256 of verifier
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert challenge == expected


def test_pkce_generate_unique():
    pairs = [_generate_pkce() for _ in range(10)]
    verifiers = {v for v, _ in pairs}
    assert len(verifiers) == 10


def test_fetch_discovery_caches_result():
    from mlflow.server.auth.oauth.oidc import _discovery_cache

    _discovery_cache.clear()

    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "authorization_endpoint": "https://idp.example.com/authorize",
        "token_endpoint": "https://idp.example.com/token",
        "jwks_uri": "https://idp.example.com/.well-known/jwks.json",
        "issuer": "https://idp.example.com",
    }
    mock_resp.raise_for_status = mock.Mock()

    url = "https://idp.example.com/.well-known/openid-configuration"
    with mock.patch("requests.get", return_value=mock_resp) as mock_get:
        result1 = _fetch_discovery(url)
        result2 = _fetch_discovery(url)
        mock_get.assert_called_once()

    assert result1["issuer"] == "https://idp.example.com"
    assert result1 is result2

    _discovery_cache.clear()


def test_fetch_discovery_raises_on_http_error():
    from mlflow.server.auth.oauth.oidc import _discovery_cache

    _discovery_cache.clear()

    mock_resp = mock.Mock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = Exception("Server error")

    with mock.patch("requests.get", return_value=mock_resp):
        with pytest.raises(Exception, match="Server error"):
            _fetch_discovery("https://idp.example.com/.well-known/openid-configuration")

    _discovery_cache.clear()
