from unittest import mock

from mlflow.server.auth.oauth.client import OAuthServiceClient, _device_code_flow


def test_oauth_service_client_bearer_token_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "my-token")
    client = OAuthServiceClient("http://localhost:5000")
    assert client._get_token() == "my-token"


def test_oauth_service_client_client_credentials_grant(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_CLIENT_ID", "client-id")
    monkeypatch.setenv("MLFLOW_TRACKING_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN_URL", "https://idp.example.com/token")

    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"access_token": "fetched-token", "expires_in": 3600}
    mock_resp.raise_for_status = mock.Mock()

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client = OAuthServiceClient("http://localhost:5000")
        token = client._get_token()
        mock_post.assert_called_once()

    assert token == "fetched-token"


def test_oauth_service_client_client_credentials_caching(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_CLIENT_ID", "client-id")
    monkeypatch.setenv("MLFLOW_TRACKING_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN_URL", "https://idp.example.com/token")

    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"access_token": "fetched-token", "expires_in": 3600}
    mock_resp.raise_for_status = mock.Mock()

    with mock.patch("requests.post", return_value=mock_resp) as mock_post:
        client = OAuthServiceClient("http://localhost:5000")
        client._get_token()
        client._get_token()
        mock_post.assert_called_once()


def test_oauth_service_client_no_credentials_returns_empty(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_CLIENT_ID", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_DEVICE_CLIENT_ID", raising=False)
    client = OAuthServiceClient("http://localhost:5000")
    assert client._get_token() == ""


def test_oauth_service_client_token_env_takes_priority(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "direct-token")
    monkeypatch.setenv("MLFLOW_TRACKING_CLIENT_ID", "client-id")
    monkeypatch.setenv("MLFLOW_TRACKING_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN_URL", "https://idp.example.com/token")

    client = OAuthServiceClient("http://localhost:5000")
    assert client._get_token() == "direct-token"


def test_device_code_flow_successful():
    device_resp = mock.Mock()
    device_resp.status_code = 200
    device_resp.json.return_value = {
        "device_code": "dc123",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://idp.example.com/device",
        "interval": 0,
        "expires_in": 600,
    }
    device_resp.raise_for_status = mock.Mock()

    token_pending = mock.Mock()
    token_pending.status_code = 400
    token_pending.headers = {"content-type": "application/json"}
    token_pending.json.return_value = {"error": "authorization_pending"}

    token_ok = mock.Mock()
    token_ok.status_code = 200
    token_ok.headers = {"content-type": "application/json"}
    token_ok.json.return_value = {"access_token": "final-token", "expires_in": 3600}

    with (
        mock.patch("requests.post", side_effect=[device_resp, token_pending, token_ok]),
        mock.patch("time.sleep"),
    ):
        result = _device_code_flow(
            client_id="test-client",
            device_auth_url="https://idp.example.com/device/code",
            token_url="https://idp.example.com/token",
        )

    assert result is not None
    assert result["access_token"] == "final-token"


def test_device_code_flow_access_denied():
    device_resp = mock.Mock()
    device_resp.status_code = 200
    device_resp.json.return_value = {
        "device_code": "dc123",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://idp.example.com/device",
        "interval": 0,
        "expires_in": 600,
    }
    device_resp.raise_for_status = mock.Mock()

    denied_resp = mock.Mock()
    denied_resp.status_code = 400
    denied_resp.headers = {"content-type": "application/json"}
    denied_resp.json.return_value = {"error": "access_denied"}

    with (
        mock.patch("requests.post", side_effect=[device_resp, denied_resp]),
        mock.patch("time.sleep"),
    ):
        result = _device_code_flow(
            client_id="test-client",
            device_auth_url="https://idp.example.com/device/code",
            token_url="https://idp.example.com/token",
        )

    assert result is None


def test_device_code_flow_slow_down_increases_interval():
    device_resp = mock.Mock()
    device_resp.status_code = 200
    device_resp.json.return_value = {
        "device_code": "dc123",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://idp.example.com/device",
        "interval": 5,
        "expires_in": 600,
    }
    device_resp.raise_for_status = mock.Mock()

    slow_down = mock.Mock()
    slow_down.status_code = 400
    slow_down.headers = {"content-type": "application/json"}
    slow_down.json.return_value = {"error": "slow_down"}

    token_ok = mock.Mock()
    token_ok.status_code = 200
    token_ok.headers = {"content-type": "application/json"}
    token_ok.json.return_value = {"access_token": "token", "expires_in": 3600}

    sleep_calls = []

    with (
        mock.patch("requests.post", side_effect=[device_resp, slow_down, token_ok]),
        mock.patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)),
    ):
        _device_code_flow(
            client_id="test",
            device_auth_url="https://idp.example.com/device/code",
            token_url="https://idp.example.com/token",
        )

    # After slow_down, interval should increase from 5 to 10
    assert any(s >= 10 for s in sleep_calls)


def test_device_code_flow_verification_uri_complete():
    device_resp = mock.Mock()
    device_resp.status_code = 200
    device_resp.json.return_value = {
        "device_code": "dc123",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://idp.example.com/device",
        "verification_uri_complete": "https://idp.example.com/device?user_code=ABCD-EFGH",
        "interval": 0,
        "expires_in": 600,
    }
    device_resp.raise_for_status = mock.Mock()

    token_ok = mock.Mock()
    token_ok.status_code = 200
    token_ok.headers = {"content-type": "application/json"}
    token_ok.json.return_value = {"access_token": "token", "expires_in": 3600}

    with (
        mock.patch("requests.post", side_effect=[device_resp, token_ok]),
        mock.patch("time.sleep"),
    ):
        result = _device_code_flow(
            client_id="test",
            device_auth_url="https://idp.example.com/device/code",
            token_url="https://idp.example.com/token",
        )

    assert result["access_token"] == "token"
