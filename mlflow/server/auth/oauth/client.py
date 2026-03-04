import logging
import os
import time

import requests

from mlflow.server.auth.client import AuthServiceClient

_logger = logging.getLogger(__name__)


class OAuthServiceClient(AuthServiceClient):
    def __init__(self, tracking_uri: str):
        super().__init__(tracking_uri)
        self._token = None
        self._token_expiry = 0

    def _get_token(self) -> str:
        # Direct bearer token from environment
        if env_token := os.environ.get("MLFLOW_TRACKING_TOKEN", ""):
            return env_token

        # Client credentials grant
        client_id = os.environ.get("MLFLOW_TRACKING_CLIENT_ID", "")
        client_secret = os.environ.get("MLFLOW_TRACKING_CLIENT_SECRET", "")
        token_url = os.environ.get("MLFLOW_TRACKING_TOKEN_URL", "")

        if client_id and client_secret and token_url:
            now = time.time()
            if self._token and now < self._token_expiry - 60:
                return self._token

            resp = requests.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=10,
            )
            resp.raise_for_status()
            token_data = resp.json()
            self._token = token_data["access_token"]
            self._token_expiry = now + token_data.get("expires_in", 3600)
            return self._token

        # Device code flow
        device_client_id = os.environ.get("MLFLOW_TRACKING_DEVICE_CLIENT_ID", "")
        device_token_url = os.environ.get("MLFLOW_TRACKING_DEVICE_TOKEN_URL", "")
        device_auth_url = os.environ.get("MLFLOW_TRACKING_DEVICE_AUTH_URL", "")

        if device_client_id and device_token_url and device_auth_url:
            now = time.time()
            if self._token and now < self._token_expiry - 60:
                return self._token

            token = _device_code_flow(
                client_id=device_client_id,
                device_auth_url=device_auth_url,
                token_url=device_token_url,
                scopes=os.environ.get("MLFLOW_TRACKING_DEVICE_SCOPES", "openid profile email"),
            )
            if token:
                self._token = token["access_token"]
                self._token_expiry = time.time() + token.get("expires_in", 3600)
                return self._token

        return ""

    def _request(self, endpoint, method, *, expected_status: int = 200, **kwargs):
        if token := self._get_token():
            headers = kwargs.pop("extra_headers", {})
            headers["Authorization"] = f"Bearer {token}"
            kwargs["extra_headers"] = headers

        return super()._request(endpoint, method, expected_status=expected_status, **kwargs)


def _device_code_flow(
    client_id: str,
    device_auth_url: str,
    token_url: str,
    scopes: str = "openid profile email",
) -> dict[str, object] | None:
    # Step 1: Request device code
    resp = requests.post(
        device_auth_url,
        data={
            "client_id": client_id,
            "scope": scopes,
        },
        timeout=10,
    )
    resp.raise_for_status()
    device_data = resp.json()

    device_code = device_data["device_code"]
    user_code = device_data["user_code"]
    verification_uri = (
        device_data.get("verification_uri_complete") or device_data["verification_uri"]
    )
    interval = device_data.get("interval", 5)
    expires_in = device_data.get("expires_in", 600)

    # Step 2: Display instructions to user
    _logger.info(
        "To authenticate, visit:\n  %s\n\nAnd enter code: %s\n\nWaiting for authorization...",
        verification_uri,
        user_code,
    )

    # Step 3: Poll for token
    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(interval)

        try:
            token_resp = requests.post(
                token_url,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": client_id,
                },
                timeout=10,
            )
        except requests.exceptions.RequestException:
            continue

        if token_resp.status_code == 200:
            tokens = token_resp.json()
            _logger.info("Authentication successful.")
            return tokens

        error_data = (
            token_resp.json()
            if token_resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        error = error_data.get("error", "")

        if error == "authorization_pending":
            continue
        elif error == "slow_down":
            interval = min(interval + 5, 60)
            continue
        elif error in ("expired_token", "access_denied"):
            _logger.error("Device code flow failed: %s", error)
            _logger.info("Authentication failed: %s", error)
            return None
        else:
            _logger.error("Device code flow unexpected error: %s", error_data)
            return None

    _logger.info("Device code expired. Please try again.")
    return None
