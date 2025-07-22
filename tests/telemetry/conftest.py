import time
from unittest.mock import Mock, patch

import pytest

import mlflow
import mlflow.telemetry.utils
from mlflow.telemetry.client import get_telemetry_client, set_telemetry_client
from mlflow.version import VERSION


@pytest.fixture(autouse=True)
def terminate_telemetry_client():
    yield
    client = get_telemetry_client()
    if client:
        client.flush(terminate=True)


@pytest.fixture
def mock_requests():
    """Fixture to mock requests.post and capture telemetry records."""
    captured_records = []

    def mock_post(url, json=None, **kwargs):
        if url == "http://127.0.0.1:9999/nonexistent":
            mock_response = Mock()
            mock_response.status_code = 404
            return mock_response
        if json and "records" in json:
            captured_records.extend(json["records"])
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "count": len(json.get("records", [])) if json else 0,
        }
        return mock_response

    with patch("requests.post", side_effect=mock_post):
        yield captured_records


@pytest.fixture(autouse=True)
def enable_telemetry(monkeypatch):
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_IN_CI_ENV_OR_TESTING", False)
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_DEV_VERSION", False)


@pytest.fixture(autouse=True)
def mock_requests_get(request, monkeypatch):
    """Fixture to mock requests.get and capture telemetry records."""
    if request.node.get_closest_marker("no_mock_requests_get"):
        yield
        return

    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_IN_CI_ENV_OR_TESTING", False)
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_DEV_VERSION", False)
    with patch("mlflow.telemetry.client.requests.get") as mock_get:
        mock_get.return_value = Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_api_map": {},
                    "disable_sdks": [],
                }
            ),
        )
        set_telemetry_client()
        client = get_telemetry_client()
        # ensure config is fetched before the test
        while not client._is_config_fetched:
            time.sleep(0.1)
        yield
        client.flush(terminate=True)
