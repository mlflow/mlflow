from unittest.mock import Mock, patch

import pytest

import mlflow
import mlflow.telemetry.utils
from mlflow.telemetry.client import TelemetryClient, _set_telemetry_client, get_telemetry_client
from mlflow.version import VERSION


@pytest.fixture(autouse=True)
def terminate_telemetry_client():
    yield
    client = get_telemetry_client()
    if client:
        client._clean_up()
        # set to None to avoid side effect in other tests
        _set_telemetry_client(None)


@pytest.fixture
def mock_requests():
    """Fixture to mock requests.post and capture telemetry records."""
    captured_records = []

    url_status_code_map = {
        "http://127.0.0.1:9999/nonexistent": 404,
        "http://127.0.0.1:9999/unauthorized": 401,
        "http://127.0.0.1:9999/forbidden": 403,
        "http://127.0.0.1:9999/bad_request": 400,
    }

    def mock_post(url, json=None, **kwargs):
        if url in url_status_code_map:
            mock_response = Mock()
            mock_response.status_code = url_status_code_map[url]
            return mock_response
        if url == "http://localhost:9999":
            if json and "records" in json:
                captured_records.extend(json["records"])
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "success",
                "count": len(json.get("records", [])) if json else 0,
            }
            return mock_response
        return Mock(status_code=404)

    with patch("requests.post", side_effect=mock_post):
        yield captured_records


@pytest.fixture(autouse=True)
def mock_requests_get(request):
    if request.node.get_closest_marker("no_mock_requests_get"):
        yield
        return

    with patch("mlflow.telemetry.client.requests.get") as mock_get:
        mock_get.return_value = Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_events": [],
                    "disable_sdks": [],
                }
            ),
        )
        yield


@pytest.fixture
def mock_telemetry_client(mock_requests_get, mock_requests):
    client = TelemetryClient()
    # ensure config is fetched before the test
    client._config_thread.join(timeout=1)
    yield client
    # TODO: add a context manager to always clean up the threads for tests
    client._clean_up()


@pytest.fixture(autouse=True)
def is_mlflow_testing(monkeypatch):
    # enable telemetry by default when running tests in local with dev version
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_TESTING_TELEMETRY", True)


@pytest.fixture
def bypass_env_check(monkeypatch):
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_TESTING_TELEMETRY", False)
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_IN_CI_ENV_OR_TESTING", False)
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_DEV_VERSION", False)
