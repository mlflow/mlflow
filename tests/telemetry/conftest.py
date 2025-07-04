import threading
from unittest.mock import Mock, patch

import pytest

from mlflow.telemetry.client import TELEMETRY_URL


@pytest.fixture(autouse=True, scope="module")
def cleanup_zombie_threads():
    for thread in threading.enumerate():
        if thread != threading.main_thread():
            thread.join(timeout=0.1)


@pytest.fixture
def mock_requests():
    """Fixture to mock requests.post and capture telemetry records."""
    captured_records = []

    def mock_post(url, json=None, **kwargs):
        if url != TELEMETRY_URL:
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
