from unittest import mock

from mlflow.telemetry.constant import BASE_URL
from mlflow.telemetry.schemas import SourceSDK, TelemetryConfig
from mlflow.telemetry.utils import (
    _get_config,
    _get_config_url,
    is_telemetry_disabled,
)
from mlflow.version import VERSION


def test_is_telemetry_disabled(monkeypatch):
    assert is_telemetry_disabled() is False

    with monkeypatch.context() as m:
        m.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
        assert is_telemetry_disabled() is True

    assert is_telemetry_disabled() is False

    with monkeypatch.context() as m:
        m.setenv("DO_NOT_TRACK", "true")
        assert is_telemetry_disabled() is True


def test_get_config_url():
    assert _get_config_url("1.0.0") == f"{BASE_URL}/config/1.0.0"
    assert _get_config_url("1.0.0.rc0") == f"{BASE_URL}/config/1.0.0.rc0"
    assert _get_config_url("1.0.0.dev0") == f"{BASE_URL}/dev/config/1.0.0.dev0"


def test_get_config():
    with mock.patch("requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(status_code=403)
        assert _get_config() is None

    with mock.patch("requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": True,
                }
            ),
        )
        assert _get_config() is None

    with mock.patch("requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": "1.0.0",
                    "disable_telemetry": False,
                    "telemetry_url": "http://localhost:9999",
                }
            ),
        )
        assert _get_config() is None

    with mock.patch("requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "telemetry_url": "http://localhost:9999",
                    "rollout_percentage": 0,
                }
            ),
        )
        assert _get_config() is None

    with mock.patch("requests.get") as mock_requests, mock.patch("random.randint", return_value=80):
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "telemetry_url": "http://localhost:9999",
                    "rollout_percentage": 70,
                }
            ),
        )
        assert _get_config() is None

    with mock.patch("requests.get") as mock_requests, mock.patch("random.randint", return_value=50):
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "telemetry_url": "http://localhost:9999",
                    "rollout_percentage": 70,
                }
            ),
        )
        assert _get_config() == TelemetryConfig(
            telemetry_url="http://localhost:9999",
            disable_api_map={},
        )

    with mock.patch("requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "telemetry_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                }
            ),
        )
        assert _get_config() == TelemetryConfig(
            telemetry_url="http://localhost:9999",
            disable_api_map={},
        )

    with mock.patch("requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "telemetry_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_api_map": {},
                    "disable_sdks": ["mlflow-tracing"],
                }
            ),
        )
        with mock.patch(
            "mlflow.telemetry.utils.get_source_sdk", return_value=SourceSDK.MLFLOW_TRACING
        ):
            assert _get_config() is None

        with mock.patch(
            "mlflow.telemetry.utils.get_source_sdk", return_value=SourceSDK.MLFLOW_SKINNY
        ):
            assert _get_config() == TelemetryConfig(
                telemetry_url="http://localhost:9999",
                disable_api_map={},
            )

        with mock.patch("mlflow.telemetry.utils.get_source_sdk", return_value=SourceSDK.MLFLOW):
            assert _get_config() == TelemetryConfig(
                telemetry_url="http://localhost:9999",
                disable_api_map={},
            )
