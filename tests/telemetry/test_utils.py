from unittest.mock import Mock, patch

from mlflow.telemetry.constant import (
    CONFIG_STAGING_URL,
    CONFIG_URL,
    FALLBACK_UI_CONFIG,
    UI_CONFIG_STAGING_URL,
    UI_CONFIG_URL,
)
from mlflow.telemetry.utils import (
    _get_config_url,
    fetch_ui_telemetry_config,
    is_telemetry_disabled,
)


def test_is_telemetry_disabled(monkeypatch, bypass_env_check):
    assert is_telemetry_disabled() is False

    with monkeypatch.context() as m:
        m.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
        assert is_telemetry_disabled() is True

    assert is_telemetry_disabled() is False

    with monkeypatch.context() as m:
        m.setenv("DO_NOT_TRACK", "true")
        assert is_telemetry_disabled() is True


def test_get_config_url(bypass_env_check):
    assert _get_config_url("1.0.0") == f"{CONFIG_URL}/1.0.0.json"
    assert _get_config_url("1.0.0.rc0") == f"{CONFIG_URL}/1.0.0.rc0.json"
    assert _get_config_url("1.0.0.dev0") == f"{CONFIG_STAGING_URL}/1.0.0.dev0.json"
    assert _get_config_url("1.0.0+abc") is None
    assert _get_config_url("1.0.0", is_ui=True) == f"{UI_CONFIG_URL}/1.0.0.json"
    assert _get_config_url("1.0.0.rc0", is_ui=True) == f"{UI_CONFIG_URL}/1.0.0.rc0.json"
    assert _get_config_url("1.0.0.dev0", is_ui=True) == f"{UI_CONFIG_STAGING_URL}/1.0.0.dev0.json"
    assert _get_config_url("1.0.0+abc", is_ui=True) is None


def test_fetch_ui_telemetry_config_fetch_success(bypass_env_check):
    mock_config = {
        "mlflow_version": "3.7.1.dev0",
        "disable_telemetry": False,
        "rollout_percentage": 100,
        "ingestion_url": "https://api.mlflow-telemetry.io/staging/log",
        "disable_sdks": [],
        "disable_os": [],
        "disable_events": [],
        "disable_ui_telemetry": False,
        "disable_ui_events": ["test_event"],
        "ui_rollout_percentage": 100,
    }

    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_config
        mock_get.return_value = mock_response

        result = fetch_ui_telemetry_config()

        assert result["disable_ui_telemetry"] is False
        assert result["disable_ui_events"] == ["test_event"]
        assert result["ui_rollout_percentage"] == 100


def test_fetch_ui_telemetry_config_fetch_error_fallback(bypass_env_check):
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetch_ui_telemetry_config()

        assert result == FALLBACK_UI_CONFIG
        assert result["disable_ui_telemetry"] is True

    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Network error")

        result = fetch_ui_telemetry_config()

        assert result == FALLBACK_UI_CONFIG
        assert result["disable_ui_telemetry"] is True
        assert result["disable_ui_events"] == []
        assert result["ui_rollout_percentage"] == 0
