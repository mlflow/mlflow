from mlflow.telemetry.constant import BASE_URL
from mlflow.telemetry.utils import (
    _get_config_url,
    is_telemetry_disabled,
)


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
    assert _get_config_url("1.0.0+abc") is None
