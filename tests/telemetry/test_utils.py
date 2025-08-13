from mlflow.telemetry.constant import CONFIG_STAGING_URL, CONFIG_URL
from mlflow.telemetry.utils import (
    _get_config_url,
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
