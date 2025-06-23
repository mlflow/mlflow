from mlflow.telemetry.utils import (
    is_telemetry_disabled,
    temporarily_disable_telemetry,
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


def test_temporarily_disable_telemetry(monkeypatch):
    with temporarily_disable_telemetry():
        assert is_telemetry_disabled() is True

    assert is_telemetry_disabled() is False
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    assert is_telemetry_disabled() is True

    with temporarily_disable_telemetry():
        assert is_telemetry_disabled() is True

    assert is_telemetry_disabled() is True
