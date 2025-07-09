from concurrent.futures import ThreadPoolExecutor

from mlflow.telemetry.utils import (
    _avoid_telemetry_tracking,
    is_invoked_from_internal_api,
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


def test_avoid_telemetry_tracking():
    with _avoid_telemetry_tracking():
        assert is_invoked_from_internal_api() is True

    assert is_invoked_from_internal_api() is False


def test_avoid_telemetry_tracking_multiple_threads():
    def thread_target(x):
        if x % 2 == 0:
            with _avoid_telemetry_tracking():
                return is_invoked_from_internal_api()
        else:
            return is_invoked_from_internal_api()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(thread_target, i) for i in range(10)]
        results = [f.result() for f in futures]
        assert results == [True, False] * 5
