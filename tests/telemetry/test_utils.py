from concurrent.futures import ThreadPoolExecutor

from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.track import track_api_usage
from mlflow.telemetry.utils import (
    _avoid_telemetry_tracking,
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


def test_avoid_telemetry_tracking(mock_requests):
    @track_api_usage
    def test_func():
        pass

    with _avoid_telemetry_tracking():
        test_func()
        get_telemetry_client().flush()
    assert len(mock_requests) == 0

    test_func()
    get_telemetry_client().flush()
    assert len(mock_requests) == 1


def test_avoid_telemetry_tracking_multiple_threads(mock_requests):
    @track_api_usage
    def test_func():
        pass

    def thread_target(x):
        if x % 2 == 0:
            with _avoid_telemetry_tracking():
                test_func()
        else:
            test_func()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(thread_target, i) for i in range(10)]
        [f.result() for f in futures]

    get_telemetry_client().flush()
    assert len(mock_requests) == 5
