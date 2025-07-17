from concurrent.futures import ThreadPoolExecutor

from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.track import track_api_usage
from mlflow.telemetry.utils import (
    _disable_telemetry,
    is_telemetry_disabled,
)

from tests.helper_functions import validate_telemetry_record


def test_is_telemetry_disabled(monkeypatch):
    assert is_telemetry_disabled() is False

    with monkeypatch.context() as m:
        m.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
        assert is_telemetry_disabled() is True

    assert is_telemetry_disabled() is False

    with monkeypatch.context() as m:
        m.setenv("DO_NOT_TRACK", "true")
        assert is_telemetry_disabled() is True


def test_disable_telemetry(mock_requests):
    @track_api_usage
    def test_func():
        pass

    with _disable_telemetry():
        test_func()
        get_telemetry_client().flush()
    assert len(mock_requests) == 0

    test_func()
    get_telemetry_client().flush()
    assert len(mock_requests) == 1
    validate_telemetry_record(mock_requests, test_func)


def test_disable_telemetry_multiple_threads(mock_requests):
    @track_api_usage
    def test_func():
        pass

    def thread_target(x):
        if x % 2 == 0:
            with _disable_telemetry():
                test_func()
        else:
            test_func()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(thread_target, i) for i in range(10)]
        for f in futures:
            f.result()

    get_telemetry_client().flush()
    assert len(mock_requests) == 5
    for i in range(len(mock_requests)):
        assert mock_requests[i]["data"]["api_name"] == test_func.__qualname__
