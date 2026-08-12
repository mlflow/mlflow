import os
import select
import signal
import threading
from unittest import mock

import pytest

from mlflow.utils import server_info
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.server_info import (
    SERVER_INFO_ENDPOINT,
    ServerInfoRequestError,
    ServerInfoResponse,
    _clear_server_info_cache,
    fetch_server_info,
)


@pytest.fixture(autouse=True)
def clear_server_info_cache():
    _clear_server_info_cache()
    yield
    _clear_server_info_cache()


def _mock_response(status_code=200, json_body=None, text=""):
    response = mock.Mock(status_code=status_code, text=text)
    if json_body is not None:
        response.json.return_value = json_body
    return response


def test_fetch_server_info_uses_expected_request_arguments():
    creds = MlflowHostCreds("https://example.com")

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        return_value=_mock_response(status_code=200, json_body={"store_type": "SqlStore"}),
    ) as mock_http:
        response = fetch_server_info(creds)

    assert response.status_code == 200
    assert response.data == {"store_type": "SqlStore"}
    _, kwargs = mock_http.call_args
    assert kwargs["host_creds"] is creds
    assert kwargs["endpoint"] == SERVER_INFO_ENDPOINT
    assert kwargs["method"] == "GET"
    assert kwargs["timeout"] == 3
    assert kwargs["max_retries"] == 0
    assert kwargs["raise_on_status"] is False


def test_fetch_server_info_caches_successful_response():
    creds = MlflowHostCreds("https://example.com")

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        return_value=_mock_response(status_code=200, json_body={"store_type": "SqlStore"}),
    ) as mock_http:
        assert fetch_server_info(creds).data == {"store_type": "SqlStore"}
        assert fetch_server_info(creds).data == {"store_type": "SqlStore"}

    mock_http.assert_called_once()


def test_fetch_server_info_refreshes_expired_success():
    creds = MlflowHostCreds("https://example.com")
    now = 0.0
    responses = [
        _mock_response(status_code=200, json_body={"store_type": "FileStore"}),
        _mock_response(status_code=200, json_body={"store_type": "SqlStore"}),
    ]

    with (
        mock.patch("mlflow.utils.server_info.time.monotonic", side_effect=lambda: now),
        mock.patch("mlflow.utils.server_info.http_request", side_effect=responses) as mock_http,
    ):
        assert fetch_server_info(creds).data == {"store_type": "FileStore"}

        now = server_info._SERVER_INFO_CACHE_TTL_SECONDS - 1
        assert fetch_server_info(creds).data == {"store_type": "FileStore"}

        now = server_info._SERVER_INFO_CACHE_TTL_SECONDS
        assert fetch_server_info(creds).data == {"store_type": "SqlStore"}

    assert mock_http.call_count == 2


def test_fetch_server_info_caches_empty_json_response():
    creds = MlflowHostCreds("https://example.com")

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        return_value=_mock_response(status_code=200, json_body={}),
    ) as mock_http:
        assert fetch_server_info(creds).data == {}
        assert fetch_server_info(creds).data == {}

    mock_http.assert_called_once()


def test_fetch_server_info_normalizes_trailing_slash():
    with mock.patch(
        "mlflow.utils.server_info.http_request",
        return_value=_mock_response(status_code=200, json_body={"store_type": "SqlStore"}),
    ) as mock_http:
        assert fetch_server_info(MlflowHostCreds("https://example.com")).data == {
            "store_type": "SqlStore"
        }
        assert fetch_server_info(MlflowHostCreds("https://example.com/")).data == {
            "store_type": "SqlStore"
        }

    mock_http.assert_called_once()


@pytest.mark.parametrize(
    ("host_1", "host_2"),
    [
        ("https://example.com", "http://example.com"),
        ("https://example.com", "https://example.com:5000"),
        ("https://example.com", "https://example.com/mlflow"),
        ("https://tracking.example.com/mlflow", "https://artifacts.example.com/mlflow"),
    ],
)
def test_fetch_server_info_keeps_distinct_hosts_separate(host_1, host_2):
    with mock.patch(
        "mlflow.utils.server_info.http_request",
        return_value=_mock_response(status_code=200, json_body={"store_type": "SqlStore"}),
    ) as mock_http:
        fetch_server_info(MlflowHostCreds(host_1))
        fetch_server_info(MlflowHostCreds(host_2))

    assert mock_http.call_count == 2


def test_fetch_server_info_evicts_oldest_success_after_cache_limit():
    calls = 0

    def mock_http_request(host_creds, **_kwargs):
        nonlocal calls
        calls += 1
        return _mock_response(status_code=200, json_body={"host": host_creds.host})

    with mock.patch("mlflow.utils.server_info.http_request", side_effect=mock_http_request):
        first = "https://example-0.com"
        for idx in range(16):
            fetch_server_info(MlflowHostCreds(f"https://example-{idx}.com"))

        fetch_server_info(MlflowHostCreds("https://example-16.com"))
        fetch_server_info(MlflowHostCreds(first))

    assert calls == 18


@pytest.mark.parametrize(
    ("response", "expected_exception"),
    [
        (_mock_response(status_code=404, text="not found"), None),
        (_mock_response(status_code=500, text="server error"), None),
        (
            mock.Mock(status_code=200, json=mock.Mock(side_effect=ValueError("bad json"))),
            ServerInfoRequestError,
        ),
        (ConnectionError("connection refused"), ServerInfoRequestError),
    ],
)
def test_fetch_server_info_does_not_cache_failures(response, expected_exception):
    creds = MlflowHostCreds("https://example.com")
    http_side_effect = response if isinstance(response, Exception) else None
    http_return_value = None if http_side_effect else response

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=http_side_effect,
        return_value=http_return_value,
    ) as mock_http:
        for _ in range(2):
            if expected_exception is None:
                fetch_server_info(creds)
            else:
                with pytest.raises(expected_exception):
                    fetch_server_info(creds)

    assert mock_http.call_count == 2


@pytest.mark.parametrize("json_body", [None, [], "not an object", 1])
def test_fetch_server_info_rejects_and_does_not_cache_non_object_json(json_body):
    creds = MlflowHostCreds("https://example.com")
    response = mock.Mock(status_code=200)
    response.json.return_value = json_body

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        return_value=response,
    ) as mock_http:
        for _ in range(2):
            with pytest.raises(ServerInfoRequestError, match="Expected a JSON object"):
                fetch_server_info(creds)

    assert mock_http.call_count == 2


def test_fetch_server_info_single_flights_same_key_success():
    creds = MlflowHostCreds("https://example.com")
    call_started = threading.Event()
    release_request = threading.Event()
    call_count = 0
    count_lock = threading.Lock()
    results = []

    def mock_http_request(**_kwargs):
        nonlocal call_count
        with count_lock:
            call_count += 1
        call_started.set()
        assert release_request.wait(timeout=5)
        return _mock_response(status_code=200, json_body={"store_type": "SqlStore"})

    def worker():
        results.append(fetch_server_info(creds).data)

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=mock_http_request,
    ):
        threads = [
            threading.Thread(target=worker, name=f"server-info-success-{idx}") for idx in range(2)
        ]
        for thread in threads:
            thread.start()
        assert call_started.wait(timeout=5)
        release_request.set()
        for thread in threads:
            thread.join(timeout=5)

    assert call_count == 1
    assert results == [{"store_type": "SqlStore"}] * 2


def test_fetch_server_info_single_flights_expired_success():
    creds = MlflowHostCreds("https://example.com")
    now = 0.0

    with (
        mock.patch("mlflow.utils.server_info.time.monotonic", side_effect=lambda: now),
        mock.patch(
            "mlflow.utils.server_info.http_request",
            return_value=_mock_response(status_code=200, json_body={"version": 1}),
        ),
    ):
        fetch_server_info(creds)

    now = server_info._SERVER_INFO_CACHE_TTL_SECONDS
    call_started = threading.Event()
    release_request = threading.Event()
    call_count = 0
    results = []

    def mock_http_request(**_kwargs):
        nonlocal call_count
        call_count += 1
        call_started.set()
        assert release_request.wait(timeout=5)
        return _mock_response(status_code=200, json_body={"version": 2})

    def worker():
        results.append(fetch_server_info(creds).data)

    with (
        mock.patch("mlflow.utils.server_info.time.monotonic", side_effect=lambda: now),
        mock.patch("mlflow.utils.server_info.http_request", side_effect=mock_http_request),
    ):
        threads = [
            threading.Thread(target=worker, name=f"server-info-expired-{idx}") for idx in range(2)
        ]
        for thread in threads:
            thread.start()
        assert call_started.wait(timeout=5)
        release_request.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    assert call_count == 1
    assert results == [{"version": 2}] * 2


def test_fetch_server_info_single_flights_same_key_non_200_and_retries_later():
    creds = MlflowHostCreds("https://example.com")
    release_request = threading.Event()
    call_started = threading.Event()
    call_count = 0
    count_lock = threading.Lock()

    def mock_http_request(**_kwargs):
        nonlocal call_count
        with count_lock:
            call_count += 1
        call_started.set()
        assert release_request.wait(timeout=5)
        return _mock_response(status_code=500, text="server error")

    responses = []

    def worker():
        responses.append(fetch_server_info(creds).status_code)

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=mock_http_request,
    ):
        leader = threading.Thread(target=worker, name="server-info-status-leader")
        leader.start()
        assert call_started.wait(timeout=5)

        in_flight = server_info._SERVER_INFO_IN_FLIGHT["https://example.com"]
        follower_waiting = threading.Event()
        original_wait = in_flight.event.wait

        def wait_for_leader(*args, **kwargs):
            follower_waiting.set()
            return original_wait(*args, **kwargs)

        with mock.patch.object(in_flight.event, "wait", side_effect=wait_for_leader):
            follower = threading.Thread(target=worker, name="server-info-status-follower")
            follower.start()
            assert follower_waiting.wait(timeout=5)
            release_request.set()

        threads = [leader, follower]
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

        assert fetch_server_info(creds).status_code == 500

    assert call_count == 2
    assert responses == [500, 500]


def test_fetch_server_info_single_flights_same_key_exceptions():
    creds = MlflowHostCreds("https://example.com")
    release_request = threading.Event()
    call_started = threading.Event()
    call_count = 0
    count_lock = threading.Lock()
    errors = []

    def mock_http_request(**_kwargs):
        nonlocal call_count
        with count_lock:
            call_count += 1
        call_started.set()
        assert release_request.wait(timeout=5)
        raise ConnectionError("connection refused")

    def worker():
        try:
            fetch_server_info(creds)
        except Exception as exc:  # pragma: no branch - each worker must error
            errors.append(type(exc))

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=mock_http_request,
    ):
        leader = threading.Thread(target=worker, name="server-info-error-leader")
        leader.start()
        assert call_started.wait(timeout=5)

        in_flight = server_info._SERVER_INFO_IN_FLIGHT["https://example.com"]
        follower_waiting = threading.Event()
        original_wait = in_flight.event.wait

        def wait_for_leader(*args, **kwargs):
            follower_waiting.set()
            return original_wait(*args, **kwargs)

        with mock.patch.object(in_flight.event, "wait", side_effect=wait_for_leader):
            follower = threading.Thread(target=worker, name="server-info-error-follower")
            follower.start()
            assert follower_waiting.wait(timeout=5)
            release_request.set()

        threads = [leader, follower]
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    assert call_count == 1
    assert errors == [ServerInfoRequestError, ServerInfoRequestError]


@pytest.mark.parametrize("failure_stage", ["fetch", "cache"])
def test_fetch_server_info_signals_followers_when_leader_raises_base_exception(failure_stage):
    class LeaderExit(BaseException):
        pass

    creds = MlflowHostCreds("https://example.com")
    release_request = threading.Event()
    call_started = threading.Event()
    errors = []

    def mock_fetch_uncached(_host_creds):
        call_started.set()
        assert release_request.wait(timeout=5)
        if failure_stage == "fetch":
            raise LeaderExit
        return ServerInfoResponse(status_code=200, data={})

    def mock_cache_success(*_args):
        raise LeaderExit

    def worker():
        try:
            fetch_server_info(creds)
        except BaseException as exc:
            errors.append(type(exc))

    with (
        mock.patch(
            "mlflow.utils.server_info._fetch_server_info_uncached",
            side_effect=mock_fetch_uncached,
        ) as mock_fetch,
        mock.patch(
            "mlflow.utils.server_info._cache_success_locked",
            side_effect=mock_cache_success,
        ) as mock_cache,
    ):
        leader = threading.Thread(target=worker, name="server-info-base-exception-leader")
        leader.start()
        assert call_started.wait(timeout=5)

        in_flight = server_info._SERVER_INFO_IN_FLIGHT["https://example.com"]
        follower_waiting = threading.Event()
        original_wait = in_flight.event.wait

        def wait_for_leader(*args, **kwargs):
            follower_waiting.set()
            return original_wait(*args, **kwargs)

        with mock.patch.object(in_flight.event, "wait", side_effect=wait_for_leader):
            follower = threading.Thread(target=worker, name="server-info-base-exception-follower")
            follower.start()
            assert follower_waiting.wait(timeout=5)
            release_request.set()

        for thread in (leader, follower):
            thread.join(timeout=5)
            assert not thread.is_alive()

    mock_fetch.assert_called_once()
    assert mock_cache.call_count == (failure_stage == "cache")
    assert errors == [LeaderExit, LeaderExit]
    assert server_info._SERVER_INFO_IN_FLIGHT == {}


def test_fetch_server_info_allows_different_keys_to_fetch_concurrently():
    barrier = threading.Barrier(2)

    def mock_http_request(**_kwargs):
        barrier.wait(timeout=5)
        return _mock_response(status_code=200, json_body={"store_type": "SqlStore"})

    errors = []

    def worker(host):
        try:
            fetch_server_info(MlflowHostCreds(host))
        except Exception as exc:  # pragma: no cover - failure would fail assertions below
            errors.append(exc)

    with mock.patch("mlflow.utils.server_info.http_request", side_effect=mock_http_request):
        threads = [
            threading.Thread(
                target=worker,
                args=("https://example-1.com",),
                name="server-info-concurrent-1",
            ),
            threading.Thread(
                target=worker,
                args=("https://example-2.com",),
                name="server-info-concurrent-2",
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

    assert errors == []


def test_clear_server_info_cache_forces_refetch():
    creds = MlflowHostCreds("https://example.com")

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        return_value=_mock_response(status_code=200, json_body={"store_type": "SqlStore"}),
    ) as mock_http:
        fetch_server_info(creds)
        _clear_server_info_cache()
        fetch_server_info(creds)

    assert mock_http.call_count == 2


def test_reset_server_info_state_after_fork_replaces_inherited_state():
    creds = MlflowHostCreds("https://example.com")
    with mock.patch(
        "mlflow.utils.server_info.http_request",
        return_value=_mock_response(status_code=200, json_body={"store_type": "SqlStore"}),
    ):
        fetch_server_info(creds)

    key = "https://in-flight.example.com"
    server_info._SERVER_INFO_IN_FLIGHT[key] = server_info._InFlightRequest()
    old_cache = server_info._SERVER_INFO_CACHE
    old_lock = server_info._SERVER_INFO_CACHE_LOCK
    old_in_flight = server_info._SERVER_INFO_IN_FLIGHT

    old_lock.acquire()
    try:
        server_info._reset_server_info_state_after_fork()
    finally:
        old_lock.release()

    assert server_info._SERVER_INFO_CACHE is not old_cache
    assert server_info._SERVER_INFO_CACHE == {}
    assert server_info._SERVER_INFO_CACHE_LOCK is not old_lock
    assert server_info._SERVER_INFO_CACHE_LOCK.acquire(blocking=False)
    server_info._SERVER_INFO_CACHE_LOCK.release()
    assert server_info._SERVER_INFO_IN_FLIGHT is not old_in_flight
    assert server_info._SERVER_INFO_IN_FLIGHT == {}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_registered_at_fork_handler_prevents_inherited_lock_deadlock():
    read_fd, write_fd = os.pipe()
    old_lock = server_info._SERVER_INFO_CACHE_LOCK
    child_reaped = False
    old_lock.acquire()
    try:
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                _clear_server_info_cache()
                os.write(write_fd, b"ok")
                os._exit(0)
            except BaseException:
                os._exit(1)

        os.close(write_fd)
        ready, _, _ = select.select([read_fd], [], [], 5)
        assert ready, "child blocked on an inherited server-info cache lock"
        assert os.read(read_fd, 2) == b"ok"
        _, status = os.waitpid(pid, 0)
        child_reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        old_lock.release()
        os.close(read_fd)
        if "pid" in locals() and pid != 0 and not child_reaped:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
