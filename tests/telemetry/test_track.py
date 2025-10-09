import time
from unittest import mock

import pytest

from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY
from mlflow.telemetry.client import (
    TelemetryClient,
    get_telemetry_client,
    set_telemetry_client,
)
from mlflow.telemetry.events import CreateLoggedModelEvent, Event
from mlflow.telemetry.schemas import Status
from mlflow.telemetry.track import _is_telemetry_disabled_for_event, record_usage_event
from mlflow.telemetry.utils import is_telemetry_disabled
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.version import VERSION


class TestEvent(Event):
    name = "test_event"


def test_record_usage_event(mock_requests, mock_telemetry_client: TelemetryClient):
    @record_usage_event(TestEvent)
    def succeed_func():
        # sleep to make sure duration_ms > 0
        time.sleep(0.01)
        return True

    @record_usage_event(TestEvent)
    def fail_func():
        time.sleep(0.01)
        raise ValueError("test")

    with mock.patch(
        "mlflow.telemetry.track.get_telemetry_client", return_value=mock_telemetry_client
    ):
        succeed_func()
        with pytest.raises(ValueError, match="test"):
            fail_func()

    mock_telemetry_client.flush()

    records = [
        record["data"] for record in mock_requests if record["data"]["event_name"] == TestEvent.name
    ]
    assert len(records) == 2
    succeed_record = records[0]
    assert succeed_record["schema_version"] == 1
    assert succeed_record["event_name"] == TestEvent.name
    assert succeed_record["status"] == Status.SUCCESS.value
    assert succeed_record["params"] is None
    assert succeed_record["duration_ms"] > 0

    fail_record = records[1]
    assert fail_record["schema_version"] == 1
    assert fail_record["event_name"] == TestEvent.name
    assert fail_record["status"] == Status.FAILURE.value
    assert fail_record["params"] is None
    assert fail_record["duration_ms"] > 0

    telemetry_info = mock_telemetry_client.info
    assert telemetry_info.items() <= succeed_record.items()
    assert telemetry_info.items() <= fail_record.items()


def test_backend_store_info(tmp_path, mock_telemetry_client: TelemetryClient):
    sqlite_uri = f"sqlite:///{tmp_path.joinpath('test.db')}"
    with _use_tracking_uri(sqlite_uri):
        mock_telemetry_client._update_backend_store()
    assert mock_telemetry_client.info["tracking_uri_scheme"] == "sqlite"

    with _use_tracking_uri(tmp_path):
        mock_telemetry_client._update_backend_store()
    assert mock_telemetry_client.info["tracking_uri_scheme"] == "file"


@pytest.mark.parametrize(
    ("env_var", "value", "expected_result"),
    [
        (MLFLOW_DISABLE_TELEMETRY.name, "true", None),
        (MLFLOW_DISABLE_TELEMETRY.name, "false", TelemetryClient),
        ("DO_NOT_TRACK", "true", None),
        ("DO_NOT_TRACK", "false", TelemetryClient),
    ],
)
def test_record_usage_event_respect_env_var(
    monkeypatch, env_var, value, expected_result, bypass_env_check
):
    monkeypatch.setenv(env_var, value)
    # mimic the behavior of `import mlflow`
    set_telemetry_client()
    telemetry_client = get_telemetry_client()
    if expected_result is None:
        assert is_telemetry_disabled() is True
        assert telemetry_client is None
    else:
        assert isinstance(telemetry_client, expected_result)
        telemetry_client._clean_up()


def test_record_usage_event_update_env_var_after_import(
    monkeypatch, mock_requests, mock_telemetry_client
):
    assert isinstance(mock_telemetry_client, TelemetryClient)

    @record_usage_event(TestEvent)
    def test_func():
        pass

    with mock.patch(
        "mlflow.telemetry.track.get_telemetry_client", return_value=mock_telemetry_client
    ):
        test_func()

        mock_telemetry_client.flush()
        events = {record["data"]["event_name"] for record in mock_requests}
        assert TestEvent.name in events
        mock_requests.clear()

        monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
        test_func()
        # no new record should be added
        assert len(mock_requests) == 0


@pytest.mark.no_mock_requests_get
def test_is_telemetry_disabled_for_event():
    def mock_requests_get(*args, **kwargs):
        time.sleep(1)
        return mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_events": ["test_event"],
                }
            ),
        )

    with mock.patch("mlflow.telemetry.client.requests.get", side_effect=mock_requests_get):
        client = TelemetryClient()
        assert client is not None
        assert client.config is None
        with mock.patch("mlflow.telemetry.track.get_telemetry_client", return_value=client):
            # do not skip when config is not fetched yet
            assert _is_telemetry_disabled_for_event(TestEvent) is False
            assert _is_telemetry_disabled_for_event(TestEvent) is False
            time.sleep(2)
            assert client._is_config_fetched is True
            assert client.config is not None
            # event not in disable_events, do not skip
            assert _is_telemetry_disabled_for_event(CreateLoggedModelEvent) is False
            # event in disable_events, skip
            assert _is_telemetry_disabled_for_event(TestEvent) is True
        # clean up
        client._clean_up()

    # test telemetry disabled after config is fetched
    def mock_requests_get(*args, **kwargs):
        time.sleep(1)
        return mock.Mock(status_code=403)

    with mock.patch("mlflow.telemetry.client.requests.get", side_effect=mock_requests_get):
        client = TelemetryClient()
        assert client is not None
        assert client.config is None
        with (
            mock.patch("mlflow.telemetry.track.get_telemetry_client", return_value=client),
            mock.patch(
                "mlflow.telemetry.client._set_telemetry_client"
            ) as mock_set_telemetry_client,
        ):
            # do not skip when config is not fetched yet
            assert _is_telemetry_disabled_for_event(CreateLoggedModelEvent) is False
            assert _is_telemetry_disabled_for_event(TestEvent) is False
            time.sleep(2)
            assert client._is_config_fetched is True
            assert client.config is None
            # global telemetry client is set to None when telemetry is disabled
            mock_set_telemetry_client.assert_called_once_with(None)
        # clean up
        client._clean_up()
