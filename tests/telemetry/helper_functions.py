from typing import Any

from mlflow.telemetry.client import TelemetryClient, get_telemetry_client
from mlflow.telemetry.schemas import BaseParams


def validate_telemetry_record(
    mock_requests,
    event_name: str,
    params=None,
    *,
    status="success",
    search_index=True,
    check_params=True,
) -> dict[str, Any]:
    """
    Validate the telemetry record at the given index.
    """
    get_telemetry_client().flush()

    if search_index:
        event_names = [record["data"]["event_name"] for record in mock_requests]
        idx = event_names.index(event_name)
    else:
        idx = 0

    record = mock_requests[idx]
    data = record["data"]
    assert data["event_name"] == event_name
    if check_params:
        if isinstance(params, BaseParams):
            assert data["params"] == params.to_json()
        else:
            assert data["params"] == params
    assert data["status"] == status
    assert data["duration_ms"] is not None
    mock_requests.clear()
    return data


def clean_up_threads(client: TelemetryClient):
    """
    Clean up the telemetry threads.
    flush(terminate=True) has a timeout, use this function if the
    threads take longer than the timeout to finish.
    """
    client._config_thread.join()
    for thread in client._consumer_threads:
        if thread.is_alive():
            thread.join()
