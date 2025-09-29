import json
from typing import Any


def validate_telemetry_record(
    mock_telemetry_client,
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
    mock_telemetry_client.flush()

    if search_index:
        event_names = [record["data"]["event_name"] for record in mock_requests]
        idx = event_names.index(event_name)
    else:
        idx = 0

    record = mock_requests[idx]
    data = record["data"]
    assert data["event_name"] == event_name
    if check_params:
        if params:
            assert data["params"] == json.dumps(params)
        else:
            assert data["params"] is None
    assert data["status"] == status
    assert data["duration_ms"] is not None
    mock_requests.clear()
    return data
