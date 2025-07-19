from typing import Any

from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.schemas import BaseParams


def validate_telemetry_record(
    mock_requests, func, params=None, *, status="success", search_index=False, check_params=True
) -> dict[str, Any]:
    """
    Validate the telemetry record at the given index.
    """
    get_telemetry_client().flush()

    if search_index:
        api_names = [record["data"]["api_name"] for record in mock_requests]
        idx = api_names.index(func.__qualname__)
    else:
        idx = 0

    record = mock_requests[idx]
    data = record["data"]
    assert data["api_module"] == func.__module__
    assert data["api_name"] == func.__qualname__
    if check_params:
        if isinstance(params, BaseParams):
            assert data["params"] == params.to_json()
        else:
            assert data["params"] == params
    assert data["status"] == status
    assert data["duration_ms"] is not None
    mock_requests.clear()
    return data
