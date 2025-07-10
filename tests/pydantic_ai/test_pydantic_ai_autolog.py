import json
from dataclasses import asdict

import mlflow
from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.schemas import AutologParams


def test_autolog_sends_telemetry_record(mock_requests):
    mlflow.pydantic_ai.autolog(log_traces=True, disable=False)

    # Wait for telemetry to be sent
    get_telemetry_client().flush()

    # Check that telemetry record was sent
    assert len(mock_requests) == 1
    autolog_record = mock_requests[0]
    data = json.loads(autolog_record["data"])
    assert data["api_module"] == mlflow.pydantic_ai.autolog.__module__
    assert data["api_name"] == "autolog"
    assert data["params"] == asdict(
        AutologParams(
            flavor="pydantic_ai",
            disable=False,
            log_traces=True,
            log_models=False,
        )
    )
    assert data["status"] == "success"
