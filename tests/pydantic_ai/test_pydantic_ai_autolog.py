import json
from dataclasses import asdict

import mlflow
from mlflow.telemetry.schemas import AutologParams

from tests.helper_functions import wait_for_telemetry_threads


def test_autolog_sends_telemetry_record(mock_requests):
    mlflow.pydantic_ai.autolog(log_traces=True, disable=False)

    # Wait for telemetry to be sent
    wait_for_telemetry_threads()

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
