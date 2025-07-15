import mlflow
from mlflow.telemetry.schemas import AutologParams

from tests.helper_functions import validate_telemetry_record


def test_autolog_sends_telemetry_record(mock_requests):
    mlflow.pydantic_ai.autolog(log_traces=True, disable=False)

    validate_telemetry_record(
        mock_requests,
        mlflow.pydantic_ai.autolog,
        params=AutologParams(
            flavor="pydantic_ai",
            disable=False,
            log_traces=True,
            log_models=False,
        ),
    )
