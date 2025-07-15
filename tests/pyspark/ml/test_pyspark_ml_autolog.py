import mlflow
from mlflow.telemetry.schemas import AutologParams

from tests.helper_functions import validate_telemetry_record


def test_autolog_sends_telemetry_record(mock_requests):
    mlflow.pyspark.ml.autolog(log_models=True, disable=False)

    validate_telemetry_record(
        mock_requests,
        mlflow.pyspark.ml.autolog,
        params=AutologParams(
            flavor="pyspark.ml",
            disable=False,
            log_traces=False,
            log_models=True,
        ),
    )
