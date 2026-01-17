import os
from datetime import timedelta

import pytest
from testcontainers.compose import DockerCompose
from testcontainers.core.wait_strategies import HttpWaitStrategy

import mlflow


@pytest.mark.parametrize(
    "compose_file",
    [
        "docker-compose.mssql-test.yaml",
        "docker-compose.mysql-test.yaml",
        "docker-compose.postgres-test.yaml",
    ],
)
def test_backend_and_artifact_store_integration(compose_file):
    compose = DockerCompose(
        context=os.path.dirname(os.path.abspath(__file__)),
        compose_file_name=[compose_file],
    )
    # Configure wait strategy before starting containers
    compose.waiting_for(
        {
            "mlflow": HttpWaitStrategy(5000, "/health")
            .for_status_code(200)
            .with_startup_timeout(timedelta(minutes=5))
        }
    )

    with compose:
        base_url = "http://localhost:5000"

        mlflow.set_tracking_uri(base_url)
        mlflow.set_experiment("integration-test")

        @mlflow.trace
        def predict(model_input: list[str]) -> list[str]:
            return model_input

        with mlflow.start_run():
            mlflow.log_param("param", 1)
            mlflow.log_metric("metric", 1.0)
            mlflow.pyfunc.log_model(
                name="test_model",
                python_model=predict,
                input_example=["a", "b", "c"],
            )
