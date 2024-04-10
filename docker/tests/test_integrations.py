import os

import pytest
from testcontainers.compose import DockerCompose

import mlflow
from mlflow.tracking.client import MlflowClient


class TestModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input, params=None):
        pass


_COMPOSE_FILE_LIST = [
    "docker-compose.aws-test.yaml",
    "docker-compose.azure-test.yaml",
    "docker-compose.gcp-test.yaml",
    "docker-compose.mssql-test.yaml",
    "docker-compose.mysql-test.yaml",
    "docker-compose.postgres-test.yaml",
]


def create_experiment_and_register_model(base_url: str):
    experiment_name = "integration-test-experiment"
    model_name = "integration-test-model"
    mlflow.set_tracking_uri(base_url)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    test_model = TestModel()
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.log_params({"param": 1})
        mlflow.log_metric("metric", 1.0)
        mlflow.pyfunc.log_model("model", python_model=test_model)
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name, tags={"status": "ready"})
        client = MlflowClient()
        client.set_registered_model_alias(model_name, "champion", "1")


@pytest.mark.parametrize("compose_file_name", _COMPOSE_FILE_LIST)
def test_backend_and_artifact_store_integration(compose_file_name):
    with DockerCompose(
        filepath=os.path.dirname(os.path.abspath(__file__)), compose_file_name=[compose_file_name]
    ) as compose:
        mlflow_host = compose.get_service_host("mlflow", 5000)
        mlflow_port = compose.get_service_port("mlflow", 5000)
        base_url = f"http://{mlflow_host}:{mlflow_port}"
        compose.wait_for(base_url)
        create_experiment_and_register_model(base_url)
