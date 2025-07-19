import json
from unittest import mock

import pandas as pd
import pytest
import sklearn.neighbors as knn

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.models.evaluation.base import _evaluate
from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.schemas import LoggedModelParams, RegisteredModelParams
from mlflow.tracing.client import TracingClient
from mlflow.tracking._model_registry.client import ModelRegistryClient
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.version import VERSION

from tests.telemetry.helper_functions import validate_telemetry_record


class TestModel(mlflow.pyfunc.PythonModel):
    def predict(self, model_input: list[str]) -> list[str]:
        return model_input


@pytest.fixture
def mlflow_client():
    return MlflowClient()


def test_create_logged_model(mock_requests):
    func = TrackingServiceClient.create_logged_model

    mlflow.create_external_model(name="model")
    validate_telemetry_record(mock_requests, func)

    mlflow.initialize_logged_model(name="model", tags={"key": "value"})
    validate_telemetry_record(mock_requests, func)

    mlflow.pyfunc.log_model(
        name="model",
        python_model=TestModel(),
    )
    validate_telemetry_record(mock_requests, func, LoggedModelParams(flavor="pyfunc"))

    mlflow.sklearn.log_model(
        knn.KNeighborsClassifier(),
        name="model",
    )
    validate_telemetry_record(mock_requests, func, LoggedModelParams(flavor="sklearn"))


def test_create_experiment(mock_requests, mlflow_client):
    mlflow.create_experiment(name="test_experiment")
    validate_telemetry_record(mock_requests, TrackingServiceClient.create_experiment)

    mlflow_client.create_experiment(name="test_experiment1")
    validate_telemetry_record(mock_requests, TrackingServiceClient.create_experiment)


def test_create_run(mock_requests, mlflow_client):
    exp_id = mlflow.create_experiment(name="test_experiment")
    with mlflow.start_run(experiment_id=exp_id):
        validate_telemetry_record(
            mock_requests, TrackingServiceClient.create_run, search_index=True, check_params=False
        )

    mlflow_client.create_run(experiment_id=exp_id)
    validate_telemetry_record(mock_requests, TrackingServiceClient.create_run, check_params=False)


def test_create_run_with_imports(mock_requests):
    import lightgbm  # noqa: F401

    with mlflow.start_run():
        data = validate_telemetry_record(
            mock_requests, TrackingServiceClient.create_run, check_params=False
        )
        assert "lightgbm" in json.loads(data["params"])["imports"]


def test_create_registered_model(mock_requests, mlflow_client):
    mlflow_client.create_registered_model(name="test_model1")
    validate_telemetry_record(
        mock_requests,
        ModelRegistryClient.create_registered_model,
        RegisteredModelParams(is_prompt=False),
    )

    mlflow.pyfunc.log_model(
        name="model",
        python_model=TestModel(),
        registered_model_name="test_model",
    )
    validate_telemetry_record(
        mock_requests,
        ModelRegistryClient.create_registered_model,
        RegisteredModelParams(is_prompt=False),
        search_index=True,
    )


def test_create_model_version(mock_requests, mlflow_client):
    mlflow_client.create_registered_model(name="test_model")
    mlflow_client.create_model_version(
        name="test_model", source="test_source", run_id="test_run_id"
    )
    validate_telemetry_record(
        mock_requests, ModelRegistryClient.create_model_version, search_index=True
    )

    mlflow.pyfunc.log_model(
        name="model",
        python_model=TestModel(),
        registered_model_name="test_model",
    )
    validate_telemetry_record(
        mock_requests, ModelRegistryClient.create_model_version, search_index=True
    )


def test_start_trace(mock_requests, mlflow_client):
    with mlflow.start_span(name="test_span"):
        pass
    validate_telemetry_record(mock_requests, TracingClient.start_trace)

    @mlflow.trace
    def test_func():
        pass

    test_func()
    validate_telemetry_record(mock_requests, TracingClient.start_trace)

    trace_id = mlflow_client.start_trace(name="test_trace").trace_id
    mlflow_client.end_trace(trace_id=trace_id)
    validate_telemetry_record(mock_requests, TracingClient.start_trace)


def test_create_prompt(mock_requests, mlflow_client):
    mlflow_client.create_prompt(name="test_prompt")
    validate_telemetry_record(mock_requests, ModelRegistryClient.create_prompt, search_index=True)

    # OSS prompt registry uses create_registered_model with a special tag
    mlflow.genai.register_prompt(
        name="greeting_prompt",
        template="Respond to the user's message as a {{style}} AI. {{greeting}}",
    )
    expected_params = RegisteredModelParams(is_prompt=True)
    validate_telemetry_record(
        mock_requests,
        ModelRegistryClient.create_registered_model,
        expected_params,
        search_index=True,
    )


def test_log_assessment(mock_requests):
    with mlflow.start_span(name="test_span") as span:
        feedback = Feedback(
            name="faithfulness",
            value=0.9,
            rationale="The model is faithful to the input.",
            metadata={"model": "gpt-4o-mini"},
        )

        mlflow.log_assessment(trace_id=span.trace_id, assessment=feedback)
    validate_telemetry_record(mock_requests, TracingClient.log_assessment)


def test_evaluate(mock_requests):
    mlflow.models.evaluate(
        data=pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    validate_telemetry_record(mock_requests, _evaluate, search_index=True)


def test_disable_api_map(mock_requests):
    disable_api = TrackingServiceClient.create_logged_model
    with mock.patch("requests.get") as mock_requests:
        mock_requests.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "telemetry_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_api_map": {disable_api.__module__: [disable_api.__qualname__]},
                    "disable_sdks": [],
                }
            ),
        )

        mlflow.create_external_model(name="model")
        mlflow.initialize_logged_model(name="model", tags={"key": "value"})
        mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
        )
        get_telemetry_client().flush()
        assert len(mock_requests) == 0
