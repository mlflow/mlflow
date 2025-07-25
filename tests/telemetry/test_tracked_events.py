import json
from unittest import mock

import pandas as pd
import pytest
import sklearn.neighbors as knn

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.telemetry.client import get_telemetry_client, set_telemetry_client
from mlflow.telemetry.events import (
    CreateExperimentEvent,
    CreateLoggedModelEvent,
    CreateModelVersionEvent,
    CreatePromptEvent,
    CreateRegisteredModelEvent,
    CreateRunEvent,
    EvaluateEvent,
    LogAssessmentEvent,
    StartTraceEvent,
)
from mlflow.version import VERSION

from tests.telemetry.helper_functions import validate_telemetry_record


class TestModel(mlflow.pyfunc.PythonModel):
    def predict(self, model_input: list[str]) -> list[str]:
        return model_input


@pytest.fixture
def mlflow_client():
    return MlflowClient()


def test_create_logged_model(mock_requests):
    event_name = CreateLoggedModelEvent.name
    mlflow.create_external_model(name="model")
    validate_telemetry_record(mock_requests, event_name)

    mlflow.initialize_logged_model(name="model", tags={"key": "value"})
    validate_telemetry_record(mock_requests, event_name)

    mlflow.pyfunc.log_model(
        name="model",
        python_model=TestModel(),
    )
    validate_telemetry_record(mock_requests, event_name, {"flavor": "pyfunc"})

    mlflow.sklearn.log_model(
        knn.KNeighborsClassifier(),
        name="model",
    )
    validate_telemetry_record(mock_requests, event_name, {"flavor": "sklearn"})


def test_create_experiment(mock_requests, mlflow_client):
    event_name = CreateExperimentEvent.name
    mlflow.create_experiment(name="test_experiment")
    validate_telemetry_record(mock_requests, event_name)

    mlflow_client.create_experiment(name="test_experiment1")
    validate_telemetry_record(mock_requests, event_name)


def test_create_run(mock_requests, mlflow_client):
    event_name = CreateRunEvent.name
    exp_id = mlflow.create_experiment(name="test_experiment")
    with mlflow.start_run(experiment_id=exp_id):
        validate_telemetry_record(mock_requests, event_name, check_params=False)

    mlflow_client.create_run(experiment_id=exp_id)
    validate_telemetry_record(mock_requests, event_name, check_params=False)


def test_create_run_with_imports(mock_requests):
    event_name = CreateRunEvent.name
    import pyspark.ml  # noqa: F401

    with mlflow.start_run():
        data = validate_telemetry_record(mock_requests, event_name, check_params=False)
        assert "pyspark.ml" in json.loads(data["params"])["imports"]


def test_create_registered_model(mock_requests, mlflow_client):
    event_name = CreateRegisteredModelEvent.name
    mlflow_client.create_registered_model(name="test_model1")
    validate_telemetry_record(
        mock_requests,
        event_name,
        {"is_prompt": False},
    )

    mlflow.pyfunc.log_model(
        name="model",
        python_model=TestModel(),
        registered_model_name="test_model",
    )
    validate_telemetry_record(
        mock_requests,
        event_name,
        {"is_prompt": False},
    )


def test_create_model_version(mock_requests, mlflow_client):
    event_name = CreateModelVersionEvent.name
    mlflow_client.create_registered_model(name="test_model")
    mlflow_client.create_model_version(
        name="test_model", source="test_source", run_id="test_run_id"
    )
    validate_telemetry_record(
        mock_requests,
        event_name,
        {"is_prompt": False},
    )

    mlflow.pyfunc.log_model(
        name="model",
        python_model=TestModel(),
        registered_model_name="test_model",
    )
    validate_telemetry_record(
        mock_requests,
        event_name,
        {"is_prompt": False},
    )

    mlflow.genai.register_prompt(
        name="ai_assistant_prompt",
        template="Respond to the user's message as a {{style}} AI. {{greeting}}",
        commit_message="Initial version of AI assistant",
    )
    validate_telemetry_record(
        mock_requests,
        event_name,
        {"is_prompt": True},
    )


def test_start_trace(mock_requests, mlflow_client):
    event_name = StartTraceEvent.name
    with mlflow.start_span(name="test_span"):
        pass
    validate_telemetry_record(mock_requests, event_name)

    @mlflow.trace
    def test_func():
        pass

    test_func()
    validate_telemetry_record(mock_requests, event_name)

    trace_id = mlflow_client.start_trace(name="test_trace").trace_id
    mlflow_client.end_trace(trace_id=trace_id)
    validate_telemetry_record(mock_requests, event_name)


def test_create_prompt(mock_requests, mlflow_client):
    mlflow_client.create_prompt(name="test_prompt")
    validate_telemetry_record(mock_requests, CreatePromptEvent.name)

    # OSS prompt registry uses create_registered_model with a special tag
    mlflow.genai.register_prompt(
        name="greeting_prompt",
        template="Respond to the user's message as a {{style}} AI. {{greeting}}",
    )
    expected_params = {"is_prompt": True}
    validate_telemetry_record(
        mock_requests,
        CreateRegisteredModelEvent.name,
        expected_params,
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
    validate_telemetry_record(mock_requests, LogAssessmentEvent.name)


def test_evaluate(mock_requests):
    mlflow.models.evaluate(
        data=pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    validate_telemetry_record(mock_requests, EvaluateEvent.name)


@pytest.mark.no_mock_requests_get
def test_disable_events(mock_requests):
    with mock.patch("mlflow.telemetry.client.requests.get") as mock_requests_get:
        mock_requests_get.return_value = mock.Mock(
            status_code=200,
            json=mock.Mock(
                return_value={
                    "mlflow_version": VERSION,
                    "disable_telemetry": False,
                    "ingestion_url": "http://localhost:9999",
                    "rollout_percentage": 100,
                    "disable_events": [CreateLoggedModelEvent.name],
                    "disable_sdks": [],
                }
            ),
        )
        set_telemetry_client()

        mlflow.create_external_model(name="model")
        mlflow.initialize_logged_model(name="model", tags={"key": "value"})
        mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
        )
        get_telemetry_client().flush()
        assert len(mock_requests) == 0

        with mlflow.start_run():
            pass
        validate_telemetry_record(mock_requests, CreateRunEvent.name, check_params=False)
