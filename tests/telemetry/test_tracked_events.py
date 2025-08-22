import json
from unittest import mock

import pandas as pd
import pytest
import sklearn.neighbors as knn

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.optimize.types import LLMParams, OptimizerOutput
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.builtin_scorers import RelevanceToQuery
from mlflow.pyfunc.model import ResponsesAgent, ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.telemetry.client import TelemetryClient
from mlflow.telemetry.events import (
    CreateExperimentEvent,
    CreateLoggedModelEvent,
    CreateModelVersionEvent,
    CreatePromptEvent,
    CreateRegisteredModelEvent,
    CreateRunEvent,
    EvaluateEvent,
    GenAIEvaluateEvent,
    GetLoggedModelEvent,
    LogAssessmentEvent,
    PromptOptimizationEvent,
    StartTraceEvent,
)
from mlflow.tracking.fluent import _initialize_logged_model

from tests.telemetry.helper_functions import validate_telemetry_record


class TestModel(mlflow.pyfunc.PythonModel):
    def predict(self, model_input: list[str]) -> str:
        return "test"


@pytest.fixture
def mlflow_client():
    return MlflowClient()


@pytest.fixture(autouse=True)
def mock_get_telemetry_client(mock_telemetry_client: TelemetryClient):
    with mock.patch(
        "mlflow.telemetry.track.get_telemetry_client", return_value=mock_telemetry_client
    ):
        yield


def test_create_logged_model(mock_requests, mock_telemetry_client: TelemetryClient):
    event_name = CreateLoggedModelEvent.name
    mlflow.create_external_model(name="model")
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, event_name, {"flavor": "external"}
    )

    mlflow.initialize_logged_model(name="model", tags={"key": "value"})
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, event_name, {"flavor": "initialize"}
    )

    _initialize_logged_model(name="model", flavor="keras")
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name, {"flavor": "keras"})

    mlflow.pyfunc.log_model(
        name="model",
        python_model=TestModel(),
    )
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, event_name, {"flavor": "pyfunc.CustomPythonModel"}
    )

    mlflow.sklearn.log_model(
        knn.KNeighborsClassifier(),
        name="model",
    )
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, event_name, {"flavor": "sklearn"}
    )

    class SimpleResponsesAgent(ResponsesAgent):
        def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
            mock_response = {
                "output": [
                    {
                        "type": "message",
                        "id": "1234",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": request.input[0].content,
                            }
                        ],
                    }
                ],
            }
            return ResponsesAgentResponse(**mock_response)

    mlflow.pyfunc.log_model(
        name="model",
        python_model=SimpleResponsesAgent(),
    )
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, event_name, {"flavor": "pyfunc.ResponsesAgent"}
    )


def test_create_experiment(mock_requests, mlflow_client, mock_telemetry_client: TelemetryClient):
    event_name = CreateExperimentEvent.name
    mlflow.create_experiment(name="test_experiment")
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name)

    mlflow_client.create_experiment(name="test_experiment1")
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name)


def test_create_run(mock_requests, mlflow_client, mock_telemetry_client: TelemetryClient):
    event_name = CreateRunEvent.name
    exp_id = mlflow.create_experiment(name="test_experiment")
    with mlflow.start_run(experiment_id=exp_id):
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, event_name, check_params=False
        )

    mlflow_client.create_run(experiment_id=exp_id)
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name, check_params=False)


def test_create_run_with_imports(mock_requests, mock_telemetry_client: TelemetryClient):
    event_name = CreateRunEvent.name
    import pyspark.ml  # noqa: F401

    with mlflow.start_run():
        data = validate_telemetry_record(
            mock_telemetry_client, mock_requests, event_name, check_params=False
        )
        assert "pyspark.ml" in json.loads(data["params"])["imports"]


def test_create_registered_model(
    mock_requests, mlflow_client, mock_telemetry_client: TelemetryClient
):
    event_name = CreateRegisteredModelEvent.name
    mlflow_client.create_registered_model(name="test_model1")
    validate_telemetry_record(
        mock_telemetry_client,
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
        mock_telemetry_client,
        mock_requests,
        event_name,
        {"is_prompt": False},
    )


def test_create_model_version(mock_requests, mlflow_client, mock_telemetry_client: TelemetryClient):
    event_name = CreateModelVersionEvent.name
    mlflow_client.create_registered_model(name="test_model")
    mlflow_client.create_model_version(
        name="test_model", source="test_source", run_id="test_run_id"
    )
    validate_telemetry_record(
        mock_telemetry_client,
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
        mock_telemetry_client,
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
        mock_telemetry_client,
        mock_requests,
        event_name,
        {"is_prompt": True},
    )


def test_start_trace(mock_requests, mlflow_client, mock_telemetry_client: TelemetryClient):
    event_name = StartTraceEvent.name
    with mlflow.start_span(name="test_span"):
        pass
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name)

    @mlflow.trace
    def test_func():
        pass

    test_func()
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name)

    trace_id = mlflow_client.start_trace(name="test_trace").trace_id
    mlflow_client.end_trace(trace_id=trace_id)
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name)


def test_create_prompt(mock_requests, mlflow_client, mock_telemetry_client: TelemetryClient):
    mlflow_client.create_prompt(name="test_prompt")
    validate_telemetry_record(mock_telemetry_client, mock_requests, CreatePromptEvent.name)

    # OSS prompt registry uses create_registered_model with a special tag
    mlflow.genai.register_prompt(
        name="greeting_prompt",
        template="Respond to the user's message as a {{style}} AI. {{greeting}}",
    )
    expected_params = {"is_prompt": True}
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        CreateRegisteredModelEvent.name,
        expected_params,
    )


def test_log_assessment(mock_requests, mock_telemetry_client: TelemetryClient):
    with mlflow.start_span(name="test_span") as span:
        feedback = Feedback(
            name="faithfulness",
            value=0.9,
            rationale="The model is faithful to the input.",
            metadata={"model": "gpt-4o-mini"},
        )

        mlflow.log_assessment(trace_id=span.trace_id, assessment=feedback)
    validate_telemetry_record(mock_telemetry_client, mock_requests, LogAssessmentEvent.name)


def test_evaluate(mock_requests, mock_telemetry_client: TelemetryClient):
    mlflow.models.evaluate(
        data=pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    validate_telemetry_record(mock_telemetry_client, mock_requests, EvaluateEvent.name)


def test_genai_evaluate(mock_requests, mock_telemetry_client: TelemetryClient):
    @mlflow.genai.scorer
    def sample_scorer(inputs, outputs, expectations):
        return 1.0

    model = TestModel()
    data = [
        {
            "inputs": {"model_input": ["What is the capital of France?"]},
            "outputs": "The capital of France is Paris.",
        }
    ]
    with mock.patch("mlflow.genai.judges.is_context_relevant"):
        mlflow.genai.evaluate(
            data=data, scorers=[sample_scorer, RelevanceToQuery()], predict_fn=model.predict
        )
        expected_params = {"builtin_scorers": ["relevance_to_query"]}
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, GenAIEvaluateEvent.name, expected_params
        )


def test_prompt_optimization(mock_requests, mock_telemetry_client: TelemetryClient):
    sample_prompt = mlflow.genai.register_prompt(
        name="test_translation_prompt",
        template="Translate the following text to {{language}}: {{input_text}}",
    )
    sample_data = pd.DataFrame(
        {
            "inputs": [
                {"input_text": "Hello", "language": "Spanish"},
                {"input_text": "World", "language": "French"},
            ],
            "expectations": [{"translation": "Hola"}, {"translation": "Monde"}],
        }
    )

    @scorer
    def sample_scorer(inputs, outputs, expectations):
        return 1.0

    with mock.patch(
        "mlflow.genai.optimize.base._DSPyMIPROv2Optimizer.optimize",
        return_value=OptimizerOutput(
            final_eval_score=1.0,
            initial_eval_score=0.5,
            optimizer_name="DSPy/MIPROv2",
            optimized_prompt="optimized",
        ),
    ):
        mlflow.genai.optimize_prompt(
            target_llm_params=LLMParams(model_name="test/model"),
            prompt=f"prompts:/{sample_prompt.name}/{sample_prompt.version}",
            train_data=sample_data,
            scorers=[sample_scorer],
        )
    validate_telemetry_record(mock_telemetry_client, mock_requests, PromptOptimizationEvent.name)


def test_get_logged_model(mock_requests, mock_telemetry_client: TelemetryClient, tmp_path):
    model_info = mlflow.sklearn.log_model(
        knn.KNeighborsClassifier(),
        name="model",
    )
    mock_telemetry_client.flush()

    mlflow.sklearn.load_model(model_info.model_uri)
    validate_telemetry_record(mock_telemetry_client, mock_requests, GetLoggedModelEvent.name)

    mlflow.pyfunc.load_model(model_info.model_uri)
    validate_telemetry_record(mock_telemetry_client, mock_requests, GetLoggedModelEvent.name)

    model_def = """
import mlflow
from mlflow.models import set_model

class TestModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: list[str], params=None) -> list[str]:
        return model_input

set_model(TestModel())
"""
    model_path = tmp_path / "model.py"
    model_path.write_text(model_def)
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=model_path,
    )
    mock_telemetry_client.flush()

    mlflow.pyfunc.load_model(model_info.model_uri)
    validate_telemetry_record(mock_telemetry_client, mock_requests, GetLoggedModelEvent.name)

    # test load model after registry
    mlflow.register_model(model_info.model_uri, name="test")
    mock_telemetry_client.flush()

    mlflow.pyfunc.load_model("models:/test/1")
    validate_telemetry_record(mock_telemetry_client, mock_requests, GetLoggedModelEvent.name)
