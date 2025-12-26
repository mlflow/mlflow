import json
import time
from unittest import mock

import pandas as pd
import pytest
import sklearn.neighbors as knn
from click.testing import CliRunner

import mlflow
from mlflow import MlflowClient
from mlflow.entities import EvaluationDataset, Expectation, Feedback, Metric, Param, RunTag
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.gateway.cli import start
from mlflow.genai.datasets import create_dataset
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.base import AlignmentOptimizer
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.builtin_scorers import (
    Completeness,
    Guidelines,
    RelevanceToQuery,
    Safety,
    UserFrustration,
)
from mlflow.pyfunc.model import ResponsesAgent, ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.telemetry.client import TelemetryClient
from mlflow.telemetry.events import (
    AiCommandRunEvent,
    AlignJudgeEvent,
    AutologgingEvent,
    CreateDatasetEvent,
    CreateExperimentEvent,
    CreateLoggedModelEvent,
    CreateModelVersionEvent,
    CreatePromptEvent,
    CreateRegisteredModelEvent,
    CreateRunEvent,
    CreateWebhookEvent,
    EvaluateEvent,
    GatewayStartEvent,
    GenAIEvaluateEvent,
    GetLoggedModelEvent,
    GitModelVersioningEvent,
    InvokeCustomJudgeModelEvent,
    LoadPromptEvent,
    LogAssessmentEvent,
    LogBatchEvent,
    LogDatasetEvent,
    LogMetricEvent,
    LogParamEvent,
    MakeJudgeEvent,
    McpRunEvent,
    MergeRecordsEvent,
    PromptOptimizationEvent,
    ScorerCallEvent,
    StartTraceEvent,
)
from mlflow.tracking.fluent import _create_dataset_input, _initialize_logged_model
from mlflow.utils.os import is_windows

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
    exp_id = mlflow.create_experiment(name="test_experiment")
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, event_name, {"experiment_id": exp_id}
    )

    exp_id = mlflow_client.create_experiment(name="test_experiment1")
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, event_name, {"experiment_id": exp_id}
    )


def test_create_run(mock_requests, mlflow_client, mock_telemetry_client: TelemetryClient):
    event_name = CreateRunEvent.name
    exp_id = mlflow.create_experiment(name="test_experiment")
    with mlflow.start_run(experiment_id=exp_id):
        record = validate_telemetry_record(
            mock_telemetry_client, mock_requests, event_name, check_params=False
        )
        assert json.loads(record["params"])["experiment_id"] == exp_id

    mlflow_client.create_run(experiment_id=exp_id)
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name, check_params=False)

    exp_id = mlflow.create_experiment(name="test_experiment2")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run():
        record = validate_telemetry_record(
            mock_telemetry_client, mock_requests, event_name, check_params=False
        )
        params = json.loads(record["params"])
        assert params["mlflow_experiment_id"] == exp_id


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
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name, check_params=False)

    @mlflow.trace
    def test_func():
        pass

    test_func()
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name, check_params=False)

    trace_id = mlflow_client.start_trace(name="test_trace").trace_id
    mlflow_client.end_trace(trace_id=trace_id)
    validate_telemetry_record(mock_telemetry_client, mock_requests, event_name, check_params=False)

    import openai  # noqa: F401

    test_func()
    data = validate_telemetry_record(
        mock_telemetry_client, mock_requests, event_name, check_params=False
    )
    assert "openai" in json.loads(data["params"])["imports"]


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
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        LogAssessmentEvent.name,
        {"type": "feedback", "source_type": "CODE"},
    )
    mlflow.log_feedback(trace_id=span.trace_id, value=0.9, name="faithfulness")
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        LogAssessmentEvent.name,
        {"type": "feedback", "source_type": "CODE"},
    )

    with mlflow.start_span(name="test_span2") as span:
        expectation = Expectation(
            name="expected_answer",
            value="MLflow",
        )

        mlflow.log_assessment(trace_id=span.trace_id, assessment=expectation)
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        LogAssessmentEvent.name,
        {"type": "expectation", "source_type": "HUMAN"},
    )
    mlflow.log_expectation(trace_id=span.trace_id, value="MLflow", name="expected_answer")
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        LogAssessmentEvent.name,
        {"type": "expectation", "source_type": "HUMAN"},
    )


def test_evaluate(mock_requests, mock_telemetry_client: TelemetryClient):
    mlflow.models.evaluate(
        data=pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    validate_telemetry_record(mock_telemetry_client, mock_requests, EvaluateEvent.name)


def test_create_webhook(mock_requests, mock_telemetry_client: TelemetryClient):
    client = MlflowClient()
    client.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
    )
    expected_params = {"events": ["model_version.created"]}
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, CreateWebhookEvent.name, expected_params
    )


def test_genai_evaluate(mock_requests, mock_telemetry_client: TelemetryClient):
    @mlflow.genai.scorer
    def decorator_scorer():
        return 1.0

    instructions_judge = make_judge(
        name="quality_judge",
        instructions="Evaluate if {{ outputs }} is high quality",
        model="openai:/gpt-4",
    )

    session_level_instruction_judge = make_judge(
        name="conversation_quality",
        instructions="Evaluate if the {{ conversation }} is engaging and coherent",
        model="openai:/gpt-4",
    )

    guidelines_scorer = Guidelines(
        name="politeness",
        guidelines=["Be polite", "Be respectful"],
    )

    builtin_scorer = RelevanceToQuery(name="relevance_check")

    session_level_builtin_scorer = UserFrustration(name="frustration_check")

    data = [
        {
            "inputs": {"model_input": ["What is MLflow?"]},
            "outputs": "MLflow is an open source platform.",
        }
    ]

    model = TestModel()

    with (
        mock.patch("mlflow.genai.judges.utils.invocation_utils.invoke_judge_model"),
    ):
        # Test with all scorer kinds and scopes, without predict_fn
        mlflow.genai.evaluate(
            data=data,
            scorers=[
                decorator_scorer,
                instructions_judge,
                session_level_instruction_judge,
                guidelines_scorer,
                builtin_scorer,
                session_level_builtin_scorer,
            ],
        )

        expected_params = {
            "predict_fn_provided": False,
            "scorer_info": [
                {"class": "UserDefinedScorer", "kind": "decorator", "scope": "response"},
                {"class": "UserDefinedScorer", "kind": "instructions", "scope": "response"},
                {"class": "UserDefinedScorer", "kind": "instructions", "scope": "session"},
                {"class": "Guidelines", "kind": "guidelines", "scope": "response"},
                {"class": "RelevanceToQuery", "kind": "builtin", "scope": "response"},
                {"class": "UserFrustration", "kind": "builtin", "scope": "session"},
            ],
            "eval_data_type": "list[dict]",
            "eval_data_size": 1,
            "eval_data_provided_fields": ["inputs", "outputs"],
        }
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, GenAIEvaluateEvent.name, expected_params
        )

        # Test with predict_fn
        mlflow.genai.evaluate(
            data=data,
            scorers=[builtin_scorer, guidelines_scorer],
            predict_fn=model.predict,
        )
        expected_params = {
            "predict_fn_provided": True,
            "scorer_info": [
                {"class": "RelevanceToQuery", "kind": "builtin", "scope": "response"},
                {"class": "Guidelines", "kind": "guidelines", "scope": "response"},
            ],
            "eval_data_type": "list[dict]",
            "eval_data_size": 1,
            "eval_data_provided_fields": ["inputs", "outputs"],
        }
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, GenAIEvaluateEvent.name, expected_params
        )


def test_genai_evaluate_telemetry_data_fields(
    mock_requests, mock_telemetry_client: TelemetryClient
):
    @mlflow.genai.scorer
    def sample_scorer():
        return 1.0

    with mock.patch("mlflow.genai.judges.utils.invocation_utils.invoke_judge_model"):
        # Test with list of dicts
        data_list = [
            {
                "inputs": {"question": "Q1"},
                "outputs": "A1",
                "expectations": {"answer": "Expected1"},
            },
            {
                "inputs": {"question": "Q2"},
                "outputs": "A2",
                "expectations": {"answer": "Expected2"},
            },
        ]
        mlflow.genai.evaluate(data=data_list, scorers=[sample_scorer])
        expected_params = {
            "predict_fn_provided": False,
            "scorer_info": [
                {"class": "UserDefinedScorer", "kind": "decorator", "scope": "response"},
            ],
            "eval_data_type": "list[dict]",
            "eval_data_size": 2,
            "eval_data_provided_fields": ["expectations", "inputs", "outputs"],
        }
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, GenAIEvaluateEvent.name, expected_params
        )

        # Test with pandas DataFrame
        df_data = pd.DataFrame(
            [
                {"inputs": {"question": "Q1"}, "outputs": "A1"},
                {"inputs": {"question": "Q2"}, "outputs": "A2"},
                {"inputs": {"question": "Q3"}, "outputs": "A3"},
            ]
        )
        mlflow.genai.evaluate(data=df_data, scorers=[sample_scorer])
        expected_params = {
            "predict_fn_provided": False,
            "scorer_info": [
                {"class": "UserDefinedScorer", "kind": "decorator", "scope": "response"},
            ],
            "eval_data_type": "pd.DataFrame",
            "eval_data_size": 3,
            "eval_data_provided_fields": ["inputs", "outputs"],
        }
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, GenAIEvaluateEvent.name, expected_params
        )

        # Test with list of Traces
        trace_ids = []
        for i in range(2):
            with mlflow.start_span(name=f"test_span_{i}") as span:
                span.set_inputs({"question": f"Q{i}"})
                span.set_outputs({"answer": f"A{i}"})
                trace_ids.append(span.trace_id)

        traces = [mlflow.get_trace(trace_id) for trace_id in trace_ids]
        mlflow.genai.evaluate(data=traces, scorers=[sample_scorer])
        expected_params = {
            "predict_fn_provided": False,
            "scorer_info": [
                {"class": "UserDefinedScorer", "kind": "decorator", "scope": "response"},
            ],
            "eval_data_type": "list[Trace]",
            "eval_data_size": 2,
            "eval_data_provided_fields": ["inputs", "outputs", "trace"],
        }
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, GenAIEvaluateEvent.name, expected_params
        )

        # Test with EvaluationDataset
        from mlflow.genai.datasets import create_dataset

        dataset = create_dataset("test_dataset")
        dataset_data = [
            {
                "inputs": {"question": "Q1"},
                "outputs": "A1",
                "expectations": {"answer": "Expected1"},
            },
            {
                "inputs": {"question": "Q2"},
                "outputs": "A2",
                "expectations": {"answer": "Expected2"},
            },
        ]
        dataset.merge_records(dataset_data)
        mlflow.genai.evaluate(data=dataset, scorers=[sample_scorer])
        expected_params = {
            "predict_fn_provided": False,
            "scorer_info": [
                {"class": "UserDefinedScorer", "kind": "decorator", "scope": "response"},
            ],
            "eval_data_type": "EvaluationDataset",
            "eval_data_size": 2,
            "eval_data_provided_fields": ["expectations", "inputs", "outputs"],
        }
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, GenAIEvaluateEvent.name, expected_params
        )


def test_prompt_optimization(mock_requests, mock_telemetry_client: TelemetryClient):
    from mlflow.genai.optimize import optimize_prompts
    from mlflow.genai.optimize.optimizers import BasePromptOptimizer
    from mlflow.genai.optimize.types import PromptOptimizerOutput

    class MockAdapter(BasePromptOptimizer):
        def __init__(self):
            self.model_name = "openai:/gpt-4o-mini"

        def optimize(self, eval_fn, train_data, target_prompts, enable_tracking):
            return PromptOptimizerOutput(optimized_prompts=target_prompts)

    sample_prompt = mlflow.genai.register_prompt(
        name="test_prompt_for_adaptation",
        template="Translate {{input_text}} to {{language}}",
    )

    sample_data = [
        {"inputs": {"input_text": "Hello", "language": "Spanish"}, "outputs": "Hola"},
        {"inputs": {"input_text": "World", "language": "French"}, "outputs": "Monde"},
    ]

    @mlflow.genai.scorers.scorer
    def exact_match_scorer(outputs, expectations):
        return 1.0 if outputs == expectations["expected_response"] else 0.0

    def predict_fn(input_text, language):
        mlflow.genai.load_prompt(f"prompts:/{sample_prompt.name}/{sample_prompt.version}")
        return "translated"

    optimize_prompts(
        predict_fn=predict_fn,
        train_data=sample_data,
        prompt_uris=[f"prompts:/{sample_prompt.name}/{sample_prompt.version}"],
        optimizer=MockAdapter(),
        scorers=[exact_match_scorer],
    )
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        PromptOptimizationEvent.name,
        {
            "optimizer_type": "MockAdapter",
            "prompt_count": 1,
            "scorer_count": 1,
            "custom_aggregation": False,
        },
    )


def test_create_dataset(mock_requests, mock_telemetry_client: TelemetryClient):
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_store:
        mock_store_instance = mock.MagicMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.create_dataset.return_value = mock.MagicMock(
            dataset_id="test-dataset-id", name="test_dataset", tags={"test": "value"}
        )

        create_dataset(name="test_dataset", tags={"test": "value"})
        validate_telemetry_record(mock_telemetry_client, mock_requests, CreateDatasetEvent.name)


def test_merge_records(mock_requests, mock_telemetry_client: TelemetryClient):
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_store:
        mock_store_instance = mock.MagicMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.get_dataset.return_value = mock.MagicMock(dataset_id="test-id")
        mock_store_instance.upsert_dataset_records.return_value = {"inserted": 2, "updated": 0}

        evaluation_dataset = EvaluationDataset(
            dataset_id="test-id",
            name="test",
            digest="digest",
            created_time=123,
            last_update_time=456,
        )

        records = [
            {"inputs": {"q": "Q1"}, "expectations": {"a": "A1"}},
            {"inputs": {"q": "Q2"}, "expectations": {"a": "A2"}},
        ]
        evaluation_dataset.merge_records(records)

        expected_params = {"record_count": 2, "input_type": "list[dict]"}
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, MergeRecordsEvent.name, expected_params
        )


def test_log_dataset(mock_requests, mock_telemetry_client: TelemetryClient):
    with mlflow.start_run() as run:
        dataset = mlflow.data.from_pandas(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
        mlflow.log_input(dataset)
        validate_telemetry_record(mock_telemetry_client, mock_requests, LogDatasetEvent.name)

        mlflow.log_inputs(datasets=[dataset], contexts=["training"], tags_list=[None])
        validate_telemetry_record(mock_telemetry_client, mock_requests, LogDatasetEvent.name)

        client = MlflowClient()
        client.log_inputs(run_id=run.info.run_id, datasets=[_create_dataset_input(dataset)])
        validate_telemetry_record(mock_telemetry_client, mock_requests, LogDatasetEvent.name)


def test_log_metric(mock_requests, mock_telemetry_client: TelemetryClient):
    with mlflow.start_run():
        mlflow.log_metric("test_metric", 1.0)
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, LogMetricEvent.name, {"synchronous": True}
        )

        mlflow.log_metric("test_metric", 1.0, synchronous=False)
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, LogMetricEvent.name, {"synchronous": False}
        )

        client = MlflowClient()
        client.log_metric(
            run_id=mlflow.active_run().info.run_id,
            key="test_metric",
            value=1.0,
            timestamp=int(time.time()),
            step=0,
        )
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, LogMetricEvent.name, {"synchronous": True}
        )

        client.log_metric(
            run_id=mlflow.active_run().info.run_id,
            key="test_metric",
            value=1.0,
            timestamp=int(time.time()),
            step=0,
            synchronous=False,
        )
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, LogMetricEvent.name, {"synchronous": False}
        )


def test_log_param(mock_requests, mock_telemetry_client: TelemetryClient):
    with mlflow.start_run():
        mlflow.log_param("test_param", "test_value")
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, LogParamEvent.name, {"synchronous": True}
        )

        mlflow.log_param("test_param", "test_value", synchronous=False)
        validate_telemetry_record(
            mock_telemetry_client, mock_requests, LogParamEvent.name, {"synchronous": False}
        )

        client = mlflow.MlflowClient()
        client.log_param(
            run_id=mlflow.active_run().info.run_id,
            key="test_param",
            value="test_value",
        )
        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            LogParamEvent.name,
            {"synchronous": True},
        )


def test_log_batch(mock_requests, mock_telemetry_client: TelemetryClient):
    with mlflow.start_run():
        mlflow.log_params(params={"test_param": "test_value"})
        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            LogBatchEvent.name,
            {"metrics": False, "params": True, "tags": False, "synchronous": True},
        )

        mlflow.log_params(params={"test_param": "test_value"}, synchronous=False)
        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            LogBatchEvent.name,
            {"metrics": False, "params": True, "tags": False, "synchronous": False},
        )

        mlflow.log_metrics(metrics={"test_metric": 1.0})
        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            LogBatchEvent.name,
            {"metrics": True, "params": False, "tags": False, "synchronous": True},
        )

        mlflow.log_metrics(metrics={"test_metric": 1.0}, synchronous=False)
        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            LogBatchEvent.name,
            {"metrics": True, "params": False, "tags": False, "synchronous": False},
        )

        mlflow.set_tags(tags={"test_tag": "test_value"})
        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            LogBatchEvent.name,
            {"metrics": False, "params": False, "tags": True, "synchronous": True},
        )

        mlflow.set_tags(tags={"test_tag": "test_value"}, synchronous=False)

        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            LogBatchEvent.name,
            {"metrics": False, "params": False, "tags": True, "synchronous": False},
        )

        client = mlflow.MlflowClient()
        client.log_batch(
            run_id=mlflow.active_run().info.run_id,
            metrics=[Metric(key="test_metric", value=1.0, timestamp=int(time.time()), step=0)],
            params=[Param(key="test_param", value="test_value")],
            tags=[RunTag(key="test_tag", value="test_value")],
        )
        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            LogBatchEvent.name,
            {"metrics": True, "params": True, "tags": True, "synchronous": True},
        )


def test_get_logged_model(mock_requests, mock_telemetry_client: TelemetryClient, tmp_path):
    model_info = mlflow.sklearn.log_model(
        knn.KNeighborsClassifier(),
        name="model",
    )
    mock_telemetry_client.flush()

    mlflow.sklearn.load_model(model_info.model_uri)
    data = validate_telemetry_record(
        mock_telemetry_client, mock_requests, GetLoggedModelEvent.name, check_params=False
    )
    assert "sklearn" in json.loads(data["params"])["imports"]

    mlflow.pyfunc.load_model(model_info.model_uri)
    data = validate_telemetry_record(
        mock_telemetry_client, mock_requests, GetLoggedModelEvent.name, check_params=False
    )

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
    data = validate_telemetry_record(
        mock_telemetry_client, mock_requests, GetLoggedModelEvent.name, check_params=False
    )

    # test load model after registry
    mlflow.register_model(model_info.model_uri, name="test")
    mock_telemetry_client.flush()

    mlflow.pyfunc.load_model("models:/test/1")
    data = validate_telemetry_record(
        mock_telemetry_client, mock_requests, GetLoggedModelEvent.name, check_params=False
    )


def test_mcp_run(mock_requests, mock_telemetry_client: TelemetryClient):
    from mlflow.mcp.cli import run

    runner = CliRunner(catch_exceptions=False)
    with mock.patch("mlflow.mcp.cli.run_server") as mock_run_server:
        runner.invoke(run)

    mock_run_server.assert_called_once()
    mock_telemetry_client.flush()
    validate_telemetry_record(mock_telemetry_client, mock_requests, McpRunEvent.name)


@pytest.mark.skipif(is_windows(), reason="Windows does not support gateway start")
def test_gateway_start(tmp_path, mock_requests, mock_telemetry_client: TelemetryClient):
    config = tmp_path.joinpath("config.yml")
    config.write_text(
        """
endpoints:
  - name: test-endpoint
    endpoint_type: llm/v1/completions
    model:
      provider: openai
      name: gpt-3.5-turbo
      config:
        openai_api_key: test-key
"""
    )

    runner = CliRunner(catch_exceptions=False)
    with mock.patch("mlflow.gateway.cli.run_app"):
        runner.invoke(start, ["--config-path", str(config)])

    mock_telemetry_client.flush()
    validate_telemetry_record(mock_telemetry_client, mock_requests, GatewayStartEvent.name)


def test_ai_command_run(mock_requests, mock_telemetry_client: TelemetryClient):
    from mlflow.ai_commands import commands

    runner = CliRunner(catch_exceptions=False)
    # Test CLI context
    with mock.patch("mlflow.ai_commands.get_command", return_value="---\ntest\n---\nTest command"):
        result = runner.invoke(commands, ["run", "test_command"])
        assert result.exit_code == 0

    mock_telemetry_client.flush()
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        AiCommandRunEvent.name,
        {"command_key": "test_command", "context": "cli"},
    )


def test_git_model_versioning(mock_requests, mock_telemetry_client):
    from mlflow.genai import enable_git_model_versioning

    with enable_git_model_versioning():
        pass

    mock_telemetry_client.flush()
    validate_telemetry_record(mock_telemetry_client, mock_requests, GitModelVersioningEvent.name)


@pytest.mark.parametrize(
    ("model_uri", "expected_provider", "litellm_available", "use_native_provider"),
    [
        ("databricks:/llama-3.1-70b", "databricks", True, False),
        ("openai:/gpt-4o-mini", "openai", True, False),
        ("endpoints:/my-endpoint", "endpoints", True, False),
        ("anthropic:/claude-3-opus", "anthropic", True, False),
    ],
)
def test_invoke_custom_judge_model(
    mock_requests,
    mock_telemetry_client: TelemetryClient,
    model_uri,
    expected_provider,
    litellm_available,
    use_native_provider,
):
    from mlflow.genai.judges.utils import invoke_judge_model
    from mlflow.utils.rest_utils import MlflowHostCreds

    mock_response = json.dumps({"result": 0.8, "rationale": "Test rationale"})

    # Mock Databricks credentials for databricks:// URIs
    mock_creds = MlflowHostCreds(host="https://test.databricks.com", token="test-token")

    with (
        mock.patch(
            "mlflow.genai.judges.utils._is_litellm_available", return_value=litellm_available
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds", return_value=mock_creds
        ),
    ):
        if use_native_provider:
            with (
                mock.patch.object(
                    __import__(
                        "mlflow.metrics.genai.model_utils", fromlist=["score_model_on_payload"]
                    ),
                    "score_model_on_payload",
                    return_value=mock_response,
                ),
                mock.patch.object(
                    __import__("mlflow.metrics.genai.model_utils", fromlist=["get_endpoint_type"]),
                    "get_endpoint_type",
                    return_value="llm/v1/chat",
                ),
            ):
                invoke_judge_model(
                    model_uri=model_uri,
                    prompt="Test prompt",
                    assessment_name="test_assessment",
                )
        else:
            with (
                mock.patch(
                    "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm_and_handle_tools",
                    return_value=(mock_response, 10),
                ),
                mock.patch(
                    "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._invoke_databricks_serving_endpoint"
                ) as mock_databricks,
            ):
                # For databricks provider, mock the databricks model invocation
                if expected_provider in ["databricks", "endpoints"]:
                    from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
                        InvokeDatabricksModelOutput,
                    )

                    mock_databricks.return_value = InvokeDatabricksModelOutput(
                        response=mock_response,
                        request_id="test-request-id",
                        num_prompt_tokens=10,
                        num_completion_tokens=20,
                    )

                invoke_judge_model(
                    model_uri=model_uri,
                    prompt="Test prompt",
                    assessment_name="test_assessment",
                )

        expected_params = {"model_provider": expected_provider}
        validate_telemetry_record(
            mock_telemetry_client,
            mock_requests,
            InvokeCustomJudgeModelEvent.name,
            expected_params,
        )


def test_make_judge(mock_requests, mock_telemetry_client: TelemetryClient):
    make_judge(
        name="test_judge",
        instructions="Evaluate the {{ inputs }} and {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=str,
    )
    expected_params = {"model_provider": "openai"}
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, MakeJudgeEvent.name, expected_params
    )

    make_judge(
        name="test_judge",
        instructions="Evaluate the {{ inputs }} and {{ outputs }}",
        feedback_value_type=str,
    )
    expected_params = {"model_provider": None}
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, MakeJudgeEvent.name, expected_params
    )


def test_align_judge(mock_requests, mock_telemetry_client: TelemetryClient):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate the {{ inputs }} and {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=str,
    )

    traces = [
        mock.MagicMock(spec=Trace),
        mock.MagicMock(spec=Trace),
    ]

    class MockOptimizer(AlignmentOptimizer):
        def align(self, judge, traces):
            return judge

    custom_optimizer = MockOptimizer()
    judge.align(traces, optimizer=custom_optimizer)

    expected_params = {"trace_count": 2, "optimizer_type": "MockOptimizer"}
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, AlignJudgeEvent.name, expected_params
    )


def test_autologging(mock_requests, mock_telemetry_client: TelemetryClient):
    try:
        mlflow.openai.autolog()

        mlflow.autolog()
        mock_telemetry_client.flush()
        data = [record["data"] for record in mock_requests]
        params = [event["params"] for event in data if event["event_name"] == AutologgingEvent.name]
        assert (
            json.dumps({"flavor": mlflow.openai.FLAVOR_NAME, "log_traces": True, "disable": False})
            in params
        )
        assert json.dumps({"flavor": "all", "log_traces": True, "disable": False}) in params
    finally:
        mlflow.autolog(disable=True)


def test_load_prompt(mock_requests, mock_telemetry_client: TelemetryClient):
    # Register a prompt first
    prompt = mlflow.genai.register_prompt(
        name="test_prompt",
        template="Hello {{name}}",
    )
    mock_telemetry_client.flush()

    # Set an alias for testing
    mlflow.genai.set_prompt_alias(name="test_prompt", version=prompt.version, alias="production")

    # Test load_prompt with version (no alias)
    mlflow.genai.load_prompt(name_or_uri="test_prompt", version=prompt.version)
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, LoadPromptEvent.name, {"uses_alias": False}
    )

    # Test load_prompt with URI and version (no alias)
    mlflow.genai.load_prompt(name_or_uri=f"prompts:/test_prompt/{prompt.version}")
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, LoadPromptEvent.name, {"uses_alias": False}
    )

    # Test load_prompt with alias
    mlflow.genai.load_prompt(name_or_uri="prompts:/test_prompt@production")
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, LoadPromptEvent.name, {"uses_alias": True}
    )

    # Test load_prompt with @latest (special alias)
    mlflow.genai.load_prompt(name_or_uri="prompts:/test_prompt@latest")
    validate_telemetry_record(
        mock_telemetry_client, mock_requests, LoadPromptEvent.name, {"uses_alias": True}
    )


def test_scorer_call_direct(mock_requests, mock_telemetry_client: TelemetryClient):
    @scorer
    def custom_scorer(outputs) -> bool:
        return len(outputs) > 0

    result = custom_scorer(outputs="test output")
    assert result is True

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "UserDefinedScorer",
            "scorer_kind": "decorator",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": False,
        },
    )

    safety_scorer = Safety()

    mock_feedback = Feedback(
        name="test_feedback",
        value="yes",
        rationale="Test rationale",
    )

    with mock.patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=mock_feedback,
    ):
        safety_scorer(outputs="test output")

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "Safety",
            "scorer_kind": "builtin",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": False,
        },
    )

    mock_requests.clear()

    guidelines_scorer = Guidelines(guidelines="The response must be in English")
    with mock.patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=mock_feedback,
    ):
        guidelines_scorer(
            inputs={"question": "What is MLflow?"}, outputs="MLflow is an ML platform"
        )

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "Guidelines",
            "scorer_kind": "guidelines",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": False,
        },
    )

    mock_requests.clear()

    class CustomClassScorer(Scorer):
        name: str = "custom_class"

        def __call__(self, *, outputs) -> bool:
            return len(outputs) > 0

    custom_class_scorer = CustomClassScorer()
    result = custom_class_scorer(outputs="test output")
    assert result is True

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "UserDefinedScorer",
            "scorer_kind": "class",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": False,
        },
    )


def test_scorer_call_from_genai_evaluate(mock_requests, mock_telemetry_client: TelemetryClient):
    @scorer
    def simple_length_checker(outputs) -> bool:
        return len(outputs) > 0

    session_judge = make_judge(
        name="conversation_quality",
        instructions="Evaluate if the {{ conversation }} is engaging and coherent",
        model="openai:/gpt-4",
    )

    # Create traces with session metadata for session-level scorer testing
    @mlflow.trace(span_type=mlflow.entities.SpanType.CHAT_MODEL)
    def model(question, session_id):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
        return f"Answer to: {question}"

    model("What is MLflow?", session_id="test_session")
    trace_1 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    model("How does MLflow work?", session_id="test_session")
    trace_2 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    test_data = pd.DataFrame(
        [
            {
                "trace": trace_1,
            },
            {
                "trace": trace_2,
            },
        ]
    )

    mock_feedback = Feedback(
        name="test_feedback",
        value="yes",
        rationale="Test",
    )

    with mock.patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=mock_feedback,
    ):
        mlflow.genai.evaluate(data=test_data, scorers=[simple_length_checker, session_judge])

    mock_telemetry_client.flush()

    scorer_call_events = [
        record for record in mock_requests if record["data"]["event_name"] == ScorerCallEvent.name
    ]

    # Should have 3 events: 2 response-level calls (one per trace)
    # + 1 session-level call (one per session)
    assert len(scorer_call_events) == 3

    event_params = [json.loads(event["data"]["params"]) for event in scorer_call_events]

    # Validate response-level scorer was called twice (once per trace)
    response_level_events = [
        params
        for params in event_params
        if params["scorer_class"] == "UserDefinedScorer"
        and params["scorer_kind"] == "decorator"
        and params["is_session_level_scorer"] is False
        and params["callsite"] == "genai.evaluate"
        and params["has_feedback_error"] is False
    ]
    assert len(response_level_events) == 2

    # Validate session-level scorer was called once (once per session)
    session_level_events = [
        params
        for params in event_params
        if params["scorer_class"] == "UserDefinedScorer"
        and params["scorer_kind"] == "instructions"
        and params["is_session_level_scorer"] is True
        and params["callsite"] == "genai.evaluate"
        and params["has_feedback_error"] is False
    ]
    assert len(session_level_events) == 1

    mock_requests.clear()


def test_scorer_call_tracks_feedback_errors(mock_requests, mock_telemetry_client: TelemetryClient):
    error_judge = make_judge(
        name="quality_judge",
        instructions="Evaluate if {{ outputs }} is high quality",
        model="openai:/gpt-4",
    )

    error_feedback = Feedback(
        name="quality_judge",
        error="Model invocation failed",
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE, source_id="openai:/gpt-4"
        ),
    )
    with mock.patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=error_feedback,
    ):
        result = error_judge(outputs="test output")
        assert result.error is not None

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "UserDefinedScorer",
            "scorer_kind": "instructions",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": True,
        },
    )

    mock_requests.clear()

    # Test Scorer returns list of Feedback with mixed errors
    @scorer
    def multi_feedback_scorer(outputs) -> list[Feedback]:
        return [
            Feedback(name="feedback1", value=1.0),
            Feedback(name="feedback2", error=ValueError("Error in feedback 2")),
            Feedback(name="feedback3", value=0.5),
        ]

    multi_feedback_scorer(outputs="test")
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "UserDefinedScorer",
            "scorer_kind": "decorator",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": True,
        },
    )

    mock_requests.clear()

    # Test Scorer returns primitive type (no Feedback error possible)
    @scorer
    def primitive_scorer(outputs) -> bool:
        return True

    primitive_scorer(outputs="test")
    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "UserDefinedScorer",
            "scorer_kind": "decorator",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": False,
        },
    )


def test_scorer_call_wrapped_builtin_scorer_direct(
    mock_requests, mock_telemetry_client: TelemetryClient
):
    completeness_scorer = Completeness()

    mock_feedback = Feedback(
        name="completeness",
        value="yes",
        rationale="Test rationale",
    )

    with mock.patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=mock_feedback,
    ):
        completeness_scorer(inputs={"question": "What is MLflow?"}, outputs="MLflow is a platform")

    mock_telemetry_client.flush()

    # Verify exactly 1 scorer_call event was created
    # (only top-level Completeness, not nested InstructionsJudge)
    scorer_call_events = [
        record for record in mock_requests if record["data"]["event_name"] == ScorerCallEvent.name
    ]
    assert len(scorer_call_events) == 1, (
        f"Expected 1 scorer call event for Completeness scorer (nested calls should be skipped), "
        f"got {len(scorer_call_events)}"
    )

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "Completeness",
            "scorer_kind": "builtin",
            "is_session_level_scorer": False,
            "callsite": "direct_scorer_call",
            "has_feedback_error": False,
        },
    )


def test_scorer_call_wrapped_builtin_scorer_from_genai_evaluate(
    mock_requests, mock_telemetry_client: TelemetryClient
):
    user_frustration_scorer = UserFrustration()

    @mlflow.trace(span_type=mlflow.entities.SpanType.CHAT_MODEL)
    def model(question, session_id):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
        return f"Answer to: {question}"

    model("What is MLflow?", session_id="test_session")
    trace_1 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    model("How does MLflow work?", session_id="test_session")
    trace_2 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    test_data = pd.DataFrame(
        [
            {"trace": trace_1},
            {"trace": trace_2},
        ]
    )

    mock_feedback = Feedback(
        name="user_frustration",
        value="no",
        rationale="Test rationale",
    )

    with mock.patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=mock_feedback,
    ):
        mlflow.genai.evaluate(data=test_data, scorers=[user_frustration_scorer])

    mock_telemetry_client.flush()

    # Verify exactly 1 scorer_call event was created for the session-level scorer
    # (one call at the session level and no nested InstructionsJudge event)
    scorer_call_events = [
        record for record in mock_requests if record["data"]["event_name"] == ScorerCallEvent.name
    ]
    assert len(scorer_call_events) == 1, (
        f"Expected 1 scorer call event for UserFrustration scorer "
        f"(nested calls should be skipped), got {len(scorer_call_events)}"
    )

    validate_telemetry_record(
        mock_telemetry_client,
        mock_requests,
        ScorerCallEvent.name,
        {
            "scorer_class": "UserFrustration",
            "scorer_kind": "builtin",
            "is_session_level_scorer": True,
            "callsite": "genai.evaluate",
            "has_feedback_error": False,
        },
    )
