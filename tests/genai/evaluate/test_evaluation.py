import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest import mock
from unittest.mock import ANY, MagicMock

import pandas as pd
import pytest

import mlflow
from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import EvaluationDataset, create_dataset
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.scorers.base import scorer
from mlflow.genai.scorers.builtin_scorers import RelevanceToQuery
from mlflow.genai.simulators import ConversationSimulator
from mlflow.server import handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores
from mlflow.tracing.constant import AssessmentMetadataKey, TraceMetadataKey

from tests.helper_functions import get_safe_port
from tests.tracing.helper import get_traces
from tests.tracking.integration_test_utils import ServerThread

_DUMMY_CHAT_RESPONSE = {
    "id": "1",
    "object": "text_completion",
    "created": "2021-10-01T00:00:00.000000Z",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "content": "This is a response",
                "role": "assistant",
            },
            "finish_reason": "length",
        }
    ],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    },
}


class TestModel:
    def predict(self, question: str) -> str:
        return "I don't know"


@scorer
def exact_match(outputs, expectations):
    return outputs == expectations["expected_response"]


@scorer
def is_concise(outputs, expectations):
    return len(outputs) <= expectations["max_length"]


@scorer
def relevance(inputs, outputs):
    return Feedback(
        name="relevance",
        value="yes",
        rationale="The response is relevant to the question",
        source=AssessmentSource(source_id="gpt", source_type="LLM_JUDGE"),
    )


@scorer
@mlflow.trace(span_type=SpanType.EVALUATOR)
def has_trace(trace):
    return trace is not None


def _validate_assessments(traces):
    """Validate assessments are added to the traces"""
    for trace in traces:
        assert len(trace.info.assessments) == 6, (
            f"Expected 6 assessments, got {len(trace.info.assessments)}"
            f"Assessments: {[a.name for a in trace.info.assessments]}"
        )  # 2 expectations + 4 feedbacks
        assessments = {a.name: a for a in trace.info.assessments}
        a_exact_match = assessments["exact_match"]
        assert isinstance(a_exact_match, Feedback)
        assert a_exact_match.trace_id == trace.info.trace_id
        assert isinstance(a_exact_match.value, bool)
        assert a_exact_match.source.source_type == AssessmentSourceType.CODE
        # Scorer name is used as source_id
        assert a_exact_match.source.source_id == "exact_match"
        assert a_exact_match.metadata[AssessmentMetadataKey.SOURCE_RUN_ID] is not None

        a_is_concise = assessments["is_concise"]
        assert isinstance(a_is_concise, Feedback)
        assert isinstance(a_is_concise.value, bool)
        assert a_is_concise.metadata[AssessmentMetadataKey.SOURCE_RUN_ID] is not None

        a_has_trace = assessments["has_trace"]
        assert isinstance(a_has_trace, Feedback)
        assert a_has_trace.value is True
        assert a_has_trace.metadata[AssessmentMetadataKey.SOURCE_RUN_ID] is not None

        a_relevance = assessments["relevance"]
        assert isinstance(a_relevance, Feedback)
        assert a_relevance.value == "yes"
        assert a_relevance.source.source_id == "gpt"
        assert a_relevance.source.source_type == "LLM_JUDGE"
        assert a_relevance.rationale == "The response is relevant to the question"
        assert a_relevance.metadata[AssessmentMetadataKey.SOURCE_RUN_ID] is not None

        a_expected_response = assessments["expected_response"]
        assert isinstance(a_expected_response, Expectation)
        assert isinstance(a_expected_response.value, str)
        assert a_expected_response.source.source_type == AssessmentSourceType.HUMAN
        assert a_expected_response.source.source_id is not None

        a_max_length = assessments["max_length"]
        assert isinstance(a_max_length, Expectation)
        assert isinstance(a_max_length.value, (int, float))
        assert a_max_length.source.source_type == AssessmentSourceType.HUMAN


def _validate_eval_result_df(result: EvaluationResult):
    search_traces_df = mlflow.search_traces(run_id=result.run_id)
    assert result.result_df is not None
    assert len(result.result_df) == len(search_traces_df)
    assert set(result.result_df.columns) >= set(search_traces_df.columns)

    actual = result.result_df.sort_values(by="trace_id").reset_index(drop=True)
    expected = search_traces_df.sort_values(by="trace_id").reset_index(drop=True)
    for i in range(len(actual)):
        assert actual.iloc[i].trace_id == expected.iloc[i].trace_id
        assert actual.iloc[i].spans == expected.iloc[i].spans
        assert actual.iloc[i].assessments == expected.iloc[i].assessments
        assert actual.iloc[i]["exact_match/value"] is not None
        assert actual.iloc[i]["is_concise/value"] is not None
        assert actual.iloc[i]["relevance/value"] is not None
        assert actual.iloc[i]["has_trace/value"] is not None
        assert actual.iloc[i]["expected_response/value"] is not None
        assert actual.iloc[i]["max_length/value"] is not None

    # backwards compatibility
    assert len(result.tables["eval_results"]) == len(result.result_df)


@dataclass
class ServerConfig:
    host_type: Literal["local", "remote", "databricks"]
    backend_type: Literal["file", "sqlalchemy"] | None = None


# Test with different server configurations
# 1. local file backend
# 2. local sqlalchemy backend
# 3. remote server running on file backend
# 4. remote server running on sqlalchemy backend
@pytest.fixture(
    params=[
        ServerConfig(host_type="local", backend_type="file"),
        ServerConfig(host_type="local", backend_type="sqlalchemy"),
        ServerConfig(host_type="remote", backend_type="file"),
        ServerConfig(host_type="remote", backend_type="sqlalchemy"),
    ],
    ids=["local_file", "local_sqlalchemy", "remote_file", "remote_sqlalchemy"],
)
def server_config(request, tmp_path: Path, db_uri: str):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    config = request.param

    match config.backend_type:
        case "file":
            backend_uri = tmp_path.joinpath("file").as_uri()
        case "sqlalchemy":
            backend_uri = db_uri

    match config.host_type:
        case "local":
            mlflow.set_tracking_uri(backend_uri)
            yield config

        case "remote":
            # Force-reset backend stores before each test.
            handlers._tracking_store = None
            handlers._model_registry_store = None
            initialize_backend_stores(backend_uri, default_artifact_root=tmp_path.as_uri())

            with ServerThread(app, get_safe_port()) as url:
                mlflow.set_tracking_uri(url)
                yield config


def test_evaluate_with_static_dataset(server_config):
    data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "MLflow is a tool for ML",
            "expectations": {
                "expected_response": "MLflow is a tool for ML",
                "max_length": 100,
            },
        },
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "Spark is a fast data processing engine",
            "expectations": {
                "expected_response": "Spark is a fast data processing engine",
                "max_length": 1,
            },
        },
    ]

    result = mlflow.genai.evaluate(
        data=data,
        scorers=[exact_match, is_concise, relevance, has_trace],
    )

    # OSS evaluator doesn't support metrics aggregation yet.
    metrics = result.metrics
    assert metrics["exact_match/mean"] == 1.0
    assert metrics["is_concise/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    # Exact number of traces should be generated
    traces = get_traces()
    assert len(traces) == len(data)

    # Traces should be associated with the eval run
    traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
    assert len(traces) == len(data)

    # Re-order traces to match with the order of the input data
    traces = sorted(traces, key=lambda t: t.data.spans[0].inputs["question"])

    for i in range(len(traces)):
        assert len(traces[i].data.spans) == 1
        span = traces[i].data.spans[0]
        assert span.name == "root_span"
        assert span.inputs == data[i]["inputs"]
        assert span.outputs == data[i]["outputs"]

    _validate_assessments(traces)
    _validate_eval_result_df(result)

    # Dataset input should be logged to the run
    run = mlflow.get_run(result.run_id)
    assert len(run.inputs.dataset_inputs) == 1
    assert run.inputs.dataset_inputs[0].dataset.name == "dataset"
    assert run.inputs.dataset_inputs[0].dataset.source_type == "code"


@pytest.mark.parametrize("is_predict_fn_traced", [True, False])
def test_evaluate_with_predict_fn(is_predict_fn_traced, server_config):
    model_id = mlflow.set_active_model(name="test-model-id").model_id

    data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {
                "expected_response": "MLflow is a tool for ML",
                "max_length": 100,
            },
        },
        {
            "inputs": {"question": "What is Spark?"},
            "expectations": {
                "expected_response": "Spark is a fast data processing engine",
                "max_length": 1,
            },
        },
    ]
    model = TestModel()
    predict_fn = mlflow.trace(model.predict) if is_predict_fn_traced else model.predict

    result = mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=data,
        scorers=[exact_match, is_concise, relevance, has_trace],
        model_id=model_id,
    )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 0.0
    assert metrics["is_concise/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    # Metrics should be logged to the model ID as well
    model = mlflow.get_logged_model(model_id)
    assert metrics == {m.key: m.value for m in model.metrics}

    # Exact number of traces should be generated
    traces = get_traces()
    assert len(traces) == len(data)

    # Traces should be associated with the eval run
    traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
    assert len(traces) == len(data)

    # Re-order traces to match with the order of the input data
    traces = sorted(traces, key=lambda t: t.data.spans[0].inputs["question"])

    # Check if the model_id is set in the traces
    assert traces[0].info.trace_metadata[TraceMetadataKey.MODEL_ID] == model_id
    assert traces[1].info.trace_metadata[TraceMetadataKey.MODEL_ID] == model_id

    # Validate assessments are added to the traces
    for i in range(len(traces)):
        assert len(traces[i].data.spans) == 1
        span = traces[i].data.spans[0]
        assert span.name == "predict"
        assert span.inputs == data[i]["inputs"]
        assert span.outputs == "I don't know"

    _validate_assessments(traces)
    _validate_eval_result_df(result)


@pytest.mark.parametrize("return_type", ["pandas", "list"])
def test_evaluate_with_traces(monkeypatch: pytest.MonkeyPatch, server_config, return_type):
    questions = ["What is MLflow?", "What is Spark?"]

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(question: str) -> str:
        return TestModel().predict(question)

    predict(questions[0])
    trace_id = mlflow.get_last_active_trace_id()
    mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_response",
        value="MLflow is a tool for ML",
        source=AssessmentSource(source_id="me", source_type="HUMAN"),
    )
    mlflow.log_expectation(
        trace_id=trace_id,
        name="max_length",
        value=100,
        source=AssessmentSource(source_id="me", source_type="HUMAN"),
    )
    predict(questions[1])
    trace_id = mlflow.get_last_active_trace_id()
    mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_response",
        value="Spark is a fast data processing engine",
        source=AssessmentSource(source_id="me", source_type="HUMAN"),
    )
    mlflow.log_expectation(
        trace_id=trace_id,
        name="max_length",
        value=1,
        source=AssessmentSource(source_id="me", source_type="HUMAN"),
    )

    data = mlflow.search_traces(return_type=return_type)
    assert len(data) == len(questions)

    result = mlflow.genai.evaluate(
        data=data,
        scorers=[exact_match, is_concise, relevance, has_trace],
    )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 0.0
    assert metrics["is_concise/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    if server_config.backend_type == "sqlalchemy":
        # Assessments should be added to the traces in-place and no new trace should be created
        traces = get_traces()
        assert len(traces) == len(questions)
    else:
        # File store doesn't support trace linking, so each trace will be cloned to the eval run
        assert len(get_traces()) == len(questions) * 2

    # Traces are associated with the eval run
    traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
    assert len(traces) == len(questions)

    # Re-order traces to match with the order of the input data
    traces = sorted(traces, key=lambda t: t.data.spans[0].inputs["question"])

    # Validate assessments are added to the traces
    _validate_assessments(traces)
    _validate_eval_result_df(result)


def test_evaluate_with_managed_dataset(is_in_databricks):
    if is_in_databricks:
        # Databricks path: Use managed dataset with mocks
        class MockDatasetClient:
            def __init__(self):
                # dataset_id -> list of records
                self.records = {}

            def create_dataset(self, uc_table_name: str, experiment_ids: list[str]):
                from databricks.agents.datasets import Dataset

                dataset = Dataset(
                    dataset_id=str(uuid.uuid4()),
                    name=uc_table_name,
                    digest=None,
                    source_type="databricks-uc-table",
                )
                self.records[dataset.dataset_id] = []
                return dataset

            def list_dataset_records(self, dataset_id: str):
                return self.records[dataset_id]

            def batch_create_dataset_records(self, name: str, dataset_id: str, records):
                self.records[dataset_id].extend(records)

            def upsert_dataset_record_expectations(
                self, name: str, dataset_id: str, record_id: str, expectations: list[dict[str, Any]]
            ):
                for record in self.records[dataset_id]:
                    if record.id == record_id:
                        record.expectations.update(expectations)

            def sync_dataset_to_uc(self, dataset_id: str, uc_table_name: str):
                pass

        mock_client = MockDatasetClient()
        with (
            mock.patch("databricks.rag_eval.datasets.api._get_client", return_value=mock_client),
            mock.patch(
                "databricks.rag_eval.datasets.entities._get_client", return_value=mock_client
            ),
            mock.patch("mlflow.genai.datasets.is_databricks_uri", return_value=True),
        ):
            dataset = create_dataset(
                uc_table_name="mlflow.managed.dataset", experiment_id="exp-123"
            )
            dataset.merge_records(
                [
                    {
                        "inputs": {"question": "What is MLflow?"},
                        "expectations": {
                            "expected_response": "MLflow is a tool for ML",
                            "max_length": 100,
                        },
                    },
                    {
                        "inputs": {"question": "What is Spark?"},
                        "expectations": {
                            "expected_response": "Spark is a fast data processing engine",
                            "max_length": 1,
                        },
                    },
                ]
            )

            result = mlflow.genai.evaluate(
                data=dataset,
                predict_fn=TestModel().predict,
                scorers=[exact_match, is_concise, relevance, has_trace],
            )
    else:
        dataset = create_dataset(
            name="eval_test_dataset", tags={"source": "test", "version": "1.0"}
        )
        dataset.merge_records(
            [
                {
                    "inputs": {"question": "What is MLflow?"},
                    "expectations": {
                        "expected_response": "MLflow is a tool for ML",
                        "max_length": 100,
                    },
                },
                {
                    "inputs": {"question": "What is Spark?"},
                    "expectations": {
                        "expected_response": "Spark is a fast data processing engine",
                        "max_length": 1,
                    },
                },
            ]
        )

        result = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=TestModel().predict,
            scorers=[exact_match, is_concise, relevance, has_trace],
        )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 0.0
    assert metrics["is_concise/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    run = mlflow.get_run(result.run_id)
    # Dataset metadata should be added to the run
    assert len(run.inputs.dataset_inputs) == 1
    assert run.inputs.dataset_inputs[0].dataset.name == dataset.name
    assert run.inputs.dataset_inputs[0].dataset.digest == dataset.digest
    # Check for the correct source_type based on whether we're in Databricks or OSS
    expected_source_type = (
        "databricks-uc-table" if is_in_databricks else "mlflow_evaluation_dataset"
    )
    assert run.inputs.dataset_inputs[0].dataset.source_type == expected_source_type

    # Traces are associated with the eval run
    traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
    assert len(traces) == 2

    _validate_assessments(traces)
    _validate_eval_result_df(result)


def test_evaluate_with_managed_dataset_from_searched_traces():
    for i in range(3):
        with mlflow.start_span(name=f"qa_span_{i}") as span:
            question = f"What is item {i}?"
            span.set_inputs({"question": question})
            span.set_outputs({"answer": f"Item {i} is something"})

            mlflow.log_expectation(
                trace_id=span.trace_id,
                name="expected_response",
                value=f"Item {i} is a detailed answer",
            )
            mlflow.log_expectation(
                trace_id=span.trace_id,
                name="max_length",
                value=50 if i % 2 == 0 else 10,
            )

    traces_df = mlflow.search_traces()

    dataset = create_dataset(
        name="traces_eval_dataset", tags={"source": "traces", "evaluation": "test"}
    )
    dataset.merge_records(traces_df)

    result = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=TestModel().predict,
        scorers=[exact_match, is_concise, has_trace],
    )

    metrics = result.metrics
    assert "exact_match/mean" in metrics
    assert "is_concise/mean" in metrics
    assert "has_trace/mean" in metrics
    assert metrics["has_trace/mean"] == 1.0


def test_model_from_deployment_endpoint(is_in_databricks):
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get_deploy_client:
        mock_client = mock_get_deploy_client.return_value
        mock_client.predict.return_value = _DUMMY_CHAT_RESPONSE

        data = [
            {
                "inputs": {
                    "messages": [
                        {"content": "You are a helpful assistant.", "role": "system"},
                        {"content": "What is Spark?", "role": "user"},
                    ],
                    "max_tokens": 10,
                }
            },
            {
                "inputs": {
                    "messages": [
                        {"content": "What is MLflow?", "role": "user"},
                    ]
                }
            },
        ]
        predict_fn = mlflow.genai.to_predict_fn("endpoints:/chat")
        result = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=[has_trace],
        )

        databricks_options = {"databricks_options": {"return_trace": True}}
        mock_client.predict.assert_has_calls(
            [
                # Test call to check if the function is traced or not
                mock.call(endpoint="chat", inputs={**data[0]["inputs"], **databricks_options}),
                # First evaluation call
                mock.call(endpoint="chat", inputs={**data[0]["inputs"], **databricks_options}),
                # Second evaluation call
                mock.call(endpoint="chat", inputs={**data[1]["inputs"], **databricks_options}),
            ],
            any_order=True,
        )

        # Validate traces
        traces = mlflow.search_traces(run_id=result.run_id, return_type="list")

        assert len(traces) == 2
        spans = traces[0].data.spans
        assert len(spans) == 1
        assert spans[0].name == "predict"
        # Eval harness runs prediction in parallel, so the order is not deterministic
        assert spans[0].inputs in (data[0]["inputs"], data[1]["inputs"])
        assert spans[0].outputs == _DUMMY_CHAT_RESPONSE


def test_missing_scorers_argument():
    with pytest.raises(TypeError, match=r"evaluate\(\) missing 1 required positional"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}])


def test_empty_scorers_allowed():
    mock_result = EvaluationResult(run_id="test-run", metrics={}, result_df=pd.DataFrame())

    data = [{"inputs": {"question": "What is MLflow?"}, "outputs": "MLflow is an ML platform"}]

    with mock.patch("mlflow.genai.evaluation.base._run_harness") as mock_evaluate_oss:
        mock_evaluate_oss.return_value = (mock_result, {})
        result = mlflow.genai.evaluate(data=data, scorers=[])

    assert result is mock_result
    mock_evaluate_oss.assert_called_once()


@pytest.mark.parametrize("pass_full_dataframe", [True, False])
def test_trace_input_can_contain_string_input(pass_full_dataframe, is_in_databricks):
    """
    The `inputs` column must be a dictionary when a static dataset is provided.
    However, when a trace is provided, it doesn't need to be validated and the
    harness can handle it nicely.
    """
    with mlflow.start_span() as span:
        span.set_inputs("What is MLflow?")
        span.set_outputs("MLflow is a tool for ML")

    traces = mlflow.search_traces()
    if not pass_full_dataframe:
        traces = traces[["trace"]]

    # Harness should run without an error
    mlflow.genai.evaluate(data=traces, scorers=[RelevanceToQuery()])


def test_max_workers_env_var(monkeypatch):
    def _validate_max_workers(expected_max_workers):
        with mock.patch(
            "mlflow.genai.evaluation.harness.ThreadPoolExecutor", wraps=ThreadPoolExecutor
        ) as mock_executor:
            mlflow.genai.evaluate(
                data=[
                    {
                        "inputs": {"question": "What is MLflow?"},
                        "outputs": "MLflow is a tool for ML",
                    }
                ],
                scorers=[RelevanceToQuery()],
            )
            # ThreadPoolExecutor is called twice in OSS (harness + scorers)
            first_call = mock_executor.call_args_list[0]
            assert first_call[1]["max_workers"] == expected_max_workers

    # default workers is 10
    _validate_max_workers(10)

    # override workers with env var
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "20")
    _validate_max_workers(20)

    # legacy env var for backward compatibility
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)
    monkeypatch.setenv("RAG_EVAL_MAX_WORKERS", "30")
    _validate_max_workers(30)


def test_dataset_name_is_logged_correctly(is_in_databricks):
    data = pd.DataFrame(
        {
            "inputs": [{"question": "What is MLflow?"}],
            "outputs": ["MLflow is a tool for ML"],
        }
    )

    with mlflow.start_run() as run:
        mlflow.genai.evaluate(
            data=data,
            scorers=[RelevanceToQuery()],
        )

    if not is_in_databricks:
        run_data = mlflow.get_run(run.info.run_id)
        assert run_data.inputs is not None
        assert run_data.inputs.dataset_inputs is not None
        assert len(run_data.inputs.dataset_inputs) > 0

        dataset_input = run_data.inputs.dataset_inputs[0]
        dataset = dataset_input.dataset
        assert dataset.name == "dataset"


def test_evaluate_with_dataset_preserves_name(is_in_databricks):
    from mlflow.entities import Dataset as DatasetEntity

    data = pd.DataFrame(
        {
            "inputs": [{"question": "What is MLflow?"}],
            "outputs": ["MLflow is a tool for ML"],
        }
    )

    mock_managed_dataset = MagicMock(spec=EvaluationDataset)
    type(mock_managed_dataset).name = mock.PropertyMock(return_value="my_managed_dataset")
    mock_managed_dataset.to_df.return_value = data
    mock_managed_dataset.digest = "test_digest"
    mock_managed_dataset.source = MagicMock()
    mock_managed_dataset.source.to_json.return_value = "{}"
    mock_managed_dataset.source._get_source_type.return_value = "test"
    mock_managed_dataset._to_mlflow_entity.return_value = DatasetEntity(
        name="my_managed_dataset",
        digest="test_digest",
        source_type="test",
        source="{}",
        schema=None,
        profile=None,
    )

    if not is_in_databricks:
        with mlflow.start_run() as run:
            mlflow.genai.evaluate(
                data=data,
                scorers=[RelevanceToQuery()],
            )

        run_data = mlflow.get_run(run.info.run_id)
        dataset_input = run_data.inputs.dataset_inputs[0]
        assert dataset_input.dataset.name == "dataset"

        with mlflow.start_run() as run:
            mlflow.genai.evaluate(
                data=mock_managed_dataset,
                scorers=[RelevanceToQuery()],
            )

        run_data = mlflow.get_run(run.info.run_id)
        dataset_input = run_data.inputs.dataset_inputs[0]
        assert dataset_input.dataset.name == "my_managed_dataset"


def test_evaluate_with_managed_dataset_preserves_name():
    mock_managed_dataset = MagicMock()
    mock_managed_dataset.dataset_id = "d-1234567890abcdef1234567890abcdef"
    mock_managed_dataset.name = "test.evaluation.sample_dataset"
    mock_managed_dataset.digest = "abc123"
    mock_managed_dataset.schema = None
    mock_managed_dataset.profile = None
    mock_managed_dataset.source_type = "databricks-uc-table"
    mock_managed_dataset.create_time = None
    mock_managed_dataset.created_by = None
    mock_managed_dataset.last_update_time = None
    mock_managed_dataset.last_updated_by = None
    mock_managed_dataset.to_df.return_value = pd.DataFrame(
        {
            "inputs": [{"question": "What is MLflow?"}],
            "outputs": ["MLflow is a tool for ML"],
        }
    )

    dataset = EvaluationDataset(mock_managed_dataset)

    with mlflow.start_run() as run:
        mlflow.genai.evaluate(
            data=dataset,
            scorers=[RelevanceToQuery()],
        )

        run_data = mlflow.get_run(run.info.run_id)

        assert run_data.inputs is not None
        assert run_data.inputs.dataset_inputs is not None
        assert len(run_data.inputs.dataset_inputs) > 0

        dataset_input = run_data.inputs.dataset_inputs[0]
        logged_dataset = dataset_input.dataset
        assert logged_dataset.name == "test.evaluation.sample_dataset"


@pytest.mark.parametrize(
    ("tags_data", "expected_calls"),
    [
        # Regular tags
        (
            [
                {"environment": "test", "model_version": "v1.0"},
                {"environment": "production", "team": "data-science"},
            ],
            [
                ("environment", "test"),
                ("model_version", "v1.0"),
                ("environment", "production"),
                ("team", "data-science"),
            ],
        ),
        # Empty tags dict
        (
            [{}, {}],
            [],
        ),
        # None tags (no tags field)
        (
            [None, None],
            [],
        ),
        # Mix of tags and empty/None
        (
            [{"env": "test"}, {}, None],
            [("env", "test")],
        ),
    ],
)
def test_evaluate_with_tags(tags_data, expected_calls):
    data = [
        {
            "inputs": {"question": f"What is question {i}?"},
            "outputs": f"Answer {i}",
            "expectations": {"expected_response": f"Answer {i}"},
            "tags": tags,
        }
        for i, tags in enumerate(tags_data)
    ]

    with mock.patch("mlflow.set_trace_tag") as mock_set_trace_tag:
        mlflow.genai.evaluate(
            data=data,
            scorers=[exact_match],
        )

        # Check that all expected calls were made (order may vary due to parallel execution)
        actual_calls = mock_set_trace_tag.call_args_list
        expected_mock_calls = [
            mock.call(trace_id=ANY, key=key, value=value) for key, value in expected_calls
        ]
        assert len(actual_calls) == len(expected_mock_calls)
        for expected_call in expected_mock_calls:
            assert expected_call in actual_calls


def test_evaluate_with_traces_tags_no_warnings():
    with mlflow.start_span() as span:
        span.set_inputs({"question": "Hello?"})

    traces = mlflow.search_traces()
    with mock.patch("mlflow.tracing.client._logger.warning") as mock_warning:
        mlflow.genai.evaluate(
            data=traces,
            scorers=[has_trace],
        )
        assert not any(
            "immutable and cannot be set on a trace" in call.args[0]
            for call in mock_warning.call_args_list
        )


def test_evaluate_with_tags_error_handling(is_in_databricks):
    data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "MLflow is a tool for ML",
            "expectations": {"expected_response": "MLflow is a tool for ML"},
            "tags": {"invalid_tag": "value"},
        }
    ]

    # Mock set_trace_tag to raise an exception
    with mock.patch("mlflow.set_trace_tag", side_effect=Exception("Tag logging failed")):
        # This should not raise an exception
        result = mlflow.genai.evaluate(
            data=data,
            scorers=[exact_match],
        )

        # Evaluation should still succeed
        assert "exact_match/mean" in result.metrics


def test_evaluate_with_invalid_tags_type():
    data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "MLflow is a tool for ML",
            "expectations": {"expected_response": "MLflow is a tool for ML"},
            "tags": "invalid_tags_string",  # Should be dict
        }
    ]

    with pytest.raises(MlflowException, match="Tags must be a dictionary"):
        mlflow.genai.evaluate(
            data=data,
            scorers=[exact_match],
        )


def test_evaluate_without_inputs_in_eval_dataset():
    answers = [
        "MLflow is an open-source platform for managing ML lifecycle",
        "Apache Spark is a fast data processing engine",
        "I don't know",
    ]
    for answer in answers:
        with mlflow.start_span() as span:
            span.set_outputs(answer)

    trace_df = mlflow.search_traces()
    trace_df["inputs"] = None
    trace_df["expectations"] = pd.Series(
        [{"expected_response": answer, "max_length": 100} for answer in answers]
    )

    result = mlflow.genai.evaluate(
        data=trace_df,
        scorers=[is_concise, exact_match, has_trace],
    )

    assert "is_concise/mean" in result.metrics
    assert "exact_match/mean" in result.metrics
    assert "has_trace/mean" in result.metrics

    @scorer
    def input_exist(inputs):
        if inputs is None:
            return False
        return True

    trace_df["outputs"] = None
    result = mlflow.genai.evaluate(
        data=trace_df,
        scorers=[input_exist],
    )
    assert result.metrics["input_exist/mean"] == 0.0


def test_evaluate_with_only_trace_in_eval_dataset():
    for _ in range(3):
        with mlflow.start_span():
            pass

    trace_df = mlflow.search_traces()
    trace_df = trace_df[["trace"]]

    result = mlflow.genai.evaluate(
        data=trace_df,
        scorers=[has_trace],
    )

    assert result.metrics["has_trace/mean"] == 1.0


@pytest.mark.parametrize("is_enabled", [True, False])
def test_evaluate_with_scorer_tracing(server_config, monkeypatch, is_enabled):
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING", str(is_enabled).lower())

    data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {
                "expected_response": "MLflow is a tool for ML",
                "max_length": 100,
            },
        },
        {
            "inputs": {"question": "What is Spark?"},
            "expectations": {
                "expected_response": "Spark is a fast data processing engine",
                "max_length": 1,
            },
        },
    ]

    result = mlflow.genai.evaluate(
        predict_fn=TestModel().predict,
        data=data,
        scorers=[exact_match, is_concise, relevance, has_trace],
    )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 0.0
    assert metrics["is_concise/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    traces = get_traces()
    if is_enabled:
        assert len(traces) == len(data) * 5  # 1 trace for prediction + 4 scorer traces
    else:
        assert len(traces) == len(data)

    # Traces should be associated with the eval run
    traces = mlflow.search_traces(
        filter_string="tags.`mlflow.eval.requestId` != 'None'",
        run_id=result.run_id,
        return_type="list",
    )
    assert len(traces) == len(data)

    # Each assessment should have a source trace ID
    for trace in traces:
        for a in trace.info.assessments:
            if isinstance(a, Feedback) and is_enabled:
                assert a.metadata[AssessmentMetadataKey.SCORER_TRACE_ID] is not None
                assert a.metadata[AssessmentMetadataKey.SCORER_TRACE_ID] != trace.info.trace_id
            else:
                assert AssessmentMetadataKey.SCORER_TRACE_ID not in a.metadata


@pytest.mark.parametrize("diff_experiment_id", [True, False])
def test_eval_with_traces_log_spans_correctly(diff_experiment_id):
    exp_id = mlflow.set_experiment("traces exp").experiment_id
    with mlflow.start_span() as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is a tool for ML"})
        span.set_attributes({"key": "value"})
        with mlflow.start_span() as child_span:
            child_span.set_inputs("test")

    # set to a different experiment
    if diff_experiment_id:
        mlflow.set_experiment("diff exp")

    # search traces from the original experiment
    trace_df = mlflow.search_traces(locations=[exp_id])

    result = mlflow.genai.evaluate(
        data=trace_df,
        scorers=[has_trace],
    )

    assert result.metrics["has_trace/mean"] == 1.0

    traces = get_traces()
    assert len(traces) == 1
    # copied trace should contain all spans
    assert len(traces[0].data.spans) == 2
    span = traces[0].data.spans[0]
    assert span.get_attribute("key") == "value"
    assert span.inputs == {"question": "What is MLflow?"}
    assert span.outputs == {"answer": "MLflow is a tool for ML"}
    child_span = traces[0].data.spans[1]
    assert child_span.inputs == "test"


def test_evaluate_with_mixed_single_turn_and_multi_turn_scorers(server_config):
    """Test evaluation with a combination of single-turn and multi-turn scorers.

    Validates that:
    - Single-turn scorers are applied to all traces
    - Multi-turn scorers are only applied to the first trace of each session
    """

    # Define a multi-turn scorer that counts conversation turns
    class ConversationLengthScorer(mlflow.genai.Scorer):
        def __init__(self):
            super().__init__(name="conversation_length")

        @property
        def is_session_level_scorer(self) -> bool:
            return True

        def __call__(self, session=None, **kwargs):
            """Return the number of turns in the conversation."""
            return len(session or [])

    # Define a single-turn scorer
    @scorer
    def response_length(outputs) -> int:
        """Return the length of the response."""
        return len(outputs) if isinstance(outputs, str) else 0

    # Create a traced model function
    @mlflow.trace(span_type=SpanType.CHAT_MODEL)
    def model(question, session_id):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
        return f"Answer to: {question}"

    # Generate traces for 2 sessions (3 turns + 2 turns = 5 total traces)
    mlflow.set_experiment("multi_turn_test")
    with mlflow.start_run() as run:
        # Session 1: 3 turns
        for q in ["Q1", "Q2", "Q3"]:
            model(q, session_id="session_1")

        # Session 2: 2 turns
        for q in ["Q4", "Q5"]:
            model(q, session_id="session_2")

        # Get traces for evaluation
        traces = mlflow.search_traces(
            locations=[run.info.experiment_id], filter_string=f'run_id = "{run.info.run_id}"'
        )

        # Evaluate with both single-turn and multi-turn scorers
        result = mlflow.genai.evaluate(
            data=traces, scorers=[response_length, ConversationLengthScorer()]
        )

    # Validate results
    result_df = result.result_df

    # Should have one row per trace
    assert len(result_df) == 5, f"Expected 5 traces, got {len(result_df)}"

    # Single-turn scorer should be applied to all traces
    single_turn_scores = result_df["response_length/value"].notna().sum()
    assert single_turn_scores == 5, (
        f"Expected single-turn scores for all 5 traces, got {single_turn_scores}"
    )

    # Multi-turn scorer should only be applied to first trace of each session (2 total)
    multi_turn_scores = result_df["conversation_length/value"].notna().sum()
    assert multi_turn_scores == 2, (
        f"Expected multi-turn scores for 2 sessions (first trace only), got {multi_turn_scores}"
    )

    # Validate the conversation length values
    # Session 1 should have 3 turns, Session 2 should have 2 turns
    conv_lengths = result_df["conversation_length/value"].dropna().sort_values().tolist()
    assert conv_lengths == [2.0, 3.0], (
        f"Expected conversation lengths [2.0, 3.0], got {conv_lengths}"
    )

    # Validate that all single-turn scores are the same (based on our dummy response)
    response_lengths = result_df["response_length/value"].dropna()
    # All responses should be "Answer to: Qx" format, so lengths should be consistent
    assert all(length > 0 for length in response_lengths), "All response lengths should be positive"


def test_evaluate_with_evaluation_dataset_and_session_level_scorers():
    # Define a session-level scorer
    class ConversationLengthScorer(mlflow.genai.Scorer):
        def __init__(self):
            super().__init__(name="conversation_length")

        @property
        def is_session_level_scorer(self) -> bool:
            return True

        def __call__(self, session=None, **kwargs):
            return len(session or [])

    # Create traces with session metadata (2 traces in session_1, 1 in session_2)
    @mlflow.trace(span_type=mlflow.entities.SpanType.CHAT_MODEL)
    def model(question, session_id):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
        return f"Answer to {question}"

    model("Q1", session_id="session_1")
    trace_1 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    model("Q2", session_id="session_1")
    trace_2 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    model("Q3", session_id="session_2")
    trace_3 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    # Create dataset from traces
    dataset = create_dataset(name="multi_turn_dataset")
    dataset.merge_records([trace_1, trace_2, trace_3])

    # Evaluate with session-level scorer
    result = mlflow.genai.evaluate(data=dataset, scorers=[ConversationLengthScorer()])
    result_df = result.result_df

    # Session-level scorer should produce 2 scores (one per session)
    assert "conversation_length/value" in result_df.columns
    assert result_df["conversation_length/value"].notna().sum() == 2

    # Verify conversation lengths: session_1 has 2 traces, session_2 has 1 trace
    conv_lengths = result_df["conversation_length/value"].dropna().sort_values().tolist()
    assert conv_lengths == [1.0, 2.0]


def test_evaluate_dataset_mixed_traces_with_and_without_sessions():
    class SessionScorer(mlflow.genai.Scorer):
        def __init__(self):
            super().__init__(name="session_length")

        @property
        def is_session_level_scorer(self):
            return True

        def __call__(self, session=None, **kwargs):
            return len(session or [])

    # Create mixed traces
    @mlflow.trace(span_type=mlflow.entities.SpanType.CHAT_MODEL)
    def model_with_session(question, session_id):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
        return "answer"

    @mlflow.trace(span_type=mlflow.entities.SpanType.CHAT_MODEL)
    def model_without_session(question):
        return "answer"

    model_with_session("Q1", "session_1")
    trace_1 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    model_without_session("Q2")
    trace_2 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    model_with_session("Q3", "session_1")
    trace_3 = mlflow.get_trace(mlflow.get_last_active_trace_id())

    # Create dataset and evaluate
    dataset = create_dataset(name="mixed_dataset")
    dataset.merge_records([trace_1, trace_2, trace_3])

    result = mlflow.genai.evaluate(data=dataset, scorers=[SessionScorer()])
    result_df = result.result_df

    # Should have 1 session-level score (for session_1 with 2 traces)
    # The trace without session should not be scored by session-level scorer
    assert result_df["session_length/value"].notna().sum() == 1
    assert result_df["session_length/value"].dropna().iloc[0] == 2.0


def test_max_scorer_workers_env_var(monkeypatch):
    @scorer
    def dummy_scorer_1(outputs):
        return True

    @scorer
    def dummy_scorer_2(outputs):
        return True

    @scorer
    def dummy_scorer_3(outputs):
        return True

    def _validate_scorer_max_workers(expected_max_workers, num_scorers):
        scorers_list = [dummy_scorer_1, dummy_scorer_2, dummy_scorer_3][:num_scorers]
        with mock.patch(
            "mlflow.genai.evaluation.harness.ThreadPoolExecutor", wraps=ThreadPoolExecutor
        ) as mock_executor:
            mlflow.genai.evaluate(
                data=[
                    {
                        "inputs": {"question": "What is MLflow?"},
                        "outputs": "MLflow is a tool for ML",
                    }
                ],
                scorers=scorers_list,
            )
            # ThreadPoolExecutor is called twice: harness loop + scorer loop
            # The second call is for scorers
            scorer_call = mock_executor.call_args_list[1]
            assert scorer_call[1]["max_workers"] == expected_max_workers

    # default scorer workers is 10, but limited by number of scorers (3)
    _validate_scorer_max_workers(expected_max_workers=3, num_scorers=3)

    # override scorer workers with env var (limit to 2)
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "2")
    _validate_scorer_max_workers(expected_max_workers=2, num_scorers=3)

    # when num_scorers < max_scorer_workers, use num_scorers
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "10")
    _validate_scorer_max_workers(expected_max_workers=2, num_scorers=2)

    # set to 1 for sequential execution
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "1")
    _validate_scorer_max_workers(expected_max_workers=1, num_scorers=3)


# ===================== ConversationSimulator Integration Tests =====================


def test_evaluate_with_conversation_simulator_requires_predict_fn():
    simulator = ConversationSimulator(
        test_cases=[{"goal": "Learn about MLflow"}],
        max_turns=2,
    )

    with pytest.raises(MlflowException, match="predict_fn is required"):
        mlflow.genai.evaluate(
            data=simulator,
            scorers=[has_trace],
        )


def test_evaluate_with_conversation_simulator_empty_simulation_error():
    def failing_predict_fn(input: list[dict[str, Any]], **kwargs):
        raise Exception("Simulated failure")

    simulator = ConversationSimulator(
        test_cases=[{"goal": "Learn about MLflow"}],
        max_turns=2,
    )

    with mock.patch(
        "mlflow.genai.simulators.simulator._invoke_model_without_tracing"
    ) as mock_invoke:
        # Simulate a failure that produces no traces
        mock_invoke.side_effect = Exception("LLM call failed")

        with pytest.raises(MlflowException, match="Simulation produced no traces"):
            mlflow.genai.evaluate(
                data=simulator,
                predict_fn=failing_predict_fn,
                scorers=[has_trace],
            )


def test_session_level_evaluation_with_predict_fn_without_simulator():
    class SessionScorer(mlflow.genai.Scorer):
        def __init__(self):
            super().__init__(name="session_scorer")

        @property
        def is_session_level_scorer(self):
            return True

        def __call__(self, session=None, **kwargs):
            return len(session or [])

    data = [
        {"inputs": {"question": "What is MLflow?"}, "outputs": "MLflow is a tool"},
    ]

    with pytest.raises(
        MlflowException,
        match=(
            r"Session-level scorers require traces with session IDs.*"
            r"session_scorer.*"
            r"Either pass a ConversationSimulator to `data` with `predict_fn`"
        ),
    ):
        mlflow.genai.evaluate(
            data=data,
            predict_fn=TestModel().predict,
            scorers=[SessionScorer()],
        )


def test_evaluate_with_conversation_simulator_calls_simulate():
    simulator = ConversationSimulator(
        test_cases=[{"goal": "Learn MLflow"}],
        max_turns=2,
    )

    def mock_predict_fn(input: list[dict[str, Any]], **kwargs):
        return {"output": "Mock response"}

    with mock.patch.object(simulator, "_simulate") as mock_simulate:
        # Return empty list to trigger the "no traces" error
        mock_simulate.return_value = []

        with pytest.raises(MlflowException, match="Simulation produced no traces"):
            mlflow.genai.evaluate(
                data=simulator,
                predict_fn=mock_predict_fn,
                scorers=[has_trace],
            )

        # Verify _simulate was called with predict_fn
        mock_simulate.assert_called_once_with(mock_predict_fn)


def test_evaluate_with_predict_fn_without_traces_creates_minimal_trace(monkeypatch):
    """
    Test that when MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION is True and predict_fn
    doesn't generate traces, a minimal trace is created to avoid NoneType errors.

    This addresses Bug 2: NoneType error when validation is skipped.
    """
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "true")

    # Create a simple predict function that doesn't generate traces
    def simple_predict_fn(summary: str) -> str:
        return summary

    data = [
        {
            "inputs": {"summary": "Test summary"},
            "outputs": "Test summary",  # Include outputs in data
        }
    ]

    @scorer(name="simple_scorer")
    def simple_scorer(inputs, outputs):
        # Simple scorer that always returns 1.0
        return 1.0

    # This should not raise AttributeError: 'NoneType' object has no attribute 'info'
    result = mlflow.genai.evaluate(
        data=data,
        predict_fn=simple_predict_fn,
        scorers=[simple_scorer],
    )

    # Verify the evaluation completed successfully
    assert result is not None
    assert "simple_scorer/mean" in result.metrics
    assert result.metrics["simple_scorer/mean"] == 1.0

    # Verify a trace was created (minimal trace)
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0] is not None
    assert traces[0].info is not None


def test_evaluate_validation_uses_unique_request_id():
    """
    Test that validation check uses a unique request_id to prevent trace ID collisions.

    This addresses Bug 1: Duplicate trace ID error during validation.
    The test verifies that validation and evaluation use different request IDs.
    """

    # Mock predict function that tracks trace context
    @mlflow.trace
    def tracking_predict_fn(summary: str) -> str:
        return summary

    data = [
        {
            "inputs": {"summary": "Test summary"},
            "outputs": "Test summary",
        }
    ]

    @scorer(name="test_scorer")
    def test_scorer(inputs, outputs):
        return 1.0

    # Run evaluation - this will call validation (with one request_id)
    # and then actual evaluation (with a different request_id)
    result = mlflow.genai.evaluate(
        data=data,
        predict_fn=tracking_predict_fn,
        scorers=[test_scorer],
    )

    # Verify evaluation completed successfully
    assert result is not None
    assert "test_scorer/mean" in result.metrics

    # Verify that traces were created
    traces = get_traces()
    assert len(traces) >= 1  # At least the evaluation trace

    # Get all trace IDs and verify they're unique (no duplicates)
    trace_ids = [trace.info.trace_id for trace in traces]
    assert len(trace_ids) == len(set(trace_ids)), "Found duplicate trace IDs"


def test_evaluate_with_auto_traced_function_no_duplicate_traces():
    """
    Test that auto-traced functions (e.g., using OpenAI) don't cause duplicate
    trace IDs during validation and evaluation.

    This simulates the real-world scenario from the bug report where OpenAI
    auto-tracing caused duplicate trace IDs.
    """

    # Simulate an auto-traced function by manually creating traces
    call_count = [0]

    @mlflow.trace
    def auto_traced_predict_fn(summary: str) -> str:
        call_count[0] += 1
        return summary

    data = [
        {
            "inputs": {"summary": "Test summary"},
            "outputs": "Test summary",
        }
    ]

    @scorer(name="test_scorer")
    def test_scorer(inputs, outputs):
        return 1.0

    # This should not raise duplicate key constraint violations
    result = mlflow.genai.evaluate(
        data=data,
        predict_fn=auto_traced_predict_fn,
        scorers=[test_scorer],
    )

    # Verify evaluation completed successfully
    assert result is not None
    assert "test_scorer/mean" in result.metrics

    # Verify the function was called at least twice (validation + evaluation)
    assert call_count[0] >= 2

    # Get all traces and verify they have unique trace IDs
    traces = get_traces()
    trace_ids = [trace.info.trace_id for trace in traces]
    # All trace IDs should be unique (no duplicates)
    assert len(trace_ids) == len(set(trace_ids)), "Found duplicate trace IDs"
