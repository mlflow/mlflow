import uuid
from importlib import import_module
from unittest import mock
from unittest.mock import patch

import pytest
from packaging.version import Version

import mlflow
from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset
from mlflow.genai.scorers.base import scorer
from mlflow.genai.scorers.builtin_scorers import Safety
from mlflow.tracing.constant import TraceMetadataKey

from tests.evaluate.test_evaluation import _DUMMY_CHAT_RESPONSE
from tests.tracing.helper import get_traces

_IS_AGENT_SDK_V1 = Version(import_module("databricks.agents").__version__).major >= 1


class TestModel:
    def predict(self, question: str) -> str:
        return "I don't know"


@scorer
def exact_match(outputs, expectations):
    return outputs == expectations["expected_response"]


@scorer
def max_length(outputs, expectations):
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
def has_trace(trace):
    return trace is not None


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
def test_evaluate_with_static_dataset():
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
        scorers=[exact_match, max_length, relevance, has_trace],
    )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 1.0
    assert metrics["max_length/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    # Exact number of traces should be generated
    traces = get_traces()
    assert len(traces) == len(data)


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
@pytest.mark.parametrize("is_predict_fn_traced", [True, False])
def test_evaluate_with_predict_fn(is_predict_fn_traced):
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
        scorers=[exact_match, max_length, relevance, has_trace],
        model_id=model_id,
    )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 0.0
    assert metrics["max_length/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    # Exact number of traces should be generated
    traces = get_traces()
    assert len(traces) == len(data)

    # Check if the model_id is set in the traces
    assert traces[0].info.trace_metadata[TraceMetadataKey.MODEL_ID] == model_id
    assert traces[1].info.trace_metadata[TraceMetadataKey.MODEL_ID] == model_id


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
@pytest.mark.parametrize("pass_full_dataframe", [True, False])
def test_evaluate_with_traces(pass_full_dataframe):
    questions = ["What is MLflow?", "What is Spark?"]

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(question: str) -> str:
        return TestModel().predict(question)

    for question in questions:
        predict(question)

    data = mlflow.search_traces()
    assert len(data) == len(questions)

    # OSS MLflow backend doesn't support assessment APIs now, so we need to manually add them
    data.iloc[0]["trace"].info.assessments = [
        Expectation(
            name="expected_response",
            trace_id="tr-123",
            value="MLflow is a tool for ML",
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
        Expectation(
            name="max_length",
            trace_id="tr-123",
            value=100,
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
    ]
    data.iloc[1]["trace"].info.assessments = [
        Expectation(
            name="expected_response",
            trace_id="tr-123",
            value="Spark is a fast data processing engine",
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
        Expectation(
            name="max_length",
            trace_id="tr-123",
            value=1,
            source=AssessmentSource(source_id="me", source_type="HUMAN"),
        ),
    ]

    if not pass_full_dataframe:
        data = data[["trace"]]

    # Disable logging traces to MLflow to avoid calling mlflow APIs which need to be mocked
    with mock.patch.dict("os.environ", {"AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED": "false"}):
        result = mlflow.genai.evaluate(
            data=data,
            scorers=[exact_match, max_length, relevance, has_trace],
        )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 0.0
    assert metrics["max_length/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    # Assessments should be added to the traces in-place and no new trace should be created
    assert len(get_traces()) == len(questions)


@pytest.mark.skipif(not _IS_AGENT_SDK_V1, reason="Databricks Agent SDK v1 is required")
def test_evaluate_with_managed_dataset():
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
            self, name: str, dataset_id: str, record_id: str, expectations: list[dict]
        ):
            for record in self.records[dataset_id]:
                if record.id == record_id:
                    record.expectations.update(expectations)

        def sync_dataset_to_uc(self, dataset_id: str, uc_table_name: str):
            pass

    mock_client = MockDatasetClient()
    with (
        mock.patch("databricks.rag_eval.datasets.api._get_client", return_value=mock_client),
        mock.patch("databricks.rag_eval.datasets.entities._get_client", return_value=mock_client),
    ):
        dataset = create_dataset(uc_table_name="mlflow.managed.dataset", experiment_id="exp-123")
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
            scorers=[exact_match, max_length, relevance, has_trace],
        )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 0.0
    assert metrics["max_length/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    run = mlflow.get_run(result.run_id)
    # Dataset metadata should be added to the run
    assert len(run.inputs.dataset_inputs) == 1
    assert run.inputs.dataset_inputs[0].dataset.name == dataset.name
    assert run.inputs.dataset_inputs[0].dataset.digest == dataset.digest
    assert run.inputs.dataset_inputs[0].dataset.source_type == "databricks-uc-table"


@mock.patch("mlflow.deployments.get_deploy_client")
def test_model_from_deployment_endpoint(mock_get_deploy_client):
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
    assert spans[0].attributes["endpoint"] == "endpoints:/chat"
    # Eval harness runs prediction in parallel, so the order is not deterministic
    assert spans[0].inputs in (data[0]["inputs"], data[1]["inputs"])
    assert spans[0].outputs == _DUMMY_CHAT_RESPONSE


@patch("mlflow.get_tracking_uri", return_value="databricks")
def test_no_scorers(mock_get_tracking_uri):
    with pytest.raises(TypeError, match=r"evaluate\(\) missing 1 required positional"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}])

    with pytest.raises(MlflowException, match=r"The `scorers` argument must be a list of"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}], scorers=[])


@pytest.mark.parametrize("pass_full_dataframe", [True, False])
def test_trace_input_can_contain_string_input(pass_full_dataframe):
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
    mlflow.genai.evaluate(data=traces, scorers=[Safety()])
