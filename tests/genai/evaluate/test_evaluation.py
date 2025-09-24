import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest import mock
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

import mlflow
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import EvaluationDataset, create_dataset
from mlflow.genai.scorers.base import scorer
from mlflow.genai.scorers.builtin_scorers import RelevanceToQuery
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.utils.mlflow_tags import MLFLOW_RUN_IS_EVALUATION

from tests.evaluate.test_evaluation import _DUMMY_CHAT_RESPONSE
from tests.tracing.helper import get_traces


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
def has_trace(trace):
    return trace is not None


def _validate_assessments(traces):
    """Validate assessments are added to the traces"""
    for trace in traces:
        assert len(trace.info.assessments) == 6  # 2 expectations + 4 feedbacks
        assessments = {a.name: a for a in trace.info.assessments}
        a_exact_match = assessments["exact_match"]
        assert isinstance(a_exact_match, Feedback)
        assert a_exact_match.trace_id == trace.info.trace_id
        assert isinstance(a_exact_match.value, bool)
        assert a_exact_match.source.source_type == AssessmentSourceType.CODE
        # Scorer name is used as source_id
        assert a_exact_match.source.source_id == "exact_match"

        a_is_concise = assessments["is_concise"]
        assert isinstance(a_is_concise, Feedback)
        assert isinstance(a_is_concise.value, bool)

        a_has_trace = assessments["has_trace"]
        assert isinstance(a_has_trace, Feedback)
        assert a_has_trace.value is True

        a_relevance = assessments["relevance"]
        assert isinstance(a_relevance, Feedback)
        assert a_relevance.value == "yes"
        assert a_relevance.source.source_id == "gpt"
        assert a_relevance.source.source_type == "LLM_JUDGE"
        assert a_relevance.rationale == "The response is relevant to the question"

        a_expected_response = assessments["expected_response"]
        assert isinstance(a_expected_response, Expectation)
        assert isinstance(a_expected_response.value, str)
        assert a_expected_response.source.source_type == AssessmentSourceType.HUMAN
        assert a_expected_response.source.source_id is not None

        a_max_length = assessments["max_length"]
        assert isinstance(a_max_length, Expectation)
        assert isinstance(a_max_length.value, int)
        assert a_max_length.source.source_type == AssessmentSourceType.HUMAN


def test_evaluate_with_static_dataset(is_in_databricks):
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

    # Dataset input should be logged to the run
    run = mlflow.get_run(result.run_id)
    assert len(run.inputs.dataset_inputs) == 1
    assert run.inputs.dataset_inputs[0].dataset.name == "dataset"
    assert run.inputs.dataset_inputs[0].dataset.source_type == "code"

    if not is_in_databricks:
        assert run.data.tags[MLFLOW_RUN_IS_EVALUATION] == "true"


@pytest.mark.parametrize("is_predict_fn_traced", [True, False])
def test_evaluate_with_predict_fn(is_predict_fn_traced, is_in_databricks):
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


@pytest.mark.skip(reason="TODO: OSS MLflow backend doesn't support trace->run linking yet")
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
    def add_assessment_to_trace_json(trace_json: str, assessments: list[Assessment]):
        trace = Trace.from_json(trace_json)
        trace.info.assessments = assessments
        return trace.to_json()

    data.at[0, "trace"] = add_assessment_to_trace_json(
        data.at[0, "trace"],
        [
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
        ],
    )
    data.at[1, "trace"] = add_assessment_to_trace_json(
        data.at[1, "trace"],
        [
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
        ],
    )

    if not pass_full_dataframe:
        data = data[["trace"]]

    # Disable logging traces to MLflow to avoid calling mlflow APIs which need to be mocked
    with mock.patch.dict("os.environ", {"AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED": "false"}):
        result = mlflow.genai.evaluate(
            data=data,
            scorers=[exact_match, is_concise, relevance, has_trace],
        )

    metrics = result.metrics
    assert metrics["exact_match/mean"] == 0.0
    assert metrics["is_concise/mean"] == 0.5
    assert metrics["relevance/mean"] == 1.0
    assert metrics["has_trace/mean"] == 1.0

    # Assessments should be added to the traces in-place and no new trace should be created
    traces = get_traces()
    assert len(traces) == len(questions)

    # Traces are associated with the eval run
    traces = mlflow.search_traces(run_id=result.run_id, return_type="list")
    assert len(traces) == len(questions)

    # Re-order traces to match with the order of the input data
    traces = sorted(traces, key=lambda t: t.data.spans[0].inputs["question"])

    # Validate assessments are added to the traces
    _validate_assessments(traces)


def test_evaluate_with_managed_dataset(is_in_databricks):
    """Test evaluation with both managed (Databricks) and OSS datasets."""
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
            mock.patch(
                "mlflow.genai.datasets.is_databricks_default_tracking_uri", return_value=True
            ),
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


@mock.patch("mlflow.deployments.get_deploy_client")
def test_model_from_deployment_endpoint(mock_get_deploy_client, is_in_databricks):
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


@patch("mlflow.get_tracking_uri", return_value="databricks")
def test_no_scorers(mock_get_tracking_uri):
    with pytest.raises(TypeError, match=r"evaluate\(\) missing 1 required positional"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}])

    with pytest.raises(MlflowException, match=r"The `scorers` argument must be a list of"):
        mlflow.genai.evaluate(data=[{"inputs": "Hello", "outputs": "Hi"}], scorers=[])


@pytest.mark.parametrize("pass_full_dataframe", [True, False])
def test_trace_input_can_contain_string_input(pass_full_dataframe, is_in_databricks):
    """
    The `inputs` column must be a dictionary when a static dataset is provided.
    However, when a trace is provided, it doesn't need to be validated and the
    harness can handle it nicely.
    """
    if not is_in_databricks:
        pytest.skip("OSS genai evaluator doesn't support trace input yet")
    with mlflow.start_span() as span:
        span.set_inputs("What is MLflow?")
        span.set_outputs("MLflow is a tool for ML")

    traces = mlflow.search_traces()
    if not pass_full_dataframe:
        traces = traces[["trace"]]

    # Harness should run without an error
    mlflow.genai.evaluate(data=traces, scorers=[RelevanceToQuery()])


def test_max_workers_env_var(is_in_databricks, monkeypatch):
    harness_module = (
        "databricks.rag_eval.evaluation" if is_in_databricks else "mlflow.genai.evaluation"
    )

    def _validate_max_workers(expected_max_workers):
        with mock.patch(
            f"{harness_module}.harness.ThreadPoolExecutor", wraps=ThreadPoolExecutor
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

    # legacy env var is supported for databricks
    if is_in_databricks:
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
    """Test that tags from evaluation data are logged to MLflow runs."""
    data = []
    for i, tags in enumerate(tags_data):
        item = {
            "inputs": {"question": f"What is question {i}?"},
            "outputs": f"Answer {i}",
            "expectations": {"expected_response": f"Answer {i}"},
            "tags": tags,
        }
        data.append(item)

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
    """Test that tag logging errors don't fail the evaluation."""
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
    """Test that invalid tag types raise appropriate validation errors."""
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
