import json
import sys
from typing import Any, Literal
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

import mlflow
from mlflow.entities import TraceData, TraceInfo, TraceLocation, TraceState
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai import scorer
from mlflow.genai.datasets import EvaluationDataset, create_dataset
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.utils import (
    _convert_scorer_to_legacy_metric,
    _convert_to_eval_set,
    classify_scorers,
    get_first_trace_in_session,
    group_traces_by_session,
    validate_tags,
)
from mlflow.genai.scorers.builtin_scorers import RelevanceToQuery
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.utils.spark_utils import is_spark_connect_mode

from tests.genai.conftest import databricks_only


@pytest.fixture(scope="module")
def spark():
    # databricks-agents installs databricks-connect
    if is_spark_connect_mode():
        pytest.skip("Local Spark Session is not supported when databricks-connect is installed.")

    from pyspark.sql import SparkSession

    with SparkSession.builder.getOrCreate() as spark:
        yield spark


def count_rows(data: Any) -> int:
    try:
        from mlflow.utils.spark_utils import get_spark_dataframe_type

        if isinstance(data, get_spark_dataframe_type()):
            return data.count()
    except Exception:
        pass

    if isinstance(data, EvaluationDataset):
        data = data.to_df()

    return len(data)


@pytest.fixture
def sample_dict_data_single():
    return [
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "actual response for first question",
            "expectations": {"expected_response": "expected response for first question"},
            "tags": {"sample_tag": "value"},
        },
    ]


@pytest.fixture
def sample_dict_data_multiple():
    return [
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "actual response for first question",
            "expectations": {"expected_response": "expected response for first question"},
            "tags": {"category": "spark"},
        },
        {
            "inputs": {"question": "How can you minimize data shuffling in Spark?"},
            "outputs": "actual response for second question",
            "expectations": {"expected_response": "expected response for second question"},
            "tags": {"category": "spark", "topic": "optimization"},
        },
        # Some records might not have expectations or tags
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "actual response for third question",
            "expectations": {},
            "tags": {},
        },
    ]


@pytest.fixture
def sample_dict_data_multiple_with_custom_expectations():
    return [
        {
            "inputs": {"question": "What is Spark?"},
            "outputs": "actual response for first question",
            "expectations": {
                "expected_response": "expected response for first question",
                "my_custom_expectation": "custom expectation for the first question",
            },
        },
        {
            "inputs": {"question": "How can you minimize data shuffling in Spark?"},
            "outputs": "actual response for second question",
            "expectations": {
                "expected_response": "expected response for second question",
                "my_custom_expectation": "custom expectation for the second question",
            },
        },
        # Some records might not have all expectations
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": "actual response for third question",
            "expectations": {
                "my_custom_expectation": "custom expectation for the third question",
            },
        },
    ]


@pytest.fixture
def sample_pd_data(sample_dict_data_multiple):
    """Returns a pandas DataFrame with sample data"""
    return pd.DataFrame(sample_dict_data_multiple)


@pytest.fixture
def sample_spark_data(sample_pd_data, spark):
    """Convert pandas DataFrame to PySpark DataFrame"""
    return spark.createDataFrame(sample_pd_data)


@pytest.fixture
def sample_spark_data_with_string_columns(sample_pd_data, spark):
    # Cast inputs and expectations columns to string
    df = sample_pd_data.copy()
    df["inputs"] = df["inputs"].apply(json.dumps)
    df["expectations"] = df["expectations"].apply(json.dumps)
    return spark.createDataFrame(df)


@pytest.fixture
def sample_evaluation_dataset(sample_dict_data_single):
    dataset = create_dataset("test")
    dataset.merge_records(sample_dict_data_single)
    return dataset


_ALL_DATA_FIXTURES = [
    "sample_dict_data_single",
    "sample_dict_data_multiple",
    "sample_dict_data_multiple_with_custom_expectations",
    "sample_pd_data",
    "sample_spark_data",
    "sample_spark_data_with_string_columns",
    "sample_evaluation_dataset",
]


class TestModel:
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, question: str) -> str:
        response = self.call_llm(messages=[{"role": "user", "content": question}])
        return response["choices"][0]["message"]["content"]

    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": "I don't know"}}]}


def get_test_traces(type=Literal["pandas", "list"]):
    model = TestModel()

    model.predict("What is MLflow?")
    trace_id = mlflow.get_last_active_trace_id()

    # Add assessments. Since log_assessment API is not supported in OSS MLflow yet, we
    # need to add it to the trace info manually.
    source = AssessmentSource(source_id="test", source_type="HUMAN")
    # 1. Expectation with reserved name "expected_response"
    mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_response",
        value="expected response for first question",
        source=source,
    )
    # 2. Expectation with reserved name "expected_facts"
    mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_facts",
        value=["fact1", "fact2"],
        source=source,
    )
    # 3. Expectation with reserved name "guidelines"
    mlflow.log_expectation(
        trace_id=trace_id,
        name="guidelines",
        value=["Be polite", "Be kind"],
        source=source,
    )
    # 4. Expectation with custom name "my_custom_expectation"
    mlflow.log_expectation(
        trace_id=trace_id,
        name="my_custom_expectation",
        value="custom expectation for the first question",
        source=source,
    )
    # 5. Non-expectation assessment
    mlflow.log_feedback(
        trace_id=trace_id,
        name="feedback",
        value="some feedback",
        source=source,
    )
    traces = mlflow.search_traces(return_type=type, order_by=["timestamp_ms ASC"])
    return [{"trace": trace} for trace in traces] if type == "list" else traces


@pytest.mark.parametrize("input_type", ["list", "pandas"])
def test_convert_to_legacy_eval_traces(input_type):
    sample_data = get_test_traces(type=input_type)
    data = _convert_to_eval_set(sample_data)

    assert "trace" in data.columns

    # "inputs" column should be derived from the trace
    assert "inputs" in data.columns
    assert list(data["inputs"]) == [{"question": "What is MLflow?"}]
    assert data["expectations"][0] == {
        "expected_response": "expected response for first question",
        "expected_facts": ["fact1", "fact2"],
        "guidelines": ["Be polite", "Be kind"],
        "my_custom_expectation": "custom expectation for the first question",
    }
    # Assessment with type "Feedback" should not be present in the transformed data
    assert "feedback" not in data.columns


@pytest.mark.parametrize("data_fixture", _ALL_DATA_FIXTURES)
def test_convert_to_eval_set_has_no_errors(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    transformed_data = _convert_to_eval_set(sample_data)

    assert "inputs" in transformed_data.columns
    assert "outputs" in transformed_data.columns
    assert "expectations" in transformed_data.columns


def test_convert_to_eval_set_without_request_and_response():
    for _ in range(3):
        with mlflow.start_span():
            pass

    trace_df = mlflow.search_traces()
    trace_df = trace_df[["trace"]]
    transformed_data = _convert_to_eval_set(trace_df)

    assert "inputs" in transformed_data.columns
    assert "outputs" in transformed_data.columns
    assert transformed_data["inputs"].isna().all()


def test_convert_to_legacy_eval_raise_for_invalid_json_columns(spark):
    # Data with invalid `inputs` column
    df = spark.createDataFrame(
        [
            {"inputs": "invalid json", "expectations": '{"expected_response": "expected"}'},
            {"inputs": "invalid json", "expectations": '{"expected_response": "expected"}'},
        ]
    )
    with pytest.raises(MlflowException, match="Failed to parse `inputs` column."):
        _convert_to_eval_set(df)

    # Data with invalid `expectations` column
    df = spark.createDataFrame(
        [
            {
                "inputs": '{"question": "What is the capital of France?"}',
                "expectations": "invalid expectations",
            },
            {
                "inputs": '{"question": "What is the capital of Germany?"}',
                "expectations": "invalid expectations",
            },
        ]
    )
    with pytest.raises(MlflowException, match="Failed to parse `expectations` column."):
        _convert_to_eval_set(df)


@pytest.mark.parametrize("data_fixture", _ALL_DATA_FIXTURES)
def test_scorer_receives_correct_data(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    received_args = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations):
        received_args.append(
            (
                inputs["question"],
                outputs,
                expectations.get("expected_response"),
                expectations.get("my_custom_expectation"),
            )
        )
        return 0

    mlflow.genai.evaluate(
        data=sample_data,
        scorers=[dummy_scorer],
    )

    all_inputs, all_outputs, all_expectations, all_custom_expectations = zip(*received_args)
    row_count = count_rows(sample_data)
    expected_inputs = [
        "What is Spark?",
        "How can you minimize data shuffling in Spark?",
        "What is MLflow?",
    ][:row_count]
    expected_outputs = [
        "actual response for first question",
        "actual response for second question",
        "actual response for third question",
    ][:row_count]
    expected_expectations = [
        "expected response for first question",
        "expected response for second question",
        None,
    ][:row_count]

    assert set(all_inputs) == set(expected_inputs)
    assert set(all_outputs) == set(expected_outputs)
    assert set(all_expectations) == set(expected_expectations)

    if data_fixture == "sample_dict_data_multiple_with_custom_expectations":
        expected_custom_expectations = [
            "custom expectation for the first question",
            "custom expectation for the second question",
            "custom expectation for the third question",
        ]
        assert set(all_custom_expectations) == set(expected_custom_expectations)


def test_input_is_required_if_trace_is_not_provided():
    with patch("mlflow.genai.evaluation.harness.run") as mock_evaluate:
        with pytest.raises(MlflowException, match="inputs.*required"):
            mlflow.genai.evaluate(
                data=pd.DataFrame({"outputs": ["Paris"]}),
                scorers=[RelevanceToQuery()],
            )

        mock_evaluate.assert_not_called()

        mlflow.genai.evaluate(
            data=pd.DataFrame(
                {"inputs": [{"question": "What is the capital of France?"}], "outputs": ["Paris"]}
            ),
            scorers=[RelevanceToQuery()],
        )
        mock_evaluate.assert_called_once()


def test_input_is_optional_if_trace_is_provided():
    with mlflow.start_span() as span:
        span.set_inputs({"question": "What is the capital of France?"})
        span.set_outputs("Paris")

    trace = mlflow.get_trace(span.trace_id)

    with patch("mlflow.genai.evaluation.harness.run") as mock_evaluate:
        mlflow.genai.evaluate(
            data=pd.DataFrame({"trace": [trace]}),
            scorers=[RelevanceToQuery()],
        )

        mock_evaluate.assert_called_once()


@pytest.mark.parametrize("input_type", ["list", "pandas"])
def test_scorer_receives_correct_data_with_trace_data(input_type, monkeypatch: pytest.MonkeyPatch):
    sample_data = get_test_traces(type=input_type)
    received_args = []

    @scorer
    def dummy_scorer(inputs, outputs, expectations, trace):
        received_args.append((inputs, outputs, expectations, trace))
        return 0

    # Disable logging traces to MLflow to avoid calling mlflow APIs which need to be mocked
    monkeypatch.setenv("AGENT_EVAL_LOG_TRACES_TO_MLFLOW_ENABLED", "false")
    mlflow.genai.evaluate(
        data=sample_data,
        scorers=[dummy_scorer],
    )

    inputs, outputs, expectations, trace = received_args[0]
    assert inputs == {"question": "What is MLflow?"}
    assert outputs == "I don't know"
    assert expectations == {
        "expected_response": "expected response for first question",
        "expected_facts": ["fact1", "fact2"],
        "guidelines": ["Be polite", "Be kind"],
        "my_custom_expectation": "custom expectation for the first question",
    }
    assert isinstance(trace, Trace)


@pytest.mark.parametrize("data_fixture", _ALL_DATA_FIXTURES)
def test_predict_fn_receives_correct_data(data_fixture, request):
    sample_data = request.getfixturevalue(data_fixture)

    received_args = []

    def predict_fn(question: str):
        received_args.append(question)
        return question

    @scorer
    def dummy_scorer(inputs, outputs):
        return 0

    mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=sample_data,
        scorers=[dummy_scorer],
    )

    received_args.pop(0)  # Remove the one-time prediction to check if a model is traced
    row_count = count_rows(sample_data)
    assert len(received_args) == row_count
    expected_contents = [
        "What is Spark?",
        "How can you minimize data shuffling in Spark?",
        "What is MLflow?",
    ][:row_count]
    # Using set because eval harness runs predict_fn in parallel
    assert set(received_args) == set(expected_contents)


def test_convert_scorer_to_legacy_metric_aggregations_attribute(monkeypatch):
    mock_metric_instance = MagicMock()

    # NB: Mocking the behavior of databricks-agents, which does not have the aggregations
    # argument for the evaluation interface for a metric.
    def mock_metric_decorator(**kwargs):
        if "aggregations" in kwargs:
            raise TypeError("metric() got an unexpected keyword argument 'aggregations'")
        assert set(kwargs.keys()) <= {"eval_fn", "name"}
        return mock_metric_instance

    mock_evals = Mock(metric=mock_metric_decorator)
    mock_evals.judges = Mock()  # Add the judges submodule to prevent AttributeError

    monkeypatch.setitem(sys.modules, "databricks.agents.evals", mock_evals)
    monkeypatch.setitem(sys.modules, "databricks.agents.evals.judges", mock_evals.judges)

    mock_scorer = Mock()
    mock_scorer.name = "test_scorer"
    mock_scorer.aggregations = ["mean", "max", "p90"]
    mock_scorer.run = Mock(return_value={"score": 1.0})

    result = _convert_scorer_to_legacy_metric(mock_scorer)

    assert result.aggregations == ["mean", "max", "p90"]


@databricks_only
def test_convert_scorer_to_legacy_metric():
    """Test that _convert_scorer_to_legacy_metric correctly sets _is_builtin_scorer attribute."""
    # Test with a built-in scorer
    builtin_scorer = RelevanceToQuery()
    legacy_metric = _convert_scorer_to_legacy_metric(builtin_scorer)

    # Verify the metric has the _is_builtin_scorer attribute set to True
    assert hasattr(legacy_metric, "_is_builtin_scorer")
    assert legacy_metric._is_builtin_scorer is True
    assert legacy_metric.name == builtin_scorer.name

    # Test with a custom scorer
    @scorer(name="custom_scorer", aggregations=["mean", "max"])
    def custom_scorer_func(inputs, outputs=None, expectations=None, **kwargs):
        return {"score": 1.0}

    custom_scorer_instance = custom_scorer_func
    legacy_metric_custom = _convert_scorer_to_legacy_metric(custom_scorer_instance)

    # Verify the metric has the _is_builtin_scorer attribute set to False
    assert hasattr(legacy_metric_custom, "_is_builtin_scorer")
    assert legacy_metric_custom._is_builtin_scorer is False
    assert legacy_metric_custom.name == custom_scorer_instance.name
    assert legacy_metric_custom.aggregations == custom_scorer_instance.aggregations


@pytest.mark.parametrize(
    "aggregations",
    [
        ["mean", "max", "mean", "median", "variance", "p90"],
        [sum, max],
    ],
)
@databricks_only
def test_scorer_pass_through_aggregations(aggregations):
    @scorer(name="custom_scorer", aggregations=aggregations)
    def custom_scorer_func(outputs):
        return {"score": 1.0}

    legacy_metric_custom = _convert_scorer_to_legacy_metric(custom_scorer_func)
    assert legacy_metric_custom.name == "custom_scorer"
    assert legacy_metric_custom.aggregations == aggregations

    builtin_scorer = RelevanceToQuery(aggregations=aggregations)
    legacy_metric_builtin = _convert_scorer_to_legacy_metric(builtin_scorer)
    assert legacy_metric_builtin.name == "relevance_to_query"
    assert legacy_metric_builtin.aggregations == builtin_scorer.aggregations


@pytest.mark.parametrize(
    "tags",
    [
        None,
        {},
        {"key": "value"},
        {"env": "test", "model": "v1.0"},
        {"key": 123},  # Values can be any type
        {"key1": "value1", "key2": None},  # Values can be any type
    ],
)
def test_validate_tags_valid(tags):
    validate_tags(tags)


@pytest.mark.parametrize(
    ("tags", "expected_error"),
    [
        ("invalid", "Tags must be a dictionary, got str"),
        (123, "Tags must be a dictionary, got int"),
        ([1, 2, 3], "Tags must be a dictionary, got list"),
        ({123: "value"}, "Invalid tags:\n  - Key 123 has type int; expected str."),
        (
            {"key1": "value1", 123: "value2"},
            "Invalid tags:\n  - Key 123 has type int; expected str.",
        ),
        (
            {123: "value1", 456: "value2"},
            (
                "Invalid tags:\n  - Key 123 has type int; expected str."
                "\n  - Key 456 has type int; expected str."
            ),
        ),
    ],
)
def test_validate_tags_invalid(tags, expected_error):
    with pytest.raises(MlflowException, match=expected_error):
        validate_tags(tags)


# ==================== Tests for Multi-Turn Helper Functions ====================


class _MultiTurnTestScorer(mlflow.genai.Scorer):
    """Test scorer that simulates a multi-turn scorer.

    Overrides is_session_level_scorer property to return True.
    """

    def __init__(self, name: str = "multi_turn_test_scorer"):
        super().__init__(name=name)

    @property
    def is_session_level_scorer(self) -> bool:
        return True

    def __call__(self, traces=None, **kwargs):
        return 1.0


def test_classify_scorers_all_single_turn():
    """Test that all scorers are classified as single-turn when none are multi-turn."""

    @scorer
    def custom_scorer1(outputs):
        return 1.0

    @scorer
    def custom_scorer2(outputs):
        return 2.0

    scorers_list = [custom_scorer1, custom_scorer2]
    single_turn, multi_turn = classify_scorers(scorers_list)

    assert len(single_turn) == 2
    assert len(multi_turn) == 0
    assert single_turn == scorers_list


def test_classify_scorers_all_multi_turn():
    """Test that all scorers are classified as multi-turn.

    When all scorers have is_session_level_scorer=True.
    """
    multi_turn_scorer1 = _MultiTurnTestScorer(name="multi_turn_scorer1")
    multi_turn_scorer2 = _MultiTurnTestScorer(name="multi_turn_scorer2")

    scorers_list = [multi_turn_scorer1, multi_turn_scorer2]
    single_turn, multi_turn = classify_scorers(scorers_list)

    assert len(single_turn) == 0
    assert len(multi_turn) == 2
    assert multi_turn == scorers_list
    # Verify they are actually multi-turn
    assert multi_turn_scorer1.is_session_level_scorer is True
    assert multi_turn_scorer2.is_session_level_scorer is True


def test_classify_scorers_mixed():
    """Test classification of mixed single-turn and multi-turn scorers."""

    @scorer
    def single_turn_scorer(outputs):
        return 1.0

    multi_turn_scorer = _MultiTurnTestScorer(name="multi_turn_scorer")

    scorers_list = [single_turn_scorer, multi_turn_scorer]
    single_turn, multi_turn = classify_scorers(scorers_list)

    assert len(single_turn) == 1
    assert len(multi_turn) == 1
    assert single_turn[0] == single_turn_scorer
    assert multi_turn[0] == multi_turn_scorer
    # Verify properties
    assert single_turn_scorer.is_session_level_scorer is False
    assert multi_turn_scorer.is_session_level_scorer is True


def test_classify_scorers_empty_list():
    """Test classification of an empty list of scorers."""
    single_turn, multi_turn = classify_scorers([])

    assert len(single_turn) == 0
    assert len(multi_turn) == 0


def _create_mock_trace(trace_id: str, session_id: str | None, request_time: int):
    """Helper to create a mock trace with session_id and request_time."""
    trace_metadata = {}
    if session_id is not None:
        trace_metadata[TraceMetadataKey.TRACE_SESSION] = session_id

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=request_time,
        execution_duration=1000,
        state=TraceState.OK,
        trace_metadata=trace_metadata,
        tags={},
    )

    trace = Mock(spec=Trace)
    trace.info = trace_info
    trace.data = TraceData(spans=[])
    return trace


def _create_mock_eval_item(trace):
    """Helper to create a mock EvalItem with a trace."""
    eval_item = Mock(spec=EvalItem)
    eval_item.trace = trace
    return eval_item


def test_group_traces_by_session_single_session():
    """Test grouping traces that all belong to a single session."""
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    trace2 = _create_mock_trace("trace-2", "session-1", 2000)
    trace3 = _create_mock_trace("trace-3", "session-1", 3000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = _create_mock_eval_item(trace2)
    eval_item3 = _create_mock_eval_item(trace3)

    eval_items = [eval_item1, eval_item2, eval_item3]
    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 1
    assert "session-1" in session_groups
    assert len(session_groups["session-1"]) == 3

    # Check that all traces are included
    session_traces = [item.trace for item in session_groups["session-1"]]
    assert trace1 in session_traces
    assert trace2 in session_traces
    assert trace3 in session_traces


def test_group_traces_by_session_multiple_sessions():
    """Test grouping traces that belong to different sessions."""
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    trace2 = _create_mock_trace("trace-2", "session-1", 2000)
    trace3 = _create_mock_trace("trace-3", "session-2", 1500)
    trace4 = _create_mock_trace("trace-4", "session-2", 2500)

    eval_items = [
        _create_mock_eval_item(trace1),
        _create_mock_eval_item(trace2),
        _create_mock_eval_item(trace3),
        _create_mock_eval_item(trace4),
    ]

    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 2
    assert "session-1" in session_groups
    assert "session-2" in session_groups
    assert len(session_groups["session-1"]) == 2
    assert len(session_groups["session-2"]) == 2


def test_group_traces_by_session_excludes_no_session_id():
    """Test that traces without session_id are excluded from grouping."""
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    trace2 = _create_mock_trace("trace-2", None, 2000)  # No session_id
    trace3 = _create_mock_trace("trace-3", "session-1", 3000)

    eval_items = [
        _create_mock_eval_item(trace1),
        _create_mock_eval_item(trace2),
        _create_mock_eval_item(trace3),
    ]

    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 1
    assert "session-1" in session_groups
    assert len(session_groups["session-1"]) == 2
    # trace2 should not be included
    session_traces = [item.trace for item in session_groups["session-1"]]
    assert trace1 in session_traces
    assert trace2 not in session_traces
    assert trace3 in session_traces


def test_group_traces_by_session_excludes_none_traces():
    """Test that eval items without traces are excluded from grouping."""
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = Mock()
    eval_item2.trace = None  # No trace

    eval_items = [eval_item1, eval_item2]
    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 1
    assert "session-1" in session_groups
    assert len(session_groups["session-1"]) == 1


def test_group_traces_by_session_empty_list():
    """Test grouping an empty list of eval items."""
    session_groups = group_traces_by_session([])

    assert len(session_groups) == 0
    assert session_groups == {}


def test_get_first_trace_in_session_chronological_order():
    """Test that the first trace is correctly identified by request_time."""
    trace1 = _create_mock_trace("trace-1", "session-1", 3000)
    trace2 = _create_mock_trace("trace-2", "session-1", 1000)  # Earliest
    trace3 = _create_mock_trace("trace-3", "session-1", 2000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = _create_mock_eval_item(trace2)
    eval_item3 = _create_mock_eval_item(trace3)

    session_items = [eval_item1, eval_item2, eval_item3]

    first_item = get_first_trace_in_session(session_items)

    assert first_item.trace == trace2
    assert first_item == eval_item2


def test_get_first_trace_in_session_single_trace():
    """Test getting the first trace when there's only one trace."""
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    eval_item1 = _create_mock_eval_item(trace1)

    session_items = [eval_item1]

    first_item = get_first_trace_in_session(session_items)

    assert first_item.trace == trace1
    assert first_item == eval_item1


def test_get_first_trace_in_session_same_timestamp():
    """Test behavior when multiple traces have the same timestamp."""
    # When timestamps are equal, min() will return the first one in the list
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    trace2 = _create_mock_trace("trace-2", "session-1", 1000)
    trace3 = _create_mock_trace("trace-3", "session-1", 1000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = _create_mock_eval_item(trace2)
    eval_item3 = _create_mock_eval_item(trace3)

    session_items = [eval_item1, eval_item2, eval_item3]

    first_item = get_first_trace_in_session(session_items)

    # Should return one of the traces with timestamp 1000 (likely the first one)
    assert first_item.trace.info.request_time == 1000


# ==================== Tests for Multi-Turn Validation ====================


def test_validate_session_level_evaluation_inputs_no_session_level_scorers():
    """Test that validation passes when there are no session-level scorers."""
    from mlflow.genai.evaluation.utils import validate_session_level_evaluation_inputs

    @scorer
    def single_turn_scorer(outputs):
        return 1.0

    scorers_list = [single_turn_scorer]

    # Should not raise any exceptions
    validate_session_level_evaluation_inputs(
        scorers=scorers_list,
        predict_fn=None,
    )


def test_validate_session_level_evaluation_inputs_feature_flag_disabled():
    """Test that validation raises error when feature flag is disabled."""
    import os

    from mlflow.exceptions import MlflowException
    from mlflow.genai.evaluation.utils import validate_session_level_evaluation_inputs

    # Make sure feature flag is disabled
    os.environ.pop("MLFLOW_ENABLE_MULTI_TURN_EVALUATION", None)

    multi_turn_scorer = _MultiTurnTestScorer()
    scorers_list = [multi_turn_scorer]

    with pytest.raises(
        MlflowException,
        match="Multi-turn evaluation is not enabled",
    ):
        validate_session_level_evaluation_inputs(
            scorers=scorers_list,
            predict_fn=None,
        )


def test_validate_session_level_evaluation_inputs_with_predict_fn():
    """Test that validation raises error when predict_fn is provided with session-level scorers."""
    import os

    from mlflow.exceptions import MlflowException
    from mlflow.genai.evaluation.utils import validate_session_level_evaluation_inputs

    # Enable feature flag
    os.environ["MLFLOW_ENABLE_MULTI_TURN_EVALUATION"] = "true"

    try:
        multi_turn_scorer = _MultiTurnTestScorer()
        scorers_list = [multi_turn_scorer]

        def dummy_predict_fn():
            return "output"

        with pytest.raises(
            MlflowException,
            match="Multi-turn scorers are not yet supported with predict_fn",
        ):
            validate_session_level_evaluation_inputs(
                scorers=scorers_list,
                predict_fn=dummy_predict_fn,
            )
    finally:
        os.environ.pop("MLFLOW_ENABLE_MULTI_TURN_EVALUATION", None)


def test_validate_session_level_evaluation_inputs_valid():
    """Test that validation passes with valid session-level input."""
    import os

    from mlflow.genai.evaluation.utils import validate_session_level_evaluation_inputs

    # Enable feature flag
    os.environ["MLFLOW_ENABLE_MULTI_TURN_EVALUATION"] = "true"

    try:
        multi_turn_scorer = _MultiTurnTestScorer()
        scorers_list = [multi_turn_scorer]

        # Should not raise any exceptions
        validate_session_level_evaluation_inputs(
            scorers=scorers_list,
            predict_fn=None,
        )
    finally:
        os.environ.pop("MLFLOW_ENABLE_MULTI_TURN_EVALUATION", None)


def test_validate_session_level_evaluation_inputs_mixed_scorers():
    """Test validation with mixed single-turn and session-level scorers."""
    import os

    from mlflow.genai.evaluation.utils import validate_session_level_evaluation_inputs

    # Enable feature flag
    os.environ["MLFLOW_ENABLE_MULTI_TURN_EVALUATION"] = "true"

    try:
        @scorer
        def single_turn_scorer(outputs):
            return 1.0

        multi_turn_scorer = _MultiTurnTestScorer()
        scorers_list = [single_turn_scorer, multi_turn_scorer]

        # Should not raise any exceptions
        validate_session_level_evaluation_inputs(
            scorers=scorers_list,
            predict_fn=None,
        )
    finally:
        os.environ.pop("MLFLOW_ENABLE_MULTI_TURN_EVALUATION", None)
