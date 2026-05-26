import json
import time
from unittest.mock import MagicMock, Mock, patch

import dspy
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import Span
from mlflow.entities.trace import Trace, TraceData, TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import JudgeField
from mlflow.genai.judges.optimizers.dspy_utils import (
    AgentEvalLM,
    agreement_metric,
    append_input_fields_section,
    construct_dspy_lm,
    convert_litellm_to_mlflow_uri,
    create_dspy_signature,
    format_demos_as_examples,
    trace_to_dspy_example,
)
from mlflow.genai.judges.optimizers.memalign.optimizer import MemAlignOptimizer
from mlflow.genai.utils.trace_utils import (
    extract_expectations_from_trace,
    extract_request_from_trace,
    extract_response_from_trace,
)
from mlflow.metrics.genai.model_utils import convert_mlflow_uri_to_litellm
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.utils import build_otel_context

from tests.genai.judges.optimizers.conftest import MockJudge


def _create_trace_with_assessments(trace_id, assessments, inputs=None, outputs=None):
    current_time_ns = int(time.time() * 1e9)
    inputs = inputs or {"inputs": "test input"}
    outputs = outputs or {"outputs": "test output"}

    otel_span = OTelReadableSpan(
        name="root_span",
        context=build_otel_context(hash(trace_id) % 100000, 100),
        parent=None,
        start_time=current_time_ns,
        end_time=current_time_ns + 1000000,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanInputs": json.dumps(inputs),
            "mlflow.spanOutputs": json.dumps(outputs),
            "mlflow.spanType": json.dumps("CHAIN"),
        },
    )

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        state=TraceState.OK,
        execution_duration=1000,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={},
        assessments=assessments,
        request_preview=json.dumps(inputs),
        response_preview=json.dumps(outputs),
    )

    return Trace(info=trace_info, data=TraceData(spans=[Span(otel_span)]))


def _create_human_assessment(name, value, rationale, create_time_ms, source_id="test_user"):
    return Feedback(
        name=name,
        value=value,
        rationale=rationale,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=source_id),
        create_time_ms=create_time_ms,
    )


def test_sanitize_judge_name(sample_trace_with_assessment, mock_judge):
    # The sanitization is now done inside trace_to_dspy_example
    # Test that it correctly handles different judge name formats

    mock_dspy = MagicMock()
    mock_example = MagicMock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        judge1 = MockJudge(name="  mock_judge  ")
        judge2 = MockJudge(name="Mock_Judge")
        judge3 = MockJudge(name="MOCK_JUDGE")
        # trace_to_dspy_example now returns a list
        assert len(trace_to_dspy_example(sample_trace_with_assessment, judge1)) > 0
        assert len(trace_to_dspy_example(sample_trace_with_assessment, judge2)) > 0
        assert len(trace_to_dspy_example(sample_trace_with_assessment, judge3)) > 0


def test_trace_to_dspy_example_two_human_assessments(trace_with_two_human_assessments, mock_judge):
    trace = trace_with_two_human_assessments
    results = trace_to_dspy_example(trace, mock_judge)

    # The fixture has conflicting labels (fail vs pass), so tie-breaking applies
    # The newer assessment (pass) wins
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dspy.Example)
    assert results[0]["result"] == "pass"
    assert results[0]["rationale"] == "Second assessment - should be used (more recent)"


def test_trace_to_dspy_example_human_vs_llm_priority(
    trace_with_human_and_llm_assessments, mock_judge
):
    trace = trace_with_human_and_llm_assessments
    results = trace_to_dspy_example(trace, mock_judge)

    # LLM assessments are filtered out, only the human assessment is returned
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dspy.Example)
    # Should use the HUMAN assessment (LLM assessments are not considered)
    assert results[0]["result"] == "fail"
    assert results[0]["rationale"] == "Human assessment - should be prioritized"


@pytest.mark.parametrize(
    ("trace_fixture", "required_fields", "expected_inputs"),
    [
        # Test different combinations of required fields
        ("sample_trace_with_assessment", ["inputs"], ["inputs"]),
        ("sample_trace_with_assessment", ["outputs"], ["outputs"]),
        ("sample_trace_with_assessment", ["inputs", "outputs"], ["inputs", "outputs"]),
        (
            "sample_trace_with_assessment",
            ["trace", "inputs", "outputs"],
            ["trace", "inputs", "outputs"],
        ),
        ("trace_with_expectations", ["expectations"], ["expectations"]),
        (
            "trace_with_expectations",
            ["inputs", "expectations"],
            ["inputs", "expectations"],
        ),
        (
            "trace_with_expectations",
            ["outputs", "expectations"],
            ["outputs", "expectations"],
        ),
        (
            "trace_with_expectations",
            ["inputs", "outputs", "expectations"],
            ["inputs", "outputs", "expectations"],
        ),
        (
            "trace_with_expectations",
            ["trace", "inputs", "outputs", "expectations"],
            ["trace", "inputs", "outputs", "expectations"],
        ),
    ],
)
def test_trace_to_dspy_example_success(request, trace_fixture, required_fields, expected_inputs):
    trace = request.getfixturevalue(trace_fixture)

    class TestJudge(MockJudge):
        def __init__(self, fields):
            super().__init__(name="mock_judge")
            self._fields = fields

        def get_input_fields(self):
            return [JudgeField(name=field, description=f"Test {field}") for field in self._fields]

    judge = TestJudge(required_fields)

    # Use real DSPy since we've skipped if it's not available
    results = trace_to_dspy_example(trace, judge)

    # Should return a list with one example (single assessment in fixture)
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, dspy.Example)

    # Build expected kwargs based on required fields
    expected_kwargs = {}
    if "trace" in required_fields:
        expected_kwargs["trace"] = trace
    if "inputs" in required_fields:
        expected_kwargs["inputs"] = extract_request_from_trace(trace)
    if "outputs" in required_fields:
        expected_kwargs["outputs"] = extract_response_from_trace(trace)
    if "expectations" in required_fields:
        expected_kwargs["expectations"] = extract_expectations_from_trace(trace)

    # Determine expected rationale based on fixture
    if trace_fixture == "trace_with_expectations":
        expected_rationale = "Meets expectations"
    else:
        expected_rationale = "This looks good"

    # Construct an expected example and assert that the result is the same
    expected_example = dspy.Example(
        result="pass",
        rationale=expected_rationale,
        **expected_kwargs,
    ).with_inputs(*expected_inputs)

    # Compare the examples
    assert result == expected_example


@pytest.mark.parametrize(
    ("trace_fixture", "required_fields"),
    [
        ("sample_trace_with_assessment", ["expectations"]),
        ("sample_trace_with_assessment", ["inputs", "expectations"]),
        ("sample_trace_with_assessment", ["outputs", "expectations"]),
        ("sample_trace_with_assessment", ["inputs", "outputs", "expectations"]),
        (
            "sample_trace_with_assessment",
            ["trace", "inputs", "outputs", "expectations"],
        ),
    ],
)
def test_trace_to_dspy_example_missing_required_fields(request, trace_fixture, required_fields):
    trace = request.getfixturevalue(trace_fixture)

    class TestJudge(MockJudge):
        def __init__(self, fields):
            super().__init__(name="mock_judge")
            self._fields = fields

        def get_input_fields(self):
            return [JudgeField(name=field, description=f"Test {field}") for field in self._fields]

    judge = TestJudge(required_fields)

    # trace_to_dspy_example now returns empty list when fields are missing
    results = trace_to_dspy_example(trace, judge)
    assert results == []


def test_trace_to_dspy_example_no_assessment(sample_trace_without_assessment, mock_judge):
    # Use the fixture for trace without assessment
    trace = sample_trace_without_assessment

    # Should return empty list since there's no matching assessment
    results = trace_to_dspy_example(trace, mock_judge)

    assert results == []


def test_create_dspy_signature(mock_judge):
    signature = create_dspy_signature(mock_judge)

    assert signature.instructions == mock_judge.instructions

    judge_input_fields = mock_judge.get_input_fields()
    for field in judge_input_fields:
        assert field.name in signature.input_fields
        assert signature.input_fields[field.name].json_schema_extra["desc"] == field.description

    judge_output_fields = mock_judge.get_output_fields()
    for field in judge_output_fields:
        assert field.name in signature.output_fields
        assert signature.output_fields[field.name].json_schema_extra["desc"] == field.description


def test_agreement_metric():
    # Test metric with matching results
    example = Mock()
    example.result = "pass"
    pred = Mock()
    pred.result = "pass"

    assert agreement_metric(example, pred) is True

    # Test metric with different results
    pred.result = "fail"
    assert agreement_metric(example, pred) is False


def test_agreement_metric_error_handling():
    # Test with invalid inputs
    result = agreement_metric(None, None)
    assert result is False


@pytest.mark.parametrize(
    ("mlflow_uri", "expected_litellm_uri"),
    [
        ("openai:/gpt-4", "openai/gpt-4"),
        ("openai:/gpt-3.5-turbo", "openai/gpt-3.5-turbo"),
        ("anthropic:/claude-3", "anthropic/claude-3"),
        ("anthropic:/claude-3.5-sonnet", "anthropic/claude-3.5-sonnet"),
        ("cohere:/command", "cohere/command"),
        ("databricks:/dbrx", "databricks/dbrx"),
    ],
)
def test_convert_mlflow_uri_to_litellm(mlflow_uri, expected_litellm_uri):
    assert convert_mlflow_uri_to_litellm(mlflow_uri) == expected_litellm_uri


@pytest.mark.parametrize(
    "invalid_uri",
    [
        "openai-gpt-4",  # Invalid format (missing colon-slash)
        "",  # Empty string
        None,  # None value
    ],
)
def test_convert_mlflow_uri_to_litellm_invalid(invalid_uri):
    with pytest.raises(MlflowException, match="Failed to convert MLflow model URI"):
        convert_mlflow_uri_to_litellm(invalid_uri)


@pytest.mark.parametrize(
    ("litellm_model", "expected_uri"),
    [
        ("openai/gpt-4", "openai:/gpt-4"),
        ("openai/gpt-3.5-turbo", "openai:/gpt-3.5-turbo"),
        ("anthropic/claude-3", "anthropic:/claude-3"),
        ("anthropic/claude-3.5-sonnet", "anthropic:/claude-3.5-sonnet"),
        ("cohere/command", "cohere:/command"),
        ("databricks/dbrx", "databricks:/dbrx"),
    ],
)
def test_convert_litellm_to_mlflow_uri(litellm_model, expected_uri):
    result = convert_litellm_to_mlflow_uri(litellm_model)
    assert result == expected_uri


@pytest.mark.parametrize(
    "invalid_model",
    [
        "openai-gpt-4",  # Missing slash
        "",  # Empty string
        None,  # None value
        "openai/",  # Missing model name
        "/gpt-4",  # Missing provider
        "//",  # Empty provider and model
    ],
)
def test_convert_litellm_to_mlflow_uri_invalid(invalid_model):
    with pytest.raises(MlflowException, match="LiteLLM|empty|None") as exc_info:
        convert_litellm_to_mlflow_uri(invalid_model)

    if invalid_model is None or invalid_model == "":
        assert "cannot be empty or None" in str(exc_info.value)
    elif "/" not in invalid_model:
        assert "Expected format: 'provider/model'" in str(exc_info.value)


@pytest.mark.parametrize(
    "mlflow_uri",
    [
        "openai:/gpt-4",
        "anthropic:/claude-3.5-sonnet",
        "cohere:/command",
        "databricks:/dbrx",
    ],
)
def test_mlflow_to_litellm_uri_round_trip_conversion(mlflow_uri):
    # Convert MLflow -> LiteLLM
    litellm_format = convert_mlflow_uri_to_litellm(mlflow_uri)
    # Convert LiteLLM -> MLflow
    result = convert_litellm_to_mlflow_uri(litellm_format)
    # Should get back the original
    assert result == mlflow_uri, f"Round-trip failed for {mlflow_uri}"


@pytest.mark.parametrize(
    ("model", "expected_type"),
    [
        ("databricks", "AgentEvalLM"),
        ("openai:/gpt-4", "dspy.LM"),
        ("anthropic:/claude-3", "dspy.LM"),
    ],
)
def test_construct_dspy_lm_utility_method(model, expected_type):
    result = construct_dspy_lm(model)

    if expected_type == "AgentEvalLM":
        assert isinstance(result, AgentEvalLM)
    elif expected_type == "dspy.LM":
        assert isinstance(result, dspy.LM)
        # Ensure MLflow URI format is converted (no :/ in the model)
        assert ":/" not in result.model


def test_agent_eval_lm_uses_optimizer_session_name():
    from mlflow.utils import AttrDict

    mock_response = AttrDict({"output": "test response", "error_message": None})

    with (
        patch("mlflow.genai.judges.optimizers.dspy_utils.call_chat_completions") as mock_call,
        patch("mlflow.genai.judges.optimizers.dspy_utils.VERSION", "1.0.0"),
    ):
        mock_call.return_value = mock_response

        agent_lm = AgentEvalLM()
        agent_lm.forward(prompt="test prompt")

        # Verify call_chat_completions was called with the optimizer session name
        mock_call.assert_called_once_with(
            user_prompt="test prompt",
            system_prompt=None,
            session_name="mlflow-judge-optimizer-v1.0.0",
            use_case="judge_alignment",
        )


@pytest.mark.parametrize(
    ("instructions", "field_names", "should_append"),
    [
        # Fields already present - should NOT append
        (
            "Evaluate {{inputs}} and {{outputs}} for quality",
            ["inputs", "outputs"],
            False,
        ),
        # Fields NOT present - should append
        (
            "Evaluate the response for quality",
            ["inputs", "outputs"],
            True,
        ),
        # No fields defined - should NOT append
        (
            "Some instructions",
            [],
            False,
        ),
        # Plain field names present but not mustached - should append
        (
            "Check the inputs and outputs carefully",
            ["inputs", "outputs"],
            True,
        ),
    ],
)
def test_append_input_fields_section(instructions, field_names, should_append):
    class TestJudge(MockJudge):
        def __init__(self, fields):
            super().__init__(name="test_judge")
            self._fields = fields

        def get_input_fields(self):
            return [JudgeField(name=f, description=f"The {f}") for f in self._fields]

    judge = TestJudge(field_names)
    result = append_input_fields_section(instructions, judge)

    if should_append:
        assert result != instructions
        assert "Inputs for assessment:" in result
        for field in field_names:
            assert f"{{{{ {field} }}}}" in result
    else:
        assert result == instructions


def test_format_demos_empty_list(mock_judge):
    result = format_demos_as_examples([], mock_judge)
    assert result == ""


def test_format_demos_multiple_demos(mock_judge):
    long_input = "x" * 600
    demos = [
        dspy.Example(inputs="Q1", outputs="A1", result="pass", rationale="Good"),
        dspy.Example(inputs="Q2", outputs="A2", result="fail", rationale="Bad"),
        dspy.Example(inputs=long_input, outputs="short", result="pass", rationale="Test"),
    ]

    result = format_demos_as_examples(demos, mock_judge)

    assert "Example 1:" in result
    assert "Example 2:" in result
    assert "Example 3:" in result
    assert "inputs: Q1" in result
    assert "inputs: Q2" in result
    # Long values should NOT be truncated
    assert long_input in result


def test_format_demos_respects_judge_fields():
    class CustomFieldsJudge(MockJudge):
        def get_input_fields(self):
            return [
                JudgeField(name="query", description="The query"),
                JudgeField(name="context", description="The context"),
            ]

        def get_output_fields(self):
            return [JudgeField(name="verdict", description="The verdict")]

    judge = CustomFieldsJudge(name="custom_judge")
    demo = dspy.Example(
        query="What is AI?",
        context="AI is artificial intelligence",
        verdict="pass",
        extra_field="should not appear",  # Not in judge fields
    )

    result = format_demos_as_examples([demo], judge)

    assert "query: What is AI?" in result
    assert "context: AI is artificial intelligence" in result
    assert "verdict: pass" in result
    assert "extra_field" not in result


def test_format_demos_raises_on_invalid_demo(mock_judge):
    class NonDictDemo:
        pass

    demos = [
        dspy.Example(inputs="Q1", outputs="A1", result="pass", rationale="Good"),
        NonDictDemo(),  # Invalid demo - should raise exception
    ]

    with pytest.raises(MlflowException, match="Demo at index 1 cannot be converted to dict"):
        format_demos_as_examples(demos, mock_judge)


# assessments_spec: list of (label, timestamp_offset) tuples
# Each tuple creates an assessment with the given label at base_time + offset
@pytest.mark.parametrize(
    ("assessments_spec", "expected_count", "expected_label"),
    [
        # No conflict: all assessments agree
        pytest.param([("pass", 1000), ("pass", 2000)], 2, "pass", id="two_agreeing_pass"),
        pytest.param(
            [("fail", 1000), ("fail", 2000), ("fail", 3000)], 3, "fail", id="three_agreeing_fail"
        ),
        # Majority wins
        pytest.param(
            [("pass", 1000), ("pass", 2000), ("fail", 3000)], 2, "pass", id="majority_2v1_pass_wins"
        ),
        pytest.param(
            [("fail", 1000), ("pass", 2000), ("fail", 3000)], 2, "fail", id="majority_2v1_fail_wins"
        ),
        pytest.param(
            [("pass", 1000), ("pass", 2000), ("pass", 3000), ("fail", 4000)],
            3,
            "pass",
            id="majority_3v1_pass_wins",
        ),
        # Tie-breaking by recency
        pytest.param([("pass", 1000), ("fail", 2000)], 1, "fail", id="tie_1v1_fail_more_recent"),
        pytest.param(
            [("pass", 1000), ("pass", 2000), ("fail", 3000), ("fail", 4000)],
            2,
            "fail",
            id="tie_2v2_fail_more_recent",
        ),
        # Three-way tie: most recent wins
        pytest.param(
            [("pass", 1000), ("fail", 2000), ("maybe", 3000)],
            1,
            "maybe",
            id="three_way_tie_maybe_most_recent",
        ),
    ],
)
def test_trace_to_dspy_example_multi_assessment(
    mock_judge, assessments_spec, expected_count, expected_label
):
    base_time = int(time.time() * 1000)
    assessments = [
        _create_human_assessment(
            "mock_judge", label, f"rationale_{i}", base_time + offset, f"user{i}"
        )
        for i, (label, offset) in enumerate(assessments_spec)
    ]

    trace = _create_trace_with_assessments("test_multi", assessments)
    results = trace_to_dspy_example(trace, mock_judge)

    assert len(results) == expected_count
    for result in results:
        assert result["result"] == expected_label


def test_trace_to_dspy_example_conflict_logs_warning(mock_judge, capsys):
    base_time = int(time.time() * 1000)
    assessments = [
        _create_human_assessment("mock_judge", "pass", "Pass 1", base_time, "alice"),
        _create_human_assessment("mock_judge", "pass", "Pass 2", base_time + 1000, "bob"),
        _create_human_assessment("mock_judge", "fail", "Discarded", base_time + 2000, "charlie"),
    ]

    trace = _create_trace_with_assessments("test_trace_123", assessments)
    trace_to_dspy_example(trace, mock_judge)

    captured = capsys.readouterr()
    assert "discarded" in captured.err.lower()
    assert "test_trace_123" in captured.err


def test_trace_to_dspy_example_assessments_without_timestamps(mock_judge):
    assessments = [
        Feedback(
            name="mock_judge",
            value="pass",
            rationale="R1",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="u1"),
        ),
        Feedback(
            name="mock_judge",
            value="pass",
            rationale="R2",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="u2"),
        ),
    ]
    trace = _create_trace_with_assessments("test_no_timestamps", assessments)
    results = trace_to_dspy_example(trace, mock_judge)

    assert len(results) == 2


def test_trace_to_dspy_example_filters_out_llm_assessments(mock_judge):
    assessments = [
        Feedback(
            name="mock_judge",
            value="pass",
            rationale="LLM",
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt"),
        ),
    ]
    trace = _create_trace_with_assessments("test_llm_only", assessments)
    results = trace_to_dspy_example(trace, mock_judge)

    assert len(results) == 0


def test_trace_to_dspy_example_empty_assessments(mock_judge):
    trace = _create_trace_with_assessments("test_empty", [])
    results = trace_to_dspy_example(trace, mock_judge)

    assert len(results) == 0


def test_trace_to_dspy_example_filters_by_judge_name(mock_judge):
    base_time = int(time.time() * 1000)
    assessments = [
        _create_human_assessment("other_judge", "pass", "Wrong judge", base_time, "user"),
    ]
    trace = _create_trace_with_assessments("test_wrong_judge", assessments)
    results = trace_to_dspy_example(trace, mock_judge)

    assert len(results) == 0


def test_trace_to_dspy_example_mixed_human_and_llm_only_uses_human(mock_judge):
    base_time = int(time.time() * 1000)

    assessments = [
        _create_human_assessment("mock_judge", "fail", "Human 1", base_time - 1000),
        Feedback(
            name="mock_judge",
            value="pass",
            rationale="LLM",
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt"),
            create_time_ms=base_time,
        ),
        _create_human_assessment("mock_judge", "fail", "Human 2", base_time - 500),
    ]

    trace = _create_trace_with_assessments("test_mixed", assessments)
    results = trace_to_dspy_example(trace, mock_judge)

    assert len(results) == 2
    for result in results:
        assert result["result"] == "fail"


def test_memalign_optimizer_handles_multi_assessment_traces(mock_judge):
    base_time = int(time.time() * 1000)
    optimizer = MemAlignOptimizer(
        reflection_lm="openai:/gpt-4o-mini",
        embedding_model="openai:/text-embedding-3-small",
    )

    trace = _create_trace_with_assessments(
        "multi_trace",
        [
            _create_human_assessment("mock_judge", "pass", f"R{i}", base_time - i * 1000)
            for i in range(3)
        ],
    )

    with (
        patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.distill_guidelines",
            return_value=[],
        ),
        patch("dspy.Embedder"),
        patch("dspy.retrievers.Embeddings"),
    ):
        aligned_judge = optimizer.align(mock_judge, [trace])

        assert len(aligned_judge._episodic_memory) == 3
        assert all(ex._trace_id == "multi_trace" for ex in aligned_judge._episodic_memory)
