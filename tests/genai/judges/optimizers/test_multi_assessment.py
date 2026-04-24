"""Tests for multi-assessment alignment behavior in trace_to_dspy_example.

This module tests the behavior when multiple human assessments with the same name
exist on a single trace:
- No conflict: all assessments have same label -> use all for stronger signal
- Conflict with majority: use majority label assessments, warn about discarded
- Tie: use most recent assessment(s) among tied groups
"""

import json
import time
from unittest import mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import Span
from mlflow.entities.trace import Trace, TraceData, TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.optimizers.dspy_utils import trace_to_dspy_example
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.utils import build_otel_context

from tests.genai.judges.optimizers.conftest import MockJudge


@pytest.fixture
def mock_judge():
    """Create a mock judge for testing."""
    return MockJudge(model="openai:/gpt-3.5-turbo")


def _create_trace_with_assessments(
    trace_id: str,
    assessments: list[Feedback],
    inputs: dict | None = None,
    outputs: dict | None = None,
) -> Trace:
    """Helper to create a trace with given assessments."""
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

    real_span = Span(otel_span)

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

    trace_data = TraceData(spans=[real_span])
    return Trace(info=trace_info, data=trace_data)


def _create_human_assessment(
    name: str,
    value: str,
    rationale: str,
    create_time_ms: int,
    source_id: str = "test_user",
) -> Feedback:
    """Helper to create a human assessment with specific timestamp."""
    return Feedback(
        name=name,
        value=value,
        rationale=rationale,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=source_id),
        create_time_ms=create_time_ms,
    )


class TestMultiAssessmentNoConflict:
    """Tests for multiple assessments with no label conflict (all agree)."""

    def test_two_assessments_same_label_returns_both(self, mock_judge):
        """When two assessments agree on the label, both should be returned."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="First reviewer says pass",
                create_time_ms=base_time - 1000,
                source_id="user1",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Second reviewer also says pass",
                create_time_ms=base_time,
                source_id="user2",
            ),
        ]

        trace = _create_trace_with_assessments("test_no_conflict_2", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Should return a list of 2 examples
        assert isinstance(results, list)
        assert len(results) == 2

        # Both should have the same result value
        for result in results:
            assert isinstance(result, dspy.Example)
            assert result["result"] == "pass"

        # But different rationales
        rationales = {result["rationale"] for result in results}
        assert "First reviewer says pass" in rationales
        assert "Second reviewer also says pass" in rationales

    def test_three_assessments_same_label_returns_all(self, mock_judge):
        """When three assessments agree, all three should be returned."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Reviewer 1",
                create_time_ms=base_time - 2000,
                source_id="user1",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Reviewer 2",
                create_time_ms=base_time - 1000,
                source_id="user2",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Reviewer 3",
                create_time_ms=base_time,
                source_id="user3",
            ),
        ]

        trace = _create_trace_with_assessments("test_no_conflict_3", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        assert isinstance(results, list)
        assert len(results) == 3

        for result in results:
            assert result["result"] == "fail"

    def test_single_assessment_returns_single_item_list(self, mock_judge):
        """Single assessment should return a list with one example (backwards compatible)."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Only reviewer",
                create_time_ms=int(time.time() * 1000),
            ),
        ]

        trace = _create_trace_with_assessments("test_single", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["result"] == "pass"
        assert results[0]["rationale"] == "Only reviewer"


class TestMultiAssessmentWithConflict:
    """Tests for multiple assessments with label conflicts."""

    def test_three_vs_one_uses_majority(self, mock_judge, capsys):
        """When 3 say 'pass' and 1 says 'fail', use the 3 'pass' assessments."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 1",
                create_time_ms=base_time - 3000,
                source_id="user1",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Fail 1 - should be discarded",
                create_time_ms=base_time - 2000,
                source_id="user2",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 2",
                create_time_ms=base_time - 1000,
                source_id="user3",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 3",
                create_time_ms=base_time,
                source_id="user4",
            ),
        ]

        trace = _create_trace_with_assessments("test_conflict_majority", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Should return 3 examples (the majority)
        assert isinstance(results, list)
        assert len(results) == 3

        # All should be 'pass'
        for result in results:
            assert result["result"] == "pass"

        # Discarded assessment's rationale should not be in results
        rationales = {result["rationale"] for result in results}
        assert "Fail 1 - should be discarded" not in rationales

        # Should have logged a warning about discarded assessment
        captured = capsys.readouterr()
        assert "discarded" in captured.err.lower()

    def test_two_vs_one_uses_majority(self, mock_judge, capsys):
        """When 2 say 'fail' and 1 says 'pass', use the 2 'fail' assessments."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass - minority",
                create_time_ms=base_time - 2000,
                source_id="user1",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Fail 1",
                create_time_ms=base_time - 1000,
                source_id="user2",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Fail 2",
                create_time_ms=base_time,
                source_id="user3",
            ),
        ]

        trace = _create_trace_with_assessments("test_conflict_2v1", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        assert isinstance(results, list)
        assert len(results) == 2

        for result in results:
            assert result["result"] == "fail"

        # Warning should mention discarded assessment
        captured = capsys.readouterr()
        assert "discarded" in captured.err.lower()

    def test_conflict_warning_includes_trace_id(self, mock_judge, capsys):
        """Warning message should include trace ID for debugging."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass",
                create_time_ms=base_time - 1000,
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 2",
                create_time_ms=base_time - 500,
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Fail",
                create_time_ms=base_time,
            ),
        ]

        trace = _create_trace_with_assessments("trace_id_for_warning_test", assessments)
        trace_to_dspy_example(trace, mock_judge)

        # Warning should mention the trace ID
        captured = capsys.readouterr()
        assert "trace_id_for_warning_test" in captured.err

    def test_conflict_warning_includes_discarded_assessment_details(self, mock_judge, capsys):
        """Warning message should include source_id and timestamp of discarded assessments."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        discarded_time = base_time - 1000
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 1",
                create_time_ms=base_time - 2000,
                source_id="reviewer_alice",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Fail - will be discarded",
                create_time_ms=discarded_time,
                source_id="reviewer_bob",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 2",
                create_time_ms=base_time,
                source_id="reviewer_charlie",
            ),
        ]

        trace = _create_trace_with_assessments("test_warning_details", assessments)
        trace_to_dspy_example(trace, mock_judge)

        # Warning should include details of the discarded assessment
        captured = capsys.readouterr()

        # Should mention the discarded assessment's source_id
        assert "reviewer_bob" in captured.err
        # Should mention the discarded assessment's timestamp
        assert str(discarded_time) in captured.err
        # Should mention the discarded label
        assert "fail" in captured.err.lower()

    def test_conflict_warning_includes_assessment_id_when_available(self, mock_judge, capsys):
        """Warning message should include assessment_id when it's available."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)

        # Create assessment with assessment_id set (simulating backend-generated ID)
        discarded_assessment = Feedback(
            name="mock_judge",
            value="fail",
            rationale="Fail - will be discarded",
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id="reviewer_bob"
            ),
            create_time_ms=base_time - 1000,
        )
        discarded_assessment.assessment_id = "assessment_abc123"

        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 1",
                create_time_ms=base_time - 2000,
                source_id="reviewer_alice",
            ),
            discarded_assessment,
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 2",
                create_time_ms=base_time,
                source_id="reviewer_charlie",
            ),
        ]

        trace = _create_trace_with_assessments("test_warning_with_id", assessments)
        trace_to_dspy_example(trace, mock_judge)

        # Should include the assessment_id in the warning
        captured = capsys.readouterr()
        assert "assessment_abc123" in captured.err


class TestMultiAssessmentTieBreaking:
    """Tests for tie-breaking when there's no clear majority."""

    def test_two_vs_two_tie_uses_most_recent(self, mock_judge, capsys):
        """When 2 say 'pass' and 2 say 'fail', use the most recent group."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            # Older group - 'pass'
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Old pass 1",
                create_time_ms=base_time - 4000,
                source_id="user1",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Old pass 2",
                create_time_ms=base_time - 3000,
                source_id="user2",
            ),
            # Newer group - 'fail' (should win tie)
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="New fail 1",
                create_time_ms=base_time - 2000,
                source_id="user3",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="New fail 2",
                create_time_ms=base_time - 1000,
                source_id="user4",
            ),
        ]

        trace = _create_trace_with_assessments("test_tie_2v2", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Should return 2 examples from the most recent group
        assert isinstance(results, list)
        assert len(results) == 2

        # All should be 'fail' (the more recent group)
        for result in results:
            assert result["result"] == "fail"

        # Warning should be logged for tie-breaking
        captured = capsys.readouterr()
        assert "tie" in captured.err.lower()

    def test_tie_warning_includes_discarded_assessment_details(self, mock_judge, capsys):
        """Tie-breaking warning should include details of all discarded assessments."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        discarded_time_1 = base_time - 4000
        discarded_time_2 = base_time - 3000

        assessments = [
            # Older group - 'pass' (will be discarded)
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Old pass 1",
                create_time_ms=discarded_time_1,
                source_id="alice",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Old pass 2",
                create_time_ms=discarded_time_2,
                source_id="bob",
            ),
            # Newer group - 'fail' (should win tie)
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="New fail 1",
                create_time_ms=base_time - 2000,
                source_id="charlie",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="New fail 2",
                create_time_ms=base_time - 1000,
                source_id="diana",
            ),
        ]

        trace = _create_trace_with_assessments("test_tie_details", assessments)
        trace_to_dspy_example(trace, mock_judge)

        # Should include details of both discarded assessments
        captured = capsys.readouterr()
        assert "alice" in captured.err
        assert "bob" in captured.err
        assert str(discarded_time_1) in captured.err
        assert str(discarded_time_2) in captured.err

    def test_three_way_tie_uses_most_recent(self, mock_judge):
        """When 1 'pass', 1 'fail', 1 'maybe', use the single most recent."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Oldest",
                create_time_ms=base_time - 2000,
                source_id="user1",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Middle",
                create_time_ms=base_time - 1000,
                source_id="user2",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="maybe",
                rationale="Most recent - should win",
                create_time_ms=base_time,
                source_id="user3",
            ),
        ]

        trace = _create_trace_with_assessments("test_tie_3way", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Should return 1 example (the most recent)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["result"] == "maybe"
        assert results[0]["rationale"] == "Most recent - should win"

    def test_tie_uses_most_recent_assessment_in_group(self, mock_judge):
        """
        In a tie, use the group whose most recent assessment is newer.

        Example: Group A has assessments at t=1, t=5
                 Group B has assessments at t=2, t=4
        Group A wins because its most recent (t=5) is newer than B's most recent (t=4).
        """
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            # Group 'pass' - most recent at base_time (wins)
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass early",
                create_time_ms=base_time - 5000,
                source_id="user1",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass late - wins tie",
                create_time_ms=base_time,  # Most recent overall
                source_id="user2",
            ),
            # Group 'fail' - most recent at base_time - 1000
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Fail early",
                create_time_ms=base_time - 4000,
                source_id="user3",
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Fail late",
                create_time_ms=base_time - 1000,
                source_id="user4",
            ),
        ]

        trace = _create_trace_with_assessments("test_tie_recency", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # 'pass' group should win because its most recent is newer
        assert len(results) == 2
        for result in results:
            assert result["result"] == "pass"


class TestMultiAssessmentEdgeCases:
    """Edge cases and special scenarios."""

    def test_assessments_without_timestamps_handled_gracefully(self, mock_judge):
        """Assessments without create_time_ms should still work."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        # Create assessments without timestamps (create_time_ms=None or 0)
        assessments = [
            Feedback(
                name="mock_judge",
                value="pass",
                rationale="No timestamp 1",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user1"
                ),
                # No create_time_ms specified
            ),
            Feedback(
                name="mock_judge",
                value="pass",
                rationale="No timestamp 2",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user2"
                ),
            ),
        ]

        trace = _create_trace_with_assessments("test_no_timestamps", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Should still return both examples
        assert isinstance(results, list)
        assert len(results) == 2

    def test_only_llm_assessments_returns_empty(self, mock_judge):
        """When only LLM assessments exist (no human), return empty list."""
        assessments = [
            Feedback(
                name="mock_judge",
                value="pass",
                rationale="LLM says pass",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"
                ),
            ),
        ]

        trace = _create_trace_with_assessments("test_llm_only", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Should return empty list (no human assessments)
        assert results == []

    def test_mixed_human_and_llm_only_uses_human(self, mock_judge):
        """Only human assessments should be considered, not LLM."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            # Human assessment
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Human says fail",
                create_time_ms=base_time - 1000,
            ),
            # LLM assessment (should be ignored)
            Feedback(
                name="mock_judge",
                value="pass",
                rationale="LLM says pass",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"
                ),
                create_time_ms=base_time,
            ),
            # Another human
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Another human says fail",
                create_time_ms=base_time - 500,
            ),
        ]

        trace = _create_trace_with_assessments("test_mixed_sources", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Should return 2 human examples, both 'fail'
        assert len(results) == 2
        for result in results:
            assert result["result"] == "fail"

    def test_different_judge_names_filtered(self, mock_judge):
        """Only assessments matching the judge name should be considered."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            # Assessment for our judge
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="For mock_judge",
                create_time_ms=base_time,
            ),
            # Assessment for different judge (should be ignored)
            _create_human_assessment(
                name="other_judge",
                value="fail",
                rationale="For other_judge",
                create_time_ms=base_time - 1000,
            ),
        ]

        trace = _create_trace_with_assessments("test_judge_filter", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Should only return 1 example for mock_judge
        assert len(results) == 1
        assert results[0]["result"] == "pass"

    def test_case_insensitive_judge_name_matching(self, mock_judge):
        """Judge name matching should be case-insensitive."""
        dspy = pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)
        assessments = [
            _create_human_assessment(
                name="MOCK_JUDGE",  # uppercase
                value="pass",
                rationale="Uppercase name",
                create_time_ms=base_time - 1000,
            ),
            _create_human_assessment(
                name="  Mock_Judge  ",  # with whitespace
                value="pass",
                rationale="With whitespace",
                create_time_ms=base_time,
            ),
        ]

        trace = _create_trace_with_assessments("test_case_insensitive", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        # Both should match mock_judge
        assert len(results) == 2

    def test_no_matching_assessments_returns_empty(self, mock_judge):
        """When no assessments match the judge, return empty list."""
        assessments = [
            _create_human_assessment(
                name="different_judge",
                value="pass",
                rationale="Wrong judge",
                create_time_ms=int(time.time() * 1000),
            ),
        ]

        trace = _create_trace_with_assessments("test_no_match", assessments)
        results = trace_to_dspy_example(trace, mock_judge)

        assert results == []

    def test_empty_assessments_returns_empty(self, mock_judge):
        """When trace has no assessments, return empty list."""
        trace = _create_trace_with_assessments("test_empty", [])
        results = trace_to_dspy_example(trace, mock_judge)

        assert results == []


class TestOptimizerMultiAssessmentIntegration:
    """Integration tests for MemAlignOptimizer handling multiple assessments per trace."""

    @pytest.fixture
    def optimizer(self):
        """Create a MemAlignOptimizer for testing."""
        from mlflow.genai.judges.optimizers.memalign.optimizer import MemAlignOptimizer

        return MemAlignOptimizer(
            reflection_lm="openai:/gpt-4o-mini",
            embedding_model="openai:/text-embedding-3-small",
        )

    def test_align_with_multi_assessment_trace_adds_all_examples(self, optimizer, mock_judge):
        """Optimizer.align() should add all examples from multi-assessment traces."""
        pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)

        # Create a trace with 3 agreeing assessments
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Reviewer 1",
                create_time_ms=base_time - 2000,
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Reviewer 2",
                create_time_ms=base_time - 1000,
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Reviewer 3",
                create_time_ms=base_time,
            ),
        ]
        trace = _create_trace_with_assessments("multi_assess_trace", assessments)

        # Mock the distill_guidelines to avoid LLM calls
        with mock.patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.distill_guidelines"
        ) as mock_distill:
            mock_distill.return_value = []

            aligned_judge = optimizer.align(mock_judge, [trace])

            # Should have 3 examples in episodic memory (one per assessment)
            assert len(aligned_judge._episodic_memory) == 3

            # All should have the same trace_id
            for example in aligned_judge._episodic_memory:
                assert example._trace_id == "multi_assess_trace"

    def test_align_with_conflict_only_adds_majority_examples(self, optimizer, mock_judge):
        """Optimizer.align() should only add majority examples when there's conflict."""
        pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)

        # Create a trace with 2 'pass' and 1 'fail' (majority is 'pass')
        assessments = [
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 1",
                create_time_ms=base_time - 2000,
            ),
            _create_human_assessment(
                name="mock_judge",
                value="fail",
                rationale="Fail - minority",
                create_time_ms=base_time - 1000,
            ),
            _create_human_assessment(
                name="mock_judge",
                value="pass",
                rationale="Pass 2",
                create_time_ms=base_time,
            ),
        ]
        trace = _create_trace_with_assessments("conflict_trace", assessments)

        with mock.patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.distill_guidelines"
        ) as mock_distill:
            mock_distill.return_value = []

            aligned_judge = optimizer.align(mock_judge, [trace])

            # Should have 2 examples (the majority 'pass' assessments)
            assert len(aligned_judge._episodic_memory) == 2

            # All should be 'pass'
            for example in aligned_judge._episodic_memory:
                assert example["result"] == "pass"

    def test_align_multiple_traces_with_different_assessment_counts(self, optimizer, mock_judge):
        """Optimizer should handle traces with varying numbers of assessments."""
        pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)

        # Trace 1: single assessment
        trace1 = _create_trace_with_assessments(
            "trace_1",
            [
                _create_human_assessment(
                    name="mock_judge",
                    value="pass",
                    rationale="Single",
                    create_time_ms=base_time,
                ),
            ],
        )

        # Trace 2: two agreeing assessments
        trace2 = _create_trace_with_assessments(
            "trace_2",
            [
                _create_human_assessment(
                    name="mock_judge",
                    value="fail",
                    rationale="Fail 1",
                    create_time_ms=base_time - 1000,
                ),
                _create_human_assessment(
                    name="mock_judge",
                    value="fail",
                    rationale="Fail 2",
                    create_time_ms=base_time,
                ),
            ],
        )

        # Trace 3: three assessments with conflict (2 pass, 1 fail)
        trace3 = _create_trace_with_assessments(
            "trace_3",
            [
                _create_human_assessment(
                    name="mock_judge",
                    value="pass",
                    rationale="Pass 1",
                    create_time_ms=base_time - 2000,
                ),
                _create_human_assessment(
                    name="mock_judge",
                    value="fail",
                    rationale="Fail - discarded",
                    create_time_ms=base_time - 1000,
                ),
                _create_human_assessment(
                    name="mock_judge",
                    value="pass",
                    rationale="Pass 2",
                    create_time_ms=base_time,
                ),
            ],
        )

        with mock.patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.distill_guidelines"
        ) as mock_distill:
            mock_distill.return_value = []

            aligned_judge = optimizer.align(mock_judge, [trace1, trace2, trace3])

            # Total examples: 1 (trace1) + 2 (trace2) + 2 (trace3, majority only) = 5
            assert len(aligned_judge._episodic_memory) == 5

            # Verify trace_ids are preserved
            trace_ids = [ex._trace_id for ex in aligned_judge._episodic_memory]
            assert trace_ids.count("trace_1") == 1
            assert trace_ids.count("trace_2") == 2
            assert trace_ids.count("trace_3") == 2

    def test_episodic_trace_ids_includes_all_assessments(self, optimizer, mock_judge):
        """_episodic_trace_ids should correctly track trace IDs for multi-assessment traces."""
        pytest.importorskip("dspy", reason="DSPy not installed")

        base_time = int(time.time() * 1000)

        # Create trace with 2 assessments
        trace = _create_trace_with_assessments(
            "multi_id_trace",
            [
                _create_human_assessment(
                    name="mock_judge",
                    value="pass",
                    rationale="R1",
                    create_time_ms=base_time - 1000,
                ),
                _create_human_assessment(
                    name="mock_judge",
                    value="pass",
                    rationale="R2",
                    create_time_ms=base_time,
                ),
            ],
        )

        with mock.patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.distill_guidelines"
        ) as mock_distill:
            mock_distill.return_value = []

            aligned_judge = optimizer.align(mock_judge, [trace])

            # _episodic_trace_ids should have 2 entries (one per example)
            # Both pointing to the same trace
            assert len(aligned_judge._episodic_trace_ids) == 2
            assert all(tid == "multi_id_trace" for tid in aligned_judge._episodic_trace_ids)
