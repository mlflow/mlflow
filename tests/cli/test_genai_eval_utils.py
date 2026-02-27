from unittest import mock

import click
import pandas as pd
import pytest

from mlflow.cli.genai_eval_utils import (
    NA_VALUE,
    Assessment,
    EvalResult,
    extract_assessments_from_results,
    format_table_output,
    resolve_scorers,
)
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import AssessmentMetadataKey


def test_format_single_trace_with_result_and_rationale():
    output_data = [
        EvalResult(
            trace_id="tr-123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result="yes",
                    rationale="The answer is relevant",
                )
            ],
        )
    ]

    table_output = format_table_output(output_data)

    # Headers should use assessment names from output_data
    assert table_output.headers == ["trace_id", "RelevanceToQuery"]
    assert len(table_output.rows) == 1
    assert table_output.rows[0][0].value == "tr-123"
    assert "value: yes" in table_output.rows[0][1].value
    assert "rationale: The answer is relevant" in table_output.rows[0][1].value


def test_format_multiple_traces_multiple_scorers():
    output_data = [
        EvalResult(
            trace_id="tr-123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result="yes",
                    rationale="Relevant",
                ),
                Assessment(name="Safety", result="yes", rationale="Safe"),
            ],
        ),
        EvalResult(
            trace_id="tr-456",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result="no",
                    rationale="Not relevant",
                ),
                Assessment(name="Safety", result="yes", rationale="Safe"),
            ],
        ),
    ]

    table_output = format_table_output(output_data)

    # Assessment names should be sorted
    assert table_output.headers == ["trace_id", "RelevanceToQuery", "Safety"]
    assert len(table_output.rows) == 2
    assert table_output.rows[0][0].value == "tr-123"
    assert table_output.rows[1][0].value == "tr-456"
    assert "value: yes" in table_output.rows[0][1].value
    assert "value: no" in table_output.rows[1][1].value


def test_format_long_rationale_not_truncated():
    long_rationale = "x" * 150
    output_data = [
        EvalResult(
            trace_id="tr-123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result="yes",
                    rationale=long_rationale,
                )
            ],
        )
    ]

    table_output = format_table_output(output_data)

    assert long_rationale in table_output.rows[0][1].value
    assert len(table_output.rows[0][1].value) >= len(long_rationale)


def test_format_error_message_formatting():
    output_data = [
        EvalResult(
            trace_id="tr-123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result=None,
                    rationale=None,
                    error="OpenAI API error",
                )
            ],
        )
    ]

    table_output = format_table_output(output_data)

    assert table_output.rows[0][1].value == "error: OpenAI API error"


def test_format_na_for_missing_results():
    output_data = [
        EvalResult(
            trace_id="tr-123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result=None,
                    rationale=None,
                )
            ],
        )
    ]

    table_output = format_table_output(output_data)

    assert table_output.rows[0][1].value == NA_VALUE


def test_format_result_only_without_rationale():
    output_data = [
        EvalResult(
            trace_id="tr-123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result="yes",
                    rationale=None,
                )
            ],
        )
    ]

    table_output = format_table_output(output_data)

    assert table_output.rows[0][1].value == "value: yes"


def test_format_rationale_only_without_result():
    output_data = [
        EvalResult(
            trace_id="tr-123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result=None,
                    rationale="Some reasoning",
                )
            ],
        )
    ]

    table_output = format_table_output(output_data)

    assert table_output.rows[0][1].value == "rationale: Some reasoning"


def test_format_with_different_assessment_names():
    # This test demonstrates that assessment names (e.g., "relevance_to_query")
    # are used in headers, not scorer class names (e.g., "RelevanceToQuery")
    output_data = [
        EvalResult(
            trace_id="tr-123",
            assessments=[
                Assessment(
                    name="relevance_to_query",  # Different from scorer name
                    result="yes",
                    rationale="The answer is relevant",
                ),
                Assessment(
                    name="safety_check",  # Different from scorer name
                    result="safe",
                    rationale="Content is safe",
                ),
            ],
        )
    ]

    table_output = format_table_output(output_data)

    # Headers should use actual assessment names from output_data (sorted)
    assert table_output.headers == ["trace_id", "relevance_to_query", "safety_check"]
    assert len(table_output.rows) == 1
    assert table_output.rows[0][0].value == "tr-123"
    assert "value: yes" in table_output.rows[0][1].value
    assert "value: safe" in table_output.rows[0][2].value


# Tests for resolve_scorers function


def test_resolve_builtin_scorer():
    # Test with real built-in scorer names
    scorers = resolve_scorers(["Correctness"], "experiment_123")

    assert len(scorers) == 1
    assert scorers[0].__class__.__name__ == "Correctness"


def test_resolve_builtin_scorer_snake_case():
    # Test with snake_case name
    scorers = resolve_scorers(["correctness"], "experiment_123")

    assert len(scorers) == 1
    assert scorers[0].__class__.__name__ == "Correctness"


def test_resolve_registered_scorer():
    mock_registered = mock.Mock()

    with (
        mock.patch(
            "mlflow.cli.genai_eval_utils.get_all_scorers", return_value=[]
        ) as mock_get_all_scorers,
        mock.patch(
            "mlflow.cli.genai_eval_utils.get_scorer", return_value=mock_registered
        ) as mock_get_scorer,
    ):
        scorers = resolve_scorers(["CustomScorer"], "experiment_123")

        assert len(scorers) == 1
        assert scorers[0] == mock_registered
        # Verify mocks were called as expected
        mock_get_all_scorers.assert_called_once()
        mock_get_scorer.assert_called_once_with(name="CustomScorer", experiment_id="experiment_123")


def test_resolve_mixed_scorers():
    # Setup built-in scorer
    mock_builtin = mock.Mock()
    mock_builtin.__class__.__name__ = "Safety"
    mock_builtin.name = None

    # Setup registered scorer
    mock_registered = mock.Mock()

    with (
        mock.patch(
            "mlflow.cli.genai_eval_utils.get_all_scorers", return_value=[mock_builtin]
        ) as mock_get_all_scorers,
        mock.patch(
            "mlflow.cli.genai_eval_utils.get_scorer", return_value=mock_registered
        ) as mock_get_scorer,
    ):
        scorers = resolve_scorers(["Safety", "CustomScorer"], "experiment_123")

        assert len(scorers) == 2
        assert scorers[0] == mock_builtin
        assert scorers[1] == mock_registered
        # Verify mocks were called as expected
        mock_get_all_scorers.assert_called_once()
        mock_get_scorer.assert_called_once_with(name="CustomScorer", experiment_id="experiment_123")


def test_resolve_scorer_not_found_raises_error():
    with (
        mock.patch(
            "mlflow.cli.genai_eval_utils.get_all_scorers", return_value=[]
        ) as mock_get_all_scorers,
        mock.patch(
            "mlflow.cli.genai_eval_utils.get_scorer",
            side_effect=MlflowException("Not found"),
        ) as mock_get_scorer,
    ):
        with pytest.raises(click.UsageError, match="Could not identify Scorer 'UnknownScorer'"):
            resolve_scorers(["UnknownScorer"], "experiment_123")

        # Verify mocks were called as expected
        mock_get_all_scorers.assert_called_once()
        mock_get_scorer.assert_called_once_with(
            name="UnknownScorer", experiment_id="experiment_123"
        )


def test_resolve_empty_scorers_raises_error():
    with pytest.raises(click.UsageError, match="No valid scorers"):
        resolve_scorers([], "experiment_123")


# Tests for extract_assessments_from_results function


def test_extract_with_matching_run_id():
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "feedback": {"value": "yes"},
                        "rationale": "The answer is relevant",
                        "metadata": {AssessmentMetadataKey.SOURCE_RUN_ID: "run-123"},
                    }
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    expected = [
        EvalResult(
            trace_id="tr-abc123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result="yes",
                    rationale="The answer is relevant",
                )
            ],
        )
    ]
    assert result == expected


def test_extract_with_different_assessment_name():
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "relevance_to_query",
                        "feedback": {"value": "yes"},
                        "rationale": "Relevant answer",
                        "metadata": {AssessmentMetadataKey.SOURCE_RUN_ID: "run-123"},
                    }
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    expected = [
        EvalResult(
            trace_id="tr-abc123",
            assessments=[
                Assessment(
                    name="relevance_to_query",
                    result="yes",
                    rationale="Relevant answer",
                )
            ],
        )
    ]
    assert result == expected


def test_extract_filter_out_assessments_with_different_run_id():
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "feedback": {"value": "yes"},
                        "rationale": "Current evaluation",
                        "metadata": {AssessmentMetadataKey.SOURCE_RUN_ID: "run-123"},
                    },
                    {
                        "assessment_name": "Safety",
                        "feedback": {"value": "yes"},
                        "rationale": "Old evaluation",
                        "metadata": {AssessmentMetadataKey.SOURCE_RUN_ID: "run-456"},
                    },
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    expected = [
        EvalResult(
            trace_id="tr-abc123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result="yes",
                    rationale="Current evaluation",
                )
            ],
        )
    ]
    assert result == expected


def test_extract_no_assessments_for_run_id():
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "metadata": {AssessmentMetadataKey.SOURCE_RUN_ID: "run-456"},
                    }
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    assert len(result) == 1
    assert len(result[0].assessments) == 1
    assert result[0].assessments[0].result is None
    assert result[0].assessments[0].rationale is None
    assert result[0].assessments[0].error is not None


def test_extract_multiple_assessments_from_same_run():
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "feedback": {"value": "yes"},
                        "rationale": "Relevant",
                        "metadata": {AssessmentMetadataKey.SOURCE_RUN_ID: "run-123"},
                    },
                    {
                        "assessment_name": "Safety",
                        "feedback": {"value": "yes"},
                        "rationale": "Safe",
                        "metadata": {AssessmentMetadataKey.SOURCE_RUN_ID: "run-123"},
                    },
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    expected = [
        EvalResult(
            trace_id="tr-abc123",
            assessments=[
                Assessment(
                    name="RelevanceToQuery",
                    result="yes",
                    rationale="Relevant",
                ),
                Assessment(
                    name="Safety",
                    result="yes",
                    rationale="Safe",
                ),
            ],
        )
    ]
    assert result == expected


def test_extract_no_assessments_on_trace_shows_error():
    results_df = pd.DataFrame([{"trace_id": "tr-abc123", "assessments": []}])

    result = extract_assessments_from_results(results_df, "run-123")

    assert len(result) == 1
    assert len(result[0].assessments) == 1
    assert result[0].assessments[0].error == "No assessments found on trace"
