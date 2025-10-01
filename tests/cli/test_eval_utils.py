"""Tests for mlflow.cli.eval_utils module."""

from unittest import mock

import click
import pandas as pd
import pytest

from mlflow.cli.eval_utils import (
    extract_assessments_from_results,
    format_table_output,
    resolve_scorers,
)
from mlflow.exceptions import MlflowException

# Tests for format_table_output function


def test_format_single_trace_with_result_and_rationale():
    """Test formatting a single trace with result and rationale."""
    output_data = [
        {
            "trace_id": "tr-123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": "yes",
                    "rationale": "The answer is relevant",
                }
            ],
        }
    ]

    headers, table_data = format_table_output(output_data)

    # Headers should use assessment names from output_data
    assert headers == ["trace_id", "RelevanceToQuery"]
    assert len(table_data) == 1
    assert table_data[0][0] == "tr-123"
    assert "value: yes" in table_data[0][1]
    assert "rationale: The answer is relevant" in table_data[0][1]


def test_format_multiple_traces_multiple_scorers():
    """Test formatting multiple traces with multiple scorers."""
    output_data = [
        {
            "trace_id": "tr-123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": "yes",
                    "rationale": "Relevant",
                },
                {"assessment_name": "Safety", "result": "yes", "rationale": "Safe"},
            ],
        },
        {
            "trace_id": "tr-456",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": "no",
                    "rationale": "Not relevant",
                },
                {"assessment_name": "Safety", "result": "yes", "rationale": "Safe"},
            ],
        },
    ]

    headers, table_data = format_table_output(output_data)

    assert headers == ["trace_id", "RelevanceToQuery", "Safety"]
    assert len(table_data) == 2
    assert table_data[0][0] == "tr-123"
    assert table_data[1][0] == "tr-456"
    assert "value: yes" in table_data[0][1]
    assert "value: no" in table_data[1][1]


def test_format_long_rationale_not_truncated():
    """Test that long rationales are displayed in full."""
    long_rationale = "x" * 150
    output_data = [
        {
            "trace_id": "tr-123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": "yes",
                    "rationale": long_rationale,
                }
            ],
        }
    ]

    headers, table_data = format_table_output(output_data)

    assert long_rationale in table_data[0][1]
    assert len(table_data[0][1]) >= len(long_rationale)


def test_format_error_message_formatting():
    """Test that error messages are formatted correctly."""
    output_data = [
        {
            "trace_id": "tr-123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": None,
                    "rationale": None,
                    "error": "OpenAI API error",
                }
            ],
        }
    ]

    headers, table_data = format_table_output(output_data)

    assert table_data[0][1] == "error: OpenAI API error"


def test_format_na_for_missing_results():
    """Test that N/A is shown when no result, rationale, or error."""
    output_data = [
        {
            "trace_id": "tr-123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": None,
                    "rationale": None,
                }
            ],
        }
    ]

    headers, table_data = format_table_output(output_data)

    assert table_data[0][1] == "N/A"


def test_format_result_only_without_rationale():
    """Test formatting when only result is present without rationale."""
    output_data = [
        {
            "trace_id": "tr-123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": "yes",
                    "rationale": None,
                }
            ],
        }
    ]

    headers, table_data = format_table_output(output_data)

    assert table_data[0][1] == "value: yes"


def test_format_rationale_only_without_result():
    """Test formatting when only rationale is present without result."""
    output_data = [
        {
            "trace_id": "tr-123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": None,
                    "rationale": "Some reasoning",
                }
            ],
        }
    ]

    headers, table_data = format_table_output(output_data)

    assert table_data[0][1] == "rationale: Some reasoning"


def test_format_with_different_assessment_names():
    """Test that assessment names from output_data are used, not scorer names."""
    # This test demonstrates that assessment names (e.g., "relevance_to_query")
    # are used in headers, not scorer class names (e.g., "RelevanceToQuery")
    output_data = [
        {
            "trace_id": "tr-123",
            "assessments": [
                {
                    "assessment_name": "relevance_to_query",  # Different from scorer name
                    "result": "yes",
                    "rationale": "The answer is relevant",
                },
                {
                    "assessment_name": "safety_check",  # Different from scorer name
                    "result": "safe",
                    "rationale": "Content is safe",
                },
            ],
        }
    ]

    headers, table_data = format_table_output(output_data)

    # Headers should use actual assessment names from output_data
    assert headers == ["trace_id", "relevance_to_query", "safety_check"]
    assert len(table_data) == 1
    assert table_data[0][0] == "tr-123"
    assert "value: yes" in table_data[0][1]
    assert "value: safe" in table_data[0][2]


# Tests for resolve_scorers function


@mock.patch("mlflow.cli.eval_utils.get_all_scorers")
def test_resolve_builtin_scorer(mock_get_all_scorers):
    """Test resolving a built-in scorer."""
    # Create a mock scorer object
    mock_scorer = mock.Mock()
    mock_scorer.__class__.__name__ = "RelevanceToQuery"
    mock_get_all_scorers.return_value = [mock_scorer]

    scorers = resolve_scorers(["RelevanceToQuery"], "experiment_123")

    assert len(scorers) == 1
    assert scorers[0] == mock_scorer


@mock.patch("mlflow.cli.eval_utils.get_scorer")
@mock.patch("mlflow.cli.eval_utils.get_all_scorers")
def test_resolve_registered_scorer(mock_get_all_scorers, mock_get_scorer):
    """Test resolving a registered scorer when not found in built-ins."""
    mock_get_all_scorers.return_value = []  # No built-in scorers
    mock_registered = mock.Mock()
    mock_get_scorer.return_value = mock_registered

    scorers = resolve_scorers(["CustomScorer"], "experiment_123")

    assert len(scorers) == 1
    assert scorers[0] == mock_registered


@mock.patch("mlflow.cli.eval_utils.get_scorer")
@mock.patch("mlflow.cli.eval_utils.get_all_scorers")
def test_resolve_mixed_scorers(mock_get_all_scorers, mock_get_scorer):
    """Test resolving a mix of built-in and registered scorers."""
    # Setup built-in scorer
    mock_builtin = mock.Mock()
    mock_builtin.__class__.__name__ = "Safety"
    mock_get_all_scorers.return_value = [mock_builtin]

    # Setup registered scorer
    mock_registered = mock.Mock()
    mock_get_scorer.return_value = mock_registered

    scorers = resolve_scorers(["Safety", "CustomScorer"], "experiment_123")

    assert len(scorers) == 2
    assert scorers[0] == mock_builtin
    assert scorers[1] == mock_registered


@mock.patch("mlflow.cli.eval_utils.get_scorer")
@mock.patch("mlflow.cli.eval_utils.get_all_scorers")
def test_resolve_scorer_not_found_raises_error(mock_get_all_scorers, mock_get_scorer):
    """Test that appropriate error is raised when scorer not found."""
    mock_get_all_scorers.return_value = []
    mock_get_scorer.side_effect = MlflowException("Not found")

    with pytest.raises(click.UsageError, match="Scorer 'UnknownScorer' not found"):
        resolve_scorers(["UnknownScorer"], "experiment_123")


@mock.patch("mlflow.cli.eval_utils.get_all_scorers")
def test_resolve_empty_scorers_raises_error(mock_get_all_scorers):
    """Test that error is raised when no scorers specified."""
    with pytest.raises(click.UsageError, match="No valid scorers"):
        resolve_scorers([], "experiment_123")


# Tests for extract_assessments_from_results function


def test_extract_with_matching_run_id():
    """Test extracting assessments that match the evaluation run_id."""
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "feedback": {"value": "yes"},
                        "rationale": "The answer is relevant",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-123"},
                    }
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    expected = [
        {
            "trace_id": "tr-abc123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": "yes",
                    "rationale": "The answer is relevant",
                }
            ],
        }
    ]
    assert result == expected


def test_extract_with_different_assessment_name():
    """Test that assessment name is preserved as-is from results."""
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "relevance_to_query",
                        "feedback": {"value": "yes"},
                        "rationale": "Relevant answer",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-123"},
                    }
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    expected = [
        {
            "trace_id": "tr-abc123",
            "assessments": [
                {
                    "assessment_name": "relevance_to_query",
                    "result": "yes",
                    "rationale": "Relevant answer",
                }
            ],
        }
    ]
    assert result == expected


def test_extract_filter_out_assessments_with_different_run_id():
    """Test that assessments with different run_id are filtered out."""
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "feedback": {"value": "yes"},
                        "rationale": "Current evaluation",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-123"},
                    },
                    {
                        "assessment_name": "Safety",
                        "feedback": {"value": "yes"},
                        "rationale": "Old evaluation",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-456"},
                    },
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    expected = [
        {
            "trace_id": "tr-abc123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": "yes",
                    "rationale": "Current evaluation",
                }
            ],
        }
    ]
    assert result == expected


def test_extract_no_assessments_for_run_id():
    """Test handling when no assessments match the run_id."""
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-456"},
                    }
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    assert len(result) == 1
    assert len(result[0]["assessments"]) == 1
    assert result[0]["assessments"][0]["result"] is None
    assert result[0]["assessments"][0]["rationale"] is None
    assert "error" in result[0]["assessments"][0]


def test_extract_multiple_assessments_from_same_run():
    """Test extracting multiple assessments from the same evaluation run."""
    results_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-abc123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "feedback": {"value": "yes"},
                        "rationale": "Relevant",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-123"},
                    },
                    {
                        "assessment_name": "Safety",
                        "feedback": {"value": "yes"},
                        "rationale": "Safe",
                        "metadata": {"mlflow.assessment.sourceRunId": "run-123"},
                    },
                ],
            }
        ]
    )

    result = extract_assessments_from_results(results_df, "run-123")

    expected = [
        {
            "trace_id": "tr-abc123",
            "assessments": [
                {
                    "assessment_name": "RelevanceToQuery",
                    "result": "yes",
                    "rationale": "Relevant",
                },
                {
                    "assessment_name": "Safety",
                    "result": "yes",
                    "rationale": "Safe",
                },
            ],
        }
    ]
    assert result == expected


def test_extract_no_assessments_on_trace_shows_error():
    """Test that error is shown when trace has no assessments at all."""
    results_df = pd.DataFrame([{"trace_id": "tr-abc123", "assessments": []}])

    result = extract_assessments_from_results(results_df, "run-123")

    assert len(result) == 1
    assert len(result[0]["assessments"]) == 1
    assert result[0]["assessments"][0]["error"] == "No assessments found on trace"
