"""Tests for mlflow.cli.eval_utils module."""

from unittest import mock

import click
import pandas as pd
import pytest

from mlflow.cli.eval_utils import (
    build_output_data,
    extract_assessments_from_traces,
    format_table_output,
    resolve_scorers,
)
from mlflow.exceptions import MlflowException


def mock_extract_from_column(df, idx, scorer_name, normalized_scorer_name):
    """Mock function for extracting assessment from standard columns."""
    # Simulate extracting from standard columns
    # Note: The normalized name uses underscores and lowercase
    value_col = f"{normalized_scorer_name}/value"
    rationale_col = f"{normalized_scorer_name}/rationale"

    assessment = {
        "assessment_name": scorer_name,
        "result": None,
        "rationale": None,
    }

    if value_col in df.columns and idx < len(df):
        value = df[value_col].iloc[idx]
        if pd.notna(value):
            assessment["result"] = value

    if rationale_col in df.columns and idx < len(df):
        rationale = df[rationale_col].iloc[idx]
        if pd.notna(rationale):
            assessment["rationale"] = str(rationale)

    return assessment


def mock_extract_from_assessments_column(assessments_data, scorer_name, normalized_scorer_name):
    """Mock function for extracting from assessments column."""
    if not assessments_data:
        return None

    for assess in assessments_data:
        if assess.get("assessment_name") == scorer_name:
            return {
                "assessment_name": scorer_name,
                "result": assess.get("result"),
                "rationale": assess.get("rationale"),
            }
    return None


def mock_format_error_message(error_msg):
    """Mock function for formatting error messages."""
    if "OpenAI" in error_msg:
        return "ERROR: Missing OpenAI API key"
    return f"ERROR: {error_msg[:50]}..."


class TestBuildOutputData:
    """Tests for build_output_data function."""

    def test_single_trace_single_scorer_with_standard_columns(self):
        """Test with single trace and scorer using standard result columns."""
        df = pd.DataFrame(
            {
                "trace_id": ["tr-123"],
                "relevancetoquery/value": ["yes"],  # normalized: relevancetoquery
                "relevancetoquery/rationale": ["The answer is relevant"],
            }
        )
        result_trace_ids = ["tr-123"]
        scorer_names = ["RelevanceToQuery"]

        output_data = build_output_data(
            df,
            result_trace_ids,
            scorer_names,
            mock_extract_from_column,
            mock_extract_from_assessments_column,
        )

        assert len(output_data) == 1
        assert output_data[0]["trace_id"] == "tr-123"
        assert len(output_data[0]["assessments"]) == 1
        assert output_data[0]["assessments"][0]["assessment_name"] == "RelevanceToQuery"
        assert output_data[0]["assessments"][0]["result"] == "yes"
        assert output_data[0]["assessments"][0]["rationale"] == "The answer is relevant"

    def test_multiple_traces_multiple_scorers(self):
        """Test with multiple traces and scorers."""
        df = pd.DataFrame(
            {
                "trace_id": ["tr-123", "tr-456"],
                "relevancetoquery/value": ["yes", "no"],  # normalized names
                "relevancetoquery/rationale": ["Relevant", "Not relevant"],
                "safety/value": ["yes", "yes"],
                "safety/rationale": ["Safe content", "Safe content"],
            }
        )
        result_trace_ids = ["tr-123", "tr-456"]
        scorer_names = ["RelevanceToQuery", "Safety"]

        output_data = build_output_data(
            df,
            result_trace_ids,
            scorer_names,
            mock_extract_from_column,
            mock_extract_from_assessments_column,
        )

        assert len(output_data) == 2
        assert output_data[0]["trace_id"] == "tr-123"
        assert output_data[1]["trace_id"] == "tr-456"
        assert len(output_data[0]["assessments"]) == 2
        assert len(output_data[1]["assessments"]) == 2

        # Check first trace
        assert output_data[0]["assessments"][0]["result"] == "yes"
        assert output_data[0]["assessments"][1]["result"] == "yes"

        # Check second trace
        assert output_data[1]["assessments"][0]["result"] == "no"
        assert output_data[1]["assessments"][1]["result"] == "yes"

    def test_with_assessments_column_fallback(self):
        """Test fallback to assessments column when standard columns don't have results."""
        df = pd.DataFrame(
            {
                "trace_id": ["tr-123"],
                "assessments": [
                    [
                        {
                            "assessment_name": "RelevanceToQuery",
                            "result": "yes",
                            "rationale": "From assessments column",
                        }
                    ]
                ],
            }
        )
        result_trace_ids = ["tr-123"]
        scorer_names = ["RelevanceToQuery"]

        output_data = build_output_data(
            df,
            result_trace_ids,
            scorer_names,
            mock_extract_from_column,
            mock_extract_from_assessments_column,
        )

        assert len(output_data) == 1
        assert output_data[0]["assessments"][0]["result"] == "yes"
        assert output_data[0]["assessments"][0]["rationale"] == "From assessments column"


class TestFormatTableOutput:
    """Tests for format_table_output function."""

    def test_single_trace_with_result_and_rationale(self):
        """Test formatting a single trace with result and rationale."""
        output_data = [
            {
                "trace_id": "tr-123",
                "assessments": [
                    {
                        "assessment_name": "RelevanceToQuery",
                        "result": "yes",
                        "rationale": "The answer is relevant to the question",
                    }
                ],
            }
        ]
        scorer_names = ["RelevanceToQuery"]

        headers, table_data = format_table_output(
            output_data, scorer_names, mock_format_error_message
        )

        assert headers == ["trace_id", "RelevanceToQuery"]
        assert len(table_data) == 1
        assert table_data[0][0] == "tr-123"
        assert "value: yes" in table_data[0][1]
        assert "rationale: The answer is relevant to the question" in table_data[0][1]

    def test_multiple_traces_multiple_scorers(self):
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
        scorer_names = ["RelevanceToQuery", "Safety"]

        headers, table_data = format_table_output(
            output_data, scorer_names, mock_format_error_message
        )

        assert headers == ["trace_id", "RelevanceToQuery", "Safety"]
        assert len(table_data) == 2
        assert table_data[0][0] == "tr-123"
        assert table_data[1][0] == "tr-456"
        assert "value: yes" in table_data[0][1]
        assert "value: no" in table_data[1][1]

    def test_long_rationale_truncation(self):
        """Test that long rationales are truncated."""
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
        scorer_names = ["RelevanceToQuery"]

        headers, table_data = format_table_output(
            output_data, scorer_names, mock_format_error_message
        )

        assert "..." in table_data[0][1]
        # The cell content should be less than the original rationale length
        assert len(table_data[0][1]) < len(long_rationale) + 20

    def test_error_message_formatting(self):
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
        scorer_names = ["RelevanceToQuery"]

        headers, table_data = format_table_output(
            output_data, scorer_names, mock_format_error_message
        )

        assert table_data[0][1] == "ERROR: Missing OpenAI API key"

    def test_na_for_missing_results(self):
        """Test that N/A is shown when no result, rationale, or error."""
        output_data = [
            {
                "trace_id": "tr-123",
                "assessments": [
                    {"assessment_name": "RelevanceToQuery", "result": None, "rationale": None}
                ],
            }
        ]
        scorer_names = ["RelevanceToQuery"]

        headers, table_data = format_table_output(
            output_data, scorer_names, mock_format_error_message
        )

        assert table_data[0][1] == "N/A"

    def test_result_only_without_rationale(self):
        """Test formatting when only result is present without rationale."""
        output_data = [
            {
                "trace_id": "tr-123",
                "assessments": [
                    {"assessment_name": "RelevanceToQuery", "result": "yes", "rationale": None}
                ],
            }
        ]
        scorer_names = ["RelevanceToQuery"]

        headers, table_data = format_table_output(
            output_data, scorer_names, mock_format_error_message
        )

        assert table_data[0][1] == "value: yes"

    def test_rationale_only_without_result(self):
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
        scorer_names = ["RelevanceToQuery"]

        headers, table_data = format_table_output(
            output_data, scorer_names, mock_format_error_message
        )

        assert table_data[0][1] == "rationale: Some reasoning"


class TestResolveScorers:
    """Tests for resolve_scorers function."""

    @mock.patch("mlflow.cli.eval_utils.get_all_scorers")
    def test_resolve_builtin_scorer(self, mock_get_all_scorers):
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
    def test_resolve_registered_scorer(self, mock_get_all_scorers, mock_get_scorer):
        """Test resolving a registered scorer."""
        mock_get_all_scorers.return_value = []  # No built-in scorers
        mock_registered = mock.Mock()
        mock_get_scorer.return_value = mock_registered

        scorers = resolve_scorers(["CustomScorer"], "experiment_123")

        assert len(scorers) == 1
        assert scorers[0] == mock_registered
        mock_get_scorer.assert_called_once_with(name="CustomScorer", experiment_id="experiment_123")

    @mock.patch("mlflow.cli.eval_utils.get_scorer")
    @mock.patch("mlflow.cli.eval_utils.get_all_scorers")
    def test_resolve_mixed_scorers(self, mock_get_all_scorers, mock_get_scorer):
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
    def test_scorer_not_found_raises_error(self, mock_get_all_scorers, mock_get_scorer):
        """Test that non-existent scorer raises appropriate error."""
        mock_builtin = mock.Mock()
        mock_builtin.__class__.__name__ = "Safety"
        mock_get_all_scorers.return_value = [mock_builtin]
        mock_get_scorer.side_effect = MlflowException("Scorer not found")

        with pytest.raises(click.UsageError, match="NonExistentScorer.*not found"):
            resolve_scorers(["NonExistentScorer"], "experiment_123")

    @mock.patch("mlflow.cli.eval_utils.get_all_scorers")
    def test_empty_scorers_raises_error(self, mock_get_all_scorers):
        """Test that empty scorer list raises error."""
        mock_get_all_scorers.return_value = []

        with pytest.raises(click.UsageError, match="No valid scorers"):
            resolve_scorers([], "experiment_123")


class TestExtractAssessmentsFromTraces:
    """Tests for extract_assessments_from_traces function."""

    @mock.patch("mlflow.cli.eval_utils.MlflowClient")
    def test_extract_with_matching_run_id(self, mock_client_class):
        """Test extracting assessments that match the evaluation run_id."""
        # Create mock trace with assessments
        mock_trace = mock.Mock()
        mock_assessment1 = mock.Mock()
        mock_assessment1.name = "RelevanceToQuery"
        mock_assessment1.run_id = "run-123"
        mock_assessment1.feedback = mock.Mock(value="yes")
        mock_assessment1.rationale = "The answer is relevant"
        mock_assessment1.error = None

        mock_trace.info.assessments = [mock_assessment1]

        # Setup mock client
        mock_client = mock.Mock()
        mock_client.get_trace.return_value = mock_trace
        mock_client_class.return_value = mock_client

        # Call function
        result = extract_assessments_from_traces(["tr-abc123"], ["RelevanceToQuery"], "run-123")

        # Verify
        assert len(result) == 1
        assert result[0]["trace_id"] == "tr-abc123"
        assert len(result[0]["assessments"]) == 1
        assert result[0]["assessments"][0]["assessment_name"] == "RelevanceToQuery"
        assert result[0]["assessments"][0]["result"] == "yes"
        assert result[0]["assessments"][0]["rationale"] == "The answer is relevant"

    @mock.patch("mlflow.cli.eval_utils.MlflowClient")
    def test_extract_with_different_assessment_name(self, mock_client_class):
        """Test that assessment name can differ from scorer name (matched by run_id)."""
        # Create mock trace with assessment that has different name than scorer
        mock_trace = mock.Mock()
        mock_assessment1 = mock.Mock()
        mock_assessment1.name = "relevance_to_query"  # Different from scorer name
        mock_assessment1.run_id = "run-123"
        mock_assessment1.feedback = mock.Mock(value="yes")
        mock_assessment1.rationale = "Relevant answer"
        mock_assessment1.error = None

        mock_trace.info.assessments = [mock_assessment1]

        # Setup mock client
        mock_client = mock.Mock()
        mock_client.get_trace.return_value = mock_trace
        mock_client_class.return_value = mock_client

        # Call function with different scorer name
        result = extract_assessments_from_traces(["tr-abc123"], ["RelevanceToQuery"], "run-123")

        # Verify - assessment should still be extracted because run_id matches
        assert len(result) == 1
        assert result[0]["trace_id"] == "tr-abc123"
        assert len(result[0]["assessments"]) == 1
        # Assessment name from trace is used, not scorer name
        assert result[0]["assessments"][0]["assessment_name"] == "relevance_to_query"
        assert result[0]["assessments"][0]["result"] == "yes"

    @mock.patch("mlflow.cli.eval_utils.MlflowClient")
    def test_filter_out_assessments_with_different_run_id(self, mock_client_class):
        """Test that assessments with different run_id are filtered out."""
        # Create mock trace with assessments from different runs
        mock_trace = mock.Mock()
        mock_assessment1 = mock.Mock()
        mock_assessment1.name = "RelevanceToQuery"
        mock_assessment1.run_id = "run-123"
        mock_assessment1.feedback = mock.Mock(value="yes")
        mock_assessment1.rationale = "Current evaluation"
        mock_assessment1.error = None

        mock_assessment2 = mock.Mock()
        mock_assessment2.name = "Safety"
        mock_assessment2.run_id = "run-456"  # Different run
        mock_assessment2.feedback = mock.Mock(value="yes")
        mock_assessment2.rationale = "Old evaluation"
        mock_assessment2.error = None

        mock_trace.info.assessments = [mock_assessment1, mock_assessment2]

        # Setup mock client
        mock_client = mock.Mock()
        mock_client.get_trace.return_value = mock_trace
        mock_client_class.return_value = mock_client

        # Call function
        result = extract_assessments_from_traces(
            ["tr-abc123"], ["RelevanceToQuery", "Safety"], "run-123"
        )

        # Verify - only assessment from run-123 should be included
        assert len(result) == 1
        assert len(result[0]["assessments"]) == 1
        assert result[0]["assessments"][0]["assessment_name"] == "RelevanceToQuery"
        assert result[0]["assessments"][0]["result"] == "yes"

    @mock.patch("mlflow.cli.eval_utils.MlflowClient")
    def test_no_assessments_for_run_id(self, mock_client_class):
        """Test handling when no assessments match the run_id."""
        # Create mock trace with assessments from different run
        mock_trace = mock.Mock()
        mock_assessment1 = mock.Mock()
        mock_assessment1.run_id = "run-456"  # Different run

        mock_trace.info.assessments = [mock_assessment1]

        # Setup mock client
        mock_client = mock.Mock()
        mock_client.get_trace.return_value = mock_trace
        mock_client_class.return_value = mock_client

        # Call function
        result = extract_assessments_from_traces(["tr-abc123"], ["RelevanceToQuery"], "run-123")

        # Verify - should return empty results with scorer names
        assert len(result) == 1
        assert len(result[0]["assessments"]) == 1
        assert result[0]["assessments"][0]["assessment_name"] == "RelevanceToQuery"
        assert result[0]["assessments"][0]["result"] is None
        assert result[0]["assessments"][0]["rationale"] is None

    @mock.patch("mlflow.cli.eval_utils.MlflowClient")
    def test_multiple_assessments_from_same_run(self, mock_client_class):
        """Test extracting multiple assessments from the same evaluation run."""
        # Create mock trace with multiple assessments from same run
        mock_trace = mock.Mock()
        mock_assessment1 = mock.Mock()
        mock_assessment1.name = "RelevanceToQuery"
        mock_assessment1.run_id = "run-123"
        mock_assessment1.feedback = mock.Mock(value="yes")
        mock_assessment1.rationale = "Relevant"
        mock_assessment1.error = None

        mock_assessment2 = mock.Mock()
        mock_assessment2.name = "Safety"
        mock_assessment2.run_id = "run-123"
        mock_assessment2.feedback = mock.Mock(value="yes")
        mock_assessment2.rationale = "Safe"
        mock_assessment2.error = None

        mock_trace.info.assessments = [mock_assessment1, mock_assessment2]

        # Setup mock client
        mock_client = mock.Mock()
        mock_client.get_trace.return_value = mock_trace
        mock_client_class.return_value = mock_client

        # Call function
        result = extract_assessments_from_traces(
            ["tr-abc123"], ["RelevanceToQuery", "Safety"], "run-123"
        )

        # Verify - both assessments should be included
        assert len(result) == 1
        assert len(result[0]["assessments"]) == 2
        assert result[0]["assessments"][0]["assessment_name"] == "RelevanceToQuery"
        assert result[0]["assessments"][1]["assessment_name"] == "Safety"
