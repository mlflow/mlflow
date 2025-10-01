"""Tests for mlflow.cli.eval_utils module."""

import pandas as pd

from mlflow.cli.eval_utils import build_output_data, format_table_output


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
