"""
Utility functions for trace evaluation output formatting.
"""

from typing import Any

import pandas as pd


def build_output_data(
    df: pd.DataFrame,
    result_trace_ids: list[str],
    scorer_names: list[str],
    extract_assessment_from_column_fn,
    extract_assessment_from_assessments_column_fn,
) -> list[dict[str, Any]]:
    """
    Build the output data structure from evaluation results.

    Args:
        df: DataFrame containing evaluation results
        result_trace_ids: List of trace IDs in the results
        scorer_names: List of scorer names that were used
        extract_assessment_from_column_fn: Function to extract from standard columns
        extract_assessment_from_assessments_column_fn: Function to extract from assessments column

    Returns:
        List of dictionaries with trace_id and assessments
    """
    output_data = []

    for idx, trace_id in enumerate(result_trace_ids):
        trace_result = {"trace_id": trace_id, "assessments": []}

        # Extract assessments from the DataFrame for this row
        for scorer_name in scorer_names:
            normalized_scorer_name = scorer_name.lower().replace(" ", "_")

            # Try standard column format first
            assessment = extract_assessment_from_column_fn(
                df, idx, scorer_name, normalized_scorer_name
            )

            # If no result from columns, try assessments column
            if assessment["result"] is None and "assessments" in df.columns and idx < len(df):
                assessments_data = df["assessments"].iloc[idx]
                assess_from_col = extract_assessment_from_assessments_column_fn(
                    assessments_data, scorer_name, normalized_scorer_name
                )
                if assess_from_col:
                    assessment = assess_from_col

            trace_result["assessments"].append(assessment)

        output_data.append(trace_result)

    return output_data


def format_table_output(
    output_data: list[dict[str, Any]], scorer_names: list[str], format_error_message_fn
) -> tuple[list[str], list[list[str]]]:
    """
    Format evaluation results as table data.

    Args:
        output_data: List of trace results with assessments
        scorer_names: List of scorer names for column headers
        format_error_message_fn: Function to format error messages

    Returns:
        Tuple of (headers, table_data) where headers is a list of column names
        and table_data is a list of rows (each row is a list of cell values)
    """
    headers = ["trace_id"] + scorer_names
    table_data = []

    for trace_result in output_data:
        row = [trace_result["trace_id"]]

        for assessment in trace_result["assessments"]:
            cell_content = ""
            result_val = assessment.get("result")
            rationale_val = assessment.get("rationale")
            error_val = assessment.get("error")

            # Format as single line: "value: X, rationale: Y"
            if rationale_val and len(str(rationale_val)) > 100:
                rationale_str = str(rationale_val)[:97] + "..."
            else:
                rationale_str = rationale_val if rationale_val else None

            if result_val is not None and rationale_str:
                cell_content = f"value: {result_val}, rationale: {rationale_str}"
            elif result_val is not None:
                cell_content = f"value: {result_val}"
            elif rationale_str:
                cell_content = f"rationale: {rationale_str}"
            elif error_val:
                cell_content = format_error_message_fn(error_val)
            else:
                cell_content = "N/A"

            row.append(cell_content)

        table_data.append(row)

    return headers, table_data
