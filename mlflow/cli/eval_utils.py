"""
Utility functions for trace evaluation output formatting.
"""

from typing import Any

import click
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import get_all_scorers, get_scorer
from mlflow.tracking import MlflowClient


def resolve_scorers(scorer_names: list[str], experiment_id: str) -> list:
    """
    Resolve scorer names to scorer objects.

    Checks built-in scorers first, then registered scorers.

    Args:
        scorer_names: List of scorer names to resolve
        experiment_id: Experiment ID for looking up registered scorers

    Returns:
        List of resolved scorer objects

    Raises:
        click.UsageError: If a scorer is not found or no valid scorers specified
    """
    resolved_scorers = []
    builtin_scorers = get_all_scorers()
    builtin_scorer_map = {scorer.__class__.__name__: scorer for scorer in builtin_scorers}

    for scorer_name in scorer_names:
        # Check if it's a built-in scorer
        if scorer_name in builtin_scorer_map:
            resolved_scorers.append(builtin_scorer_map[scorer_name])
        else:
            # Try to get it as a registered scorer
            try:
                registered_scorer = get_scorer(name=scorer_name, experiment_id=experiment_id)
                resolved_scorers.append(registered_scorer)
            except MlflowException:
                # Provide helpful error message
                available_builtin = ", ".join(builtin_scorer_map.keys())
                raise click.UsageError(
                    f"Scorer '{scorer_name}' not found. "
                    f"Available built-in scorers: {available_builtin}. "
                    f"Or register a custom scorer first."
                )

    if not resolved_scorers:
        raise click.UsageError("No valid scorers specified")

    return resolved_scorers


def extract_assessments_from_traces(
    trace_ids: list[str], scorer_names: list[str]
) -> list[dict[str, Any]]:
    """
    Extract assessments by reading traces back from MLflow.

    After evaluation, assessments are attached to traces. This function
    reads the traces back and extracts assessments by matching scorer names.

    Args:
        trace_ids: List of trace IDs to read
        scorer_names: List of scorer names to extract assessments for

    Returns:
        List of dictionaries with trace_id and assessments
    """
    client = MlflowClient()
    output_data = []

    for trace_id in trace_ids:
        # Read trace back to get assessments
        trace = client.get_trace(trace_id)

        trace_result = {"trace_id": trace_id, "assessments": []}

        # Get assessments from the trace
        if trace and trace.info.assessments:
            # Create a map of assessment names to assessments
            assessment_map = {}
            for assessment in trace.info.assessments:
                # assessment is a Feedback object, use attribute access
                assessment_name = assessment.name
                assessment_map[assessment_name] = assessment

            # Extract assessments for each requested scorer
            for scorer_name in scorer_names:
                # Try to match by scorer name (case-insensitive)
                normalized_scorer = scorer_name.lower().replace(" ", "_")

                assessment_dict = {
                    "assessment_name": scorer_name,
                    "result": None,
                    "rationale": None,
                }

                # Try different name variations
                for assess_name, assess_data in assessment_map.items():
                    if any(
                        [
                            assess_name == scorer_name,
                            assess_name == normalized_scorer,
                            assess_name.lower() == scorer_name.lower(),
                            assess_name.lower().replace("_", "")
                            == scorer_name.lower().replace("_", ""),
                        ]
                    ):
                        # Extract result and rationale from Feedback object
                        if assess_data.feedback:
                            assessment_dict["result"] = assess_data.feedback.value

                        # Check top-level rationale
                        if assess_data.rationale:
                            assessment_dict["rationale"] = assess_data.rationale

                        # Check for errors
                        if assess_data.error:
                            assessment_dict["error"] = str(assess_data.error)
                        break

                trace_result["assessments"].append(assessment_dict)
        else:
            # No assessments found, create empty entries
            for scorer_name in scorer_names:
                trace_result["assessments"].append(
                    {
                        "assessment_name": scorer_name,
                        "result": None,
                        "rationale": None,
                    }
                )

        output_data.append(trace_result)

    return output_data


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
