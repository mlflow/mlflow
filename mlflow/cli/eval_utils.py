"""
Utility functions for trace evaluation output formatting.
"""

from typing import Any

import click
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import get_all_scorers, get_scorer


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
                available_builtin = ", ".join(builtin_scorer_map.keys())
                raise click.UsageError(
                    f"Scorer '{scorer_name}' not found. "
                    f"Available built-in scorers: {available_builtin}. "
                    f"Or register a custom scorer first."
                )

    if not resolved_scorers:
        raise click.UsageError("No valid scorers specified")

    return resolved_scorers


def extract_assessments_from_results(
    results_df: pd.DataFrame, evaluation_run_id: str
) -> list[dict[str, Any]]:
    """
    Extract assessments from evaluation results DataFrame.

    The evaluate() function returns results with a DataFrame that contains
    an 'assessments' column. Each row has a list of assessment dictionaries
    with metadata including 'mlflow.assessment.sourceRunId' that we use to
    filter assessments from this specific evaluation run.

    Args:
        results_df: DataFrame from evaluate() results containing assessments column
        evaluation_run_id: The MLflow run ID from the evaluation that generated the assessments

    Returns:
        List of dictionaries with trace_id and assessments
    """
    output_data = []

    for _, row in results_df.iterrows():
        trace_id = row["trace_id"]
        trace_result = {"trace_id": trace_id, "assessments": []}

        assessments_list = row.get("assessments", [])

        if assessments_list:
            for assessment_dict in assessments_list:
                # Check if this assessment is from our evaluation run
                metadata = assessment_dict.get("metadata", {})
                source_run_id = metadata.get("mlflow.assessment.sourceRunId")

                if source_run_id == evaluation_run_id:
                    result_dict = {
                        "assessment_name": assessment_dict.get("assessment_name"),
                        "result": None,
                        "rationale": None,
                    }

                    # Extract feedback value
                    feedback = assessment_dict.get("feedback")
                    if feedback and isinstance(feedback, dict):
                        result_dict["result"] = feedback.get("value")

                    # Extract rationale
                    rationale = assessment_dict.get("rationale")
                    if rationale:
                        result_dict["rationale"] = rationale

                    # Check for errors (if assessment has error info)
                    error = assessment_dict.get("error")
                    if error:
                        result_dict["error"] = str(error)

                    trace_result["assessments"].append(result_dict)

        # If no assessments were found for this trace, add error markers
        if not trace_result["assessments"]:
            trace_result["assessments"].append(
                {
                    "assessment_name": "N/A",
                    "result": None,
                    "rationale": None,
                    "error": "No assessments found on trace",
                }
            )

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
            result_val = assessment.get("result")
            rationale_val = assessment.get("rationale")
            error_val = assessment.get("error")

            if error_val:
                cell_content = f"error: {error_val}"
            elif result_val is not None and rationale_val:
                cell_content = f"value: {result_val}, rationale: {rationale_val}"
            elif result_val is not None:
                cell_content = f"value: {result_val}"
            elif rationale_val:
                cell_content = f"rationale: {rationale_val}"
            else:
                cell_content = "N/A"

            row.append(cell_content)

        table_data.append(row)

    return headers, table_data
