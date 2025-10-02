"""
Utility functions for trace evaluation output formatting.
"""

from dataclasses import dataclass
from typing import Any

import click
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Scorer, get_all_scorers, get_scorer
from mlflow.tracing.constant import AssessmentMetadataKey

# Represents the absence of a value for an assessment
NA_VALUE = "N/A"


@dataclass
class Cell:
    """
    Structured cell data for table display with metadata.

    This dataclass provides richer information about table cells beyond just
    the displayed string value, enabling future enhancements like tooltips,
    interactive features, or programmatic access to assessment data.
    """

    value: str
    """The formatted display value for the cell"""

    assessment_name: str | None = None
    """The name of the assessment this cell represents"""

    result: Any | None = None
    """The raw result value from the assessment"""

    rationale: str | None = None
    """The rationale text explaining the assessment"""

    error: str | None = None
    """Error message if the assessment failed"""


@dataclass
class TableOutput:
    """Container for formatted table data."""

    headers: list[str]
    rows: list[list[Cell]]


def _format_assessment_cell(assessment: dict[str, Any] | None) -> Cell:
    """
    Format a single assessment cell for table display.

    Args:
        assessment: Assessment dictionary with result, rationale, and error fields

    Returns:
        Cell object with formatted value and metadata
    """
    if not assessment:
        return Cell(value=NA_VALUE)

    assessment_name = assessment.get("assessment_name")
    result_val = assessment.get("result")
    rationale_val = assessment.get("rationale")
    error_val = assessment.get("error")

    if error_val:
        display_value = f"error: {error_val}"
    elif result_val is not None and rationale_val:
        display_value = f"value: {result_val}, rationale: {rationale_val}"
    elif result_val is not None:
        display_value = f"value: {result_val}"
    elif rationale_val:
        display_value = f"rationale: {rationale_val}"
    else:
        display_value = NA_VALUE

    return Cell(
        value=display_value,
        assessment_name=assessment_name,
        result=result_val,
        rationale=rationale_val,
        error=error_val,
    )


def resolve_scorers(scorer_names: list[str], experiment_id: str) -> list[Scorer]:
    """
    Resolve scorer names to scorer objects.

    Checks built-in scorers first, then registered scorers.
    Supports both class names (e.g., "RelevanceToQuery") and snake_case
    scorer names (e.g., "relevance_to_query").

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
    # Build map with both class name and snake_case name for lookup
    builtin_scorer_map = {}
    for scorer in builtin_scorers:
        # Map by class name (e.g., "RelevanceToQuery")
        builtin_scorer_map[scorer.__class__.__name__] = scorer
        # Map by scorer.name (snake_case, e.g., "relevance_to_query")
        if scorer.name is not None:
            builtin_scorer_map[scorer.name] = scorer

    for scorer_name in scorer_names:
        if scorer_name in builtin_scorer_map:
            resolved_scorers.append(builtin_scorer_map[scorer_name])
        else:
            # Try to get it as a registered scorer
            try:
                registered_scorer = get_scorer(name=scorer_name, experiment_id=experiment_id)
                resolved_scorers.append(registered_scorer)
            except MlflowException:
                available_builtin = ", ".join(
                    sorted({scorer.__class__.__name__ for scorer in builtin_scorers})
                )
                raise click.UsageError(
                    f"Scorer '{scorer_name}' not found. "
                    f"Only built-in or registered scorers can be resolved. "
                    f"Available built-in scorers: {available_builtin}. "
                    f"To use a custom scorer, register it first in experiment {experiment_id} "
                    f"using the register_scorer() API."
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
    with metadata including AssessmentMetadataKey.SOURCE_RUN_ID that we use to
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
        trace_assessments = {"trace_id": trace_id, "assessments": []}

        for assessment_dict in row.get("assessments", []):
            # Only consider assessments from the evaluation run
            metadata = assessment_dict.get("metadata", {})
            source_run_id = metadata.get(AssessmentMetadataKey.SOURCE_RUN_ID)

            if source_run_id != evaluation_run_id:
                continue

            result_dict = {
                "assessment_name": assessment_dict.get("assessment_name"),
                "result": None,
                "rationale": None,
            }

            if (feedback := assessment_dict.get("feedback")) and isinstance(feedback, dict):
                result_dict["result"] = feedback.get("value")

            if rationale := assessment_dict.get("rationale"):
                result_dict["rationale"] = rationale

            if error := assessment_dict.get("error"):
                result_dict["error"] = str(error)

            trace_assessments["assessments"].append(result_dict)

        # If no assessments were found for this trace, add error markers
        if not trace_assessments["assessments"]:
            trace_assessments["assessments"].append(
                {
                    "assessment_name": NA_VALUE,
                    "result": None,
                    "rationale": None,
                    "error": "No assessments found on trace",
                }
            )

        output_data.append(trace_assessments)

    return output_data


def format_table_output(output_data: list[dict[str, Any]]) -> TableOutput:
    """
    Format evaluation results as table data.

    Args:
        output_data: List of trace results with assessments

    Returns:
        TableOutput dataclass containing headers and rows
    """
    # Extract unique assessment names from output_data to use as column headers
    assessment_names_set = set()
    for trace_result in output_data:
        for assessment in trace_result["assessments"]:
            if (name := assessment.get("assessment_name")) and name != NA_VALUE:
                assessment_names_set.add(name)

    # Sort for consistent ordering
    assessment_names = sorted(assessment_names_set)

    headers = ["trace_id"] + assessment_names
    table_data = []

    for trace_result in output_data:
        # Create Cell for trace_id column
        row = [Cell(value=trace_result["trace_id"])]

        # Build a map of assessment_name -> assessment for this trace
        assessment_map = {
            assessment.get("assessment_name"): assessment
            for assessment in trace_result["assessments"]
            if assessment.get("assessment_name") != NA_VALUE
        }

        # For each assessment name in headers, get the corresponding assessment
        for assessment_name in assessment_names:
            cell_content = _format_assessment_cell(assessment_map.get(assessment_name))
            row.append(cell_content)

        table_data.append(row)

    return TableOutput(headers=headers, rows=table_data)
