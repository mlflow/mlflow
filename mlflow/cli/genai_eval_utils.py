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
class Assessment:
    """
    Structured assessment data for a trace evaluation.

    This dataclass provides type-safe access to assessment results,
    replacing dict-based access for better clarity and safety.
    """

    assessment_name: str | None
    """The name of the assessment"""

    result: Any | None = None
    """The result value from the assessment"""

    rationale: str | None = None
    """The rationale text explaining the assessment"""

    error: str | None = None
    """Error message if the assessment failed"""


@dataclass
class EvalResult:
    """
    Container for evaluation results for a single trace.

    This dataclass provides structured access to trace evaluation data,
    replacing dict-based access for better type safety.
    """

    trace_id: str
    """The trace ID"""

    assessments: list[Assessment]
    """List of Assessment objects for this trace"""


@dataclass
class TableOutput:
    """Container for formatted table data."""

    headers: list[str]
    rows: list[list[Cell]]


def _format_assessment_cell(assessment: Assessment | None) -> Cell:
    """
    Format a single assessment cell for table display.

    Args:
        assessment: Assessment object with result, rationale, and error fields

    Returns:
        Cell object with formatted value and metadata
    """
    if not assessment:
        return Cell(value=NA_VALUE)

    if assessment.error:
        display_value = f"error: {assessment.error}"
    elif assessment.result is not None and assessment.rationale:
        display_value = f"value: {assessment.result}, rationale: {assessment.rationale}"
    elif assessment.result is not None:
        display_value = f"value: {assessment.result}"
    elif assessment.rationale:
        display_value = f"rationale: {assessment.rationale}"
    else:
        display_value = NA_VALUE

    return Cell(
        value=display_value,
        assessment_name=assessment.assessment_name,
        result=assessment.result,
        rationale=assessment.rationale,
        error=assessment.error,
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
) -> list[EvalResult]:
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
        List of EvalResult objects with trace_id and assessments
    """
    output_data = []

    for _, row in results_df.iterrows():
        trace_id = row.get("trace_id", "unknown")
        assessments_list = []

        for assessment_dict in row.get("assessments", []):
            # Only consider assessments from the evaluation run
            metadata = assessment_dict.get("metadata", {})
            source_run_id = metadata.get(AssessmentMetadataKey.SOURCE_RUN_ID)

            if source_run_id != evaluation_run_id:
                continue

            assessment_name = assessment_dict.get("assessment_name")
            result_val = None
            rationale_val = None
            error_val = None

            if (feedback := assessment_dict.get("feedback")) and isinstance(feedback, dict):
                result_val = feedback.get("value")

            if rationale := assessment_dict.get("rationale"):
                rationale_val = rationale

            if error := assessment_dict.get("error"):
                error_val = str(error)

            assessments_list.append(
                Assessment(
                    assessment_name=assessment_name,
                    result=result_val,
                    rationale=rationale_val,
                    error=error_val,
                )
            )

        # If no assessments were found for this trace, add error markers
        if not assessments_list:
            assessments_list.append(
                Assessment(
                    assessment_name=NA_VALUE,
                    result=None,
                    rationale=None,
                    error="No assessments found on trace",
                )
            )

        output_data.append(EvalResult(trace_id=trace_id, assessments=assessments_list))

    return output_data


def format_table_output(output_data: list[EvalResult]) -> TableOutput:
    """
    Format evaluation results as table data.

    Args:
        output_data: List of EvalResult objects with assessments

    Returns:
        TableOutput dataclass containing headers and rows
    """
    # Extract unique assessment names from output_data to use as column headers
    # Note: assessment_name can be None, so we filter it out
    assessment_names_set = set()
    for trace_result in output_data:
        for assessment in trace_result.assessments:
            if assessment.assessment_name and assessment.assessment_name != NA_VALUE:
                assessment_names_set.add(assessment.assessment_name)

    # Sort for consistent ordering
    assessment_names = sorted(assessment_names_set)

    headers = ["trace_id"] + assessment_names
    table_data = []

    for trace_result in output_data:
        # Create Cell for trace_id column
        row = [Cell(value=trace_result.trace_id)]

        # Build a map of assessment_name -> assessment for this trace
        assessment_map = {
            assessment.assessment_name: assessment
            for assessment in trace_result.assessments
            if assessment.assessment_name and assessment.assessment_name != NA_VALUE
        }

        # For each assessment name in headers, get the corresponding assessment
        for assessment_name in assessment_names:
            cell_content = _format_assessment_cell(assessment_map.get(assessment_name))
            row.append(cell_content)

        table_data.append(row)

    return TableOutput(headers=headers, rows=table_data)
