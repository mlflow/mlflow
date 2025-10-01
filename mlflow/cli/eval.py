"""
CLI commands for evaluating traces with scorers.
"""

import json
from typing import Any

import click
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation import evaluate
from mlflow.genai.scorers import get_all_scorers, get_scorer
from mlflow.tracking import MlflowClient


def _extract_assessment_from_column(
    df: pd.DataFrame, idx: int, scorer_name: str, normalized_scorer_name: str
) -> dict:
    """
    Extract assessment values from standard evaluation result columns.

    Args:
        df: DataFrame containing evaluation results
        idx: Row index to extract from
        scorer_name: Original scorer name
        normalized_scorer_name: Normalized scorer name (lowercase with underscores)

    Returns:
        Dictionary with assessment_name, result, rationale, and error fields
    """
    assessment = {
        "assessment_name": scorer_name,
        "result": None,
        "rationale": None,
    }

    value_col = f"{normalized_scorer_name}/value"
    rationale_col = f"{normalized_scorer_name}/rationale"
    error_col = f"{normalized_scorer_name}/error_message"

    # Extract value
    if value_col in df.columns and idx < len(df):
        value = df[value_col].iloc[idx]
        if pd.notna(value):
            # Convert to native Python type if needed
            if hasattr(value, "item"):
                assessment["result"] = value.item()
            else:
                assessment["result"] = value

    # Extract rationale
    if rationale_col in df.columns and idx < len(df):
        rationale = df[rationale_col].iloc[idx]
        if pd.notna(rationale):
            assessment["rationale"] = str(rationale)

    # Extract error if no result
    if assessment["result"] is None and error_col in df.columns and idx < len(df):
        error_msg = df[error_col].iloc[idx]
        if pd.notna(error_msg):
            assessment["error"] = str(error_msg)

    return assessment


def _extract_assessment_from_assessments_column(
    assessments_info: list, scorer_name: str, normalized_scorer_name: str
) -> dict | None:
    """
    Extract assessment from the assessments column format.

    Args:
        assessments_data: List of assessment dictionaries
        scorer_name: Original scorer name
        normalized_scorer_name: Normalized scorer name

    Returns:
        Assessment dictionary if found, None otherwise
    """
    if not assessments_info:
        return None

    for assessment_info in assessments_info:
        # Assessment is a dictionary - access keys directly
        assess_name = assessment_info.get("assessment_name", "") or assessment_info.get("name", "")

        # Try different possible name matches
        if any(
            [
                assess_name == normalized_scorer_name,
                assess_name.lower() == scorer_name.lower(),
                assess_name.lower().replace("_", "") == scorer_name.lower().replace("_", ""),
                assess_name == scorer_name,
            ]
        ):
            assessment = {
                "assessment_name": scorer_name,
                "result": None,
                "rationale": None,
            }

            feedback = assessment_info.get("feedback", {})
            if feedback:
                if "value" in feedback:
                    assessment["result"] = feedback["value"]
                if "rationale" in feedback:
                    assessment["rationale"] = feedback["rationale"]

            # Also check for rationale at the top level
            if "rationale" in assessment_info:
                assessment["rationale"] = assessment_info["rationale"]

            # Check for errors in feedback
            if feedback and "error" in feedback and feedback["error"]:
                error_info = feedback["error"]
                error_msg = (
                    error_info.get("error_message") or error_info.get("message", "")
                    if isinstance(error_info, dict)
                    else str(error_info)
                )
                if error_msg:
                    assessment["error"] = str(error_msg)

            return assessment

    return None


def _format_error_message(error_msg: str) -> str:
    """Format error message for display."""
    if "OpenAIException" in error_msg and "api_key" in error_msg:
        return "ERROR: Missing OpenAI API key"
    elif "AuthenticationError" in error_msg:
        return "ERROR: Authentication failed"
    else:
        return f"ERROR: {error_msg[:50]}..."


def _get_results_dataframe(results: Any, debug: bool = False) -> pd.DataFrame:
    """
    Extract the results DataFrame from evaluation results object.

    Args:
        results: EvaluationResult object from mlflow.evaluate()
        debug: Whether to output debug information

    Returns:
        DataFrame containing evaluation results

    Raises:
        click.UsageError: If no results DataFrame can be found
    """
    df = None

    # Check for results in tables attribute
    if hasattr(results, "tables") and results.tables:
        for table_name in ["eval_results", "eval_results_table", "results"]:
            if table_name in results.tables:
                df = results.tables[table_name]
                break

    # Try alternative attribute names
    if df is None:
        for attr_name in ["result_df", "_result_df"]:
            if hasattr(results, attr_name):
                attr_value = getattr(results, attr_name)
                if hasattr(attr_value, "columns"):  # It's a DataFrame-like object
                    df = attr_value
                    break

    if df is None:
        if debug:
            click.echo("Debug: Available attributes on results:", err=True)
            click.echo(
                f"  - dir(results): {[attr for attr in dir(results) if not attr.startswith('_')]}",
                err=True,
            )
            if hasattr(results, "tables"):
                tables_info = list(results.tables.keys()) if results.tables else "None"
                click.echo(f"  - results.tables keys: {tables_info}", err=True)
        raise click.UsageError("No evaluation results DataFrame found in results object")

    return df


def evaluate_traces(
    experiment_id: str,
    trace_ids: str,
    scorers: str,
    output: str = "table",
    debug: bool = False,
) -> None:
    """
    Evaluate traces with specified scorers and output results.

    Args:
        experiment_id: The experiment ID to use for evaluation
        trace_ids: Comma-separated list of trace IDs to evaluate
        scorers: Comma-separated list of scorer names
        output: Output format ('table' or 'json')
        debug: Whether to output debug information
    """
    # Set the experiment context using the ID directly
    # This works with both local and Databricks tracking URIs
    import os

    os.environ["MLFLOW_EXPERIMENT_ID"] = experiment_id

    # Parse trace IDs
    trace_id_list = [tid.strip() for tid in trace_ids.split(",")]

    # Get the traces directly
    client = MlflowClient()
    traces = []

    for trace_id in trace_id_list:
        try:
            trace = client.get_trace(trace_id, display=False)
        except Exception as e:
            raise click.UsageError(f"Failed to get trace '{trace_id}': {e}")

        if trace is None:
            raise click.UsageError(f"Trace with ID '{trace_id}' not found")

        # Verify the trace belongs to the specified experiment
        if trace.info.experiment_id != experiment_id:
            raise click.UsageError(
                f"Trace '{trace_id}' belongs to experiment '{trace.info.experiment_id}', "
                f"not the specified experiment '{experiment_id}'"
            )

        traces.append(trace)

    # Create a DataFrame with trace column for evaluate()
    traces_df = pd.DataFrame([{"trace_id": t.info.trace_id, "trace": t} for t in traces])

    # Parse scorer names
    scorer_names = [name.strip() for name in scorers.split(",")]

    # ALKIS: Refactor the resolution of scorers into a separate method in eval_utils.py. Add tests.
    # Resolve scorers - check built-in first, then registered
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

    # Run evaluation
    try:
        trace_count = len(trace_id_list)
        if trace_count == 1:
            click.echo(
                f"Evaluating trace {trace_id_list[0]} with scorers: {', '.join(scorer_names)}..."
            )
        else:
            click.echo(
                f"Evaluating {trace_count} traces with scorers: {', '.join(scorer_names)}..."
            )

        # Pass the DataFrame to evaluate()
        results = evaluate(data=traces_df, scorers=resolved_scorers)
    except Exception as e:
        raise click.UsageError(f"Evaluation failed: {e}")

    # Get the results DataFrame
    df = _get_results_dataframe(results, debug=debug)

    # Get trace_id column if it exists, otherwise use the original list
    result_trace_ids = df["trace_id"].tolist() if "trace_id" in df.columns else trace_id_list

    # Build results data structure (used by both JSON and table output)
    # ALKIS: Do this import at the top of the file. Same for other local imports in this file.
    from mlflow.cli.eval_utils import build_output_data

    # ALKIS: Use a different logic to extract assessments. Use the get_trace() API to read back the traces. The traces will now have assessments attached to them, and these assessments will have the name of the scorer who generated them. Match the names to the scorer.
    output_data = build_output_data(
        df,
        result_trace_ids,
        scorer_names,
        _extract_assessment_from_column,
        _extract_assessment_from_assessments_column,
    )

    # Format and display results
    if output == "json":
        # Output single object for single trace, array for multiple
        if len(output_data) == 1:
            click.echo(json.dumps(output_data[0], indent=2))
        else:
            click.echo(json.dumps(output_data, indent=2))
    else:
        # Table output format
        from mlflow.cli.eval_utils import format_table_output
        from mlflow.utils.string_utils import _create_table

        headers, table_data = format_table_output(output_data, scorer_names, _format_error_message)

        # Display the table with a clear separator
        click.echo("")  # Add blank line after MLflow messages
        click.echo(_create_table(table_data, headers=headers))
