"""
CLI commands for evaluating traces with scorers.
"""

import json

import click
import pandas as pd

from mlflow.cli.eval_utils import (
    extract_assessments_from_results,
    format_table_output,
    resolve_scorers,
)
from mlflow.genai.evaluation import evaluate
from mlflow.tracking import MlflowClient
from mlflow.utils.string_utils import _create_table


def _format_error_message(error_msg: str) -> str:
    """Format error message for display."""
    if "OpenAIException" in error_msg and "api_key" in error_msg:
        return "ERROR: Missing OpenAI API key"
    elif "AuthenticationError" in error_msg:
        return "ERROR: Authentication failed"
    else:
        return f"ERROR: {error_msg[:50]}..."


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

    # Resolve scorers - check built-in first, then registered
    resolved_scorers = resolve_scorers(scorer_names, experiment_id)

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
        evaluation_run_id = results.run_id
    except Exception as e:
        raise click.UsageError(f"Evaluation failed: {e}")

    # Extract assessments from the results DataFrame
    # The evaluate() function returns results with a DataFrame in results.tables['eval_results']
    # that contains an 'assessments' column with all assessment data
    results_df = results.tables["eval_results"]
    output_data = extract_assessments_from_results(results_df, evaluation_run_id)

    # Format and display results
    if output == "json":
        # Output single object for single trace, array for multiple
        if len(output_data) == 1:
            click.echo(json.dumps(output_data[0], indent=2))
        else:
            click.echo(json.dumps(output_data, indent=2))
    else:
        # Table output format
        # Collect all unique assessment names from the results for column headers
        assessment_names = []
        for trace_result in output_data:
            for assessment in trace_result["assessments"]:
                name = assessment.get("assessment_name")
                if name and name not in assessment_names and name != "N/A":
                    assessment_names.append(name)

        headers, table_data = format_table_output(
            output_data, assessment_names, _format_error_message
        )

        # Display the table with a clear separator
        click.echo("")  # Add blank line after MLflow messages
        click.echo(_create_table(table_data, headers=headers))
