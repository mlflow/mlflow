"""
CLI commands for evaluating traces with scorers.
"""

import json

import click
import pandas as pd

import mlflow
from mlflow.cli.eval_utils import (
    extract_assessments_from_results,
    format_table_output,
    resolve_scorers,
)
from mlflow.genai.evaluation import evaluate
from mlflow.tracking import MlflowClient
from mlflow.utils.string_utils import _create_table


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
    mlflow.set_experiment(experiment_id=experiment_id)

    # Gather traces
    trace_id_list = [tid.strip() for tid in trace_ids.split(",")]
    client = MlflowClient()
    traces = []

    for trace_id in trace_id_list:
        try:
            trace = client.get_trace(trace_id, display=False)
        except Exception as e:
            raise click.UsageError(f"Failed to get trace '{trace_id}': {e}")

        if trace is None:
            raise click.UsageError(f"Trace with ID '{trace_id}' not found")

        if trace.info.experiment_id != experiment_id:
            raise click.UsageError(
                f"Trace '{trace_id}' belongs to experiment '{trace.info.experiment_id}', "
                f"not the specified experiment '{experiment_id}'"
            )

        traces.append(trace)

    traces_df = pd.DataFrame([{"trace_id": t.info.trace_id, "trace": t} for t in traces])

    # Resolve scorers
    scorer_names = [name.strip() for name in scorers.split(",")]
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

        results = evaluate(data=traces_df, scorers=resolved_scorers)
        evaluation_run_id = results.run_id
    except Exception as e:
        raise click.UsageError(f"Evaluation failed: {e}")

    # Parse results
    results_df = results.tables["eval_results"]
    output_data = extract_assessments_from_results(results_df, evaluation_run_id)

    # Format and display results
    if output == "json":
        if len(output_data) == 1:
            click.echo(json.dumps(output_data[0], indent=2))
        else:
            click.echo(json.dumps(output_data, indent=2))
    else:
        headers, table_data = format_table_output(output_data)
        click.echo("")
        click.echo(_create_table(table_data, headers=headers))
