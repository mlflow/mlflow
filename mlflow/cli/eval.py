"""
CLI commands for evaluating traces with scorers.
"""

import json
from typing import Literal

import click
import pandas as pd

import mlflow
from mlflow.cli.genai_eval_utils import (
    extract_assessments_from_results,
    format_table_output,
    resolve_scorers,
)
from mlflow.entities import Trace
from mlflow.genai.evaluation import evaluate
from mlflow.tracking import MlflowClient
from mlflow.utils.string_utils import _create_table


def _gather_traces(trace_ids: str, experiment_id: str) -> list[Trace]:
    """
    Gather and validate traces from the tracking store.

    Args:
        trace_ids: Comma-separated list of trace IDs to gather
        experiment_id: Expected experiment ID for all traces

    Returns:
        List of Trace objects

    Raises:
        click.UsageError: If any trace is not found or belongs to wrong experiment
    """
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

    return traces


def evaluate_traces(
    experiment_id: str,
    trace_ids: str,
    scorers: str,
    output_format: Literal["table", "json"] = "table",
) -> None:
    """
    Evaluate traces with specified scorers and output results.

    Args:
        experiment_id: The experiment ID to use for evaluation
        trace_ids: Comma-separated list of trace IDs to evaluate
        scorers: Comma-separated list of scorer names
        output_format: Output format ('table' or 'json')
    """
    mlflow.set_experiment(experiment_id=experiment_id)

    traces = _gather_traces(trace_ids, experiment_id)
    traces_df = pd.DataFrame([{"trace_id": t.info.trace_id, "trace": t} for t in traces])

    scorer_names = [name.strip() for name in scorers.split(",")]
    resolved_scorers = resolve_scorers(scorer_names, experiment_id)

    trace_count = len(traces)
    scorers_list = ", ".join(scorer_names)
    if trace_count == 1:
        trace_id = traces[0].info.trace_id
        click.echo(f"Evaluating trace {trace_id} with scorers: {scorers_list}...")
    else:
        click.echo(f"Evaluating {trace_count} traces with scorers: {scorers_list}...")

    try:
        results = evaluate(data=traces_df, scorers=resolved_scorers)
        evaluation_run_id = results.run_id
    except Exception as e:
        raise click.UsageError(f"Evaluation failed: {e}")

    results_df = results.result_df
    output_data = extract_assessments_from_results(results_df, evaluation_run_id)

    if output_format == "json":
        # Convert EvalResult objects to dicts for JSON serialization
        json_data = [
            {
                "trace_id": result.trace_id,
                "assessments": [
                    {
                        "name": assessment.name,
                        "result": assessment.result,
                        "rationale": assessment.rationale,
                        "error": assessment.error,
                    }
                    for assessment in result.assessments
                ],
            }
            for result in output_data
        ]
        if len(json_data) == 1:
            click.echo(json.dumps(json_data[0], indent=2))
        else:
            click.echo(json.dumps(json_data, indent=2))
    else:
        table_output = format_table_output(output_data)
        # Extract string values from Cell objects for table display
        table_data = [[cell.value for cell in row] for row in table_output.rows]
        # Add new line in the output before the final result.
        click.echo("")
        click.echo(_create_table(table_data, headers=table_output.headers))
