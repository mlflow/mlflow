"""
CLI for runs
"""

import json

import click

import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunStatus, ViewType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME
from mlflow.exceptions import MlflowException
from mlflow.mcp.decorator import mlflow_mcp
from mlflow.tracking import _get_store
from mlflow.utils.string_utils import _create_table
from mlflow.utils.time import conv_longdate_to_str

RUN_ID = click.option("--run-id", type=click.STRING, required=True)


@click.group("runs")
def commands():
    """
    Manage runs. To manage runs of experiments associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.
    """


@commands.command("list")
@mlflow_mcp(tool_name="list_runs")
@click.option(
    "--experiment-id",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    help="Specify the experiment ID for list of runs.",
    required=True,
)
@click.option(
    "--view",
    "-v",
    default="active_only",
    help="Select view type for list experiments. Valid view types are "
    "'active_only' (default), 'deleted_only', and 'all'.",
)
def list_run(experiment_id: str, view: str) -> None:
    """
    List all runs of the specified experiment in the configured tracking server.
    """
    store = _get_store()
    view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
    runs = store.search_runs([experiment_id], None, view_type)
    table = []
    for run in runs:
        run_name = run.info.run_name or ""
        table.append([conv_longdate_to_str(run.info.start_time), run_name, run.info.run_id])
    click.echo(_create_table(sorted(table, reverse=True), headers=["Date", "Name", "ID"]))


@commands.command("delete")
@mlflow_mcp(tool_name="delete_run")
@RUN_ID
def delete_run(run_id: str) -> None:
    """
    Mark a run for deletion. Return an error if the run does not exist or
    is already marked. You can restore a marked run with ``restore_run``,
    or permanently delete a run in the backend store.
    """
    store = _get_store()
    store.delete_run(run_id)
    click.echo(f"Run with ID {run_id} has been deleted.")


@commands.command("restore")
@mlflow_mcp(tool_name="restore_run")
@RUN_ID
def restore_run(run_id: str) -> None:
    """
    Restore a deleted run.
    Returns an error if the run is active or has been permanently deleted.
    """
    store = _get_store()
    store.restore_run(run_id)
    click.echo(f"Run with id {run_id} has been restored.")


@commands.command("describe")
@mlflow_mcp(tool_name="describe_run")
@RUN_ID
def describe_run(run_id: str) -> None:
    """
    All of run details will print to the stdout as JSON format.
    """
    store = _get_store()
    run = store.get_run(run_id)
    json_run = json.dumps(run.to_dictionary(), indent=4)
    click.echo(json_run)


@commands.command("create")
@mlflow_mcp(tool_name="create_run")
@click.option(
    "--experiment-id",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    help="ID of the experiment under which to create the run. "
    "Must specify either this or --experiment-name.",
)
@click.option(
    "--experiment-name",
    envvar=MLFLOW_EXPERIMENT_NAME.name,
    type=click.STRING,
    help="Name of the experiment under which to create the run. "
    "Must specify either this or --experiment-id.",
)
@click.option(
    "--run-name",
    type=click.STRING,
    help="Optional human-readable name for the run (e.g., 'baseline-model-v1').",
)
@click.option(
    "--description",
    type=click.STRING,
    help="Optional longer description of what this run represents.",
)
@click.option(
    "--tags",
    "-t",
    multiple=True,
    help="Key-value pairs to categorize and filter runs. Use multiple times for "
    "multiple tags. Format: key=value (e.g., env=prod, model=xgboost, version=1.0).",
)
@click.option(
    "--status",
    type=click.Choice(["FINISHED", "FAILED", "KILLED"], case_sensitive=False),
    default="FINISHED",
    help="Final status of the run. Options: FINISHED (default), FAILED, or KILLED.",
)
@click.option(
    "--parent-run-id",
    type=click.STRING,
    help="Optional ID of a parent run to create a nested run under.",
)
def create_run(
    experiment_id: str | None,
    experiment_name: str | None,
    run_name: str | None,
    description: str | None,
    tags: tuple[str, ...],
    status: str,
    parent_run_id: str | None,
) -> None:
    """
    Create a new MLflow run and immediately end it with the specified status.

    This command is useful for creating runs programmatically for testing, scripting,
    or recording completed experiments. The run will be created and immediately closed
    with the specified status (FINISHED, FAILED, or KILLED).
    """
    # Validate that exactly one of experiment_id or experiment_name is provided
    if (experiment_id is not None and experiment_name is not None) or (
        experiment_id is None and experiment_name is None
    ):
        raise click.UsageError("Must specify exactly one of --experiment-id or --experiment-name.")

    # Parse tags from key=value format
    tags_dict = {}
    if tags:
        for tag in tags:
            match tag.split("=", 1):
                case [key, value]:
                    if key in tags_dict:
                        raise click.UsageError(f"Duplicate tag key: '{key}'")
                    tags_dict[key] = value
                case _:
                    raise click.UsageError(
                        f"Invalid tag format: '{tag}'. Tags must be in key=value format."
                    )

    # Set the experiment if using experiment_name
    if experiment_name:
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        experiment_id = experiment.experiment_id

    # Start the run with the specified parameters
    try:
        # Start the run
        active_run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            nested=bool(parent_run_id),
            parent_run_id=parent_run_id,
            tags=tags_dict,
            description=description,
        )
        run_id = active_run.info.run_id
        actual_experiment_id = active_run.info.experiment_id

        # End the run with the specified status
        mlflow.end_run(status=RunStatus.to_string(getattr(RunStatus, status.upper())))

        # Output the created run information
        output = {
            "run_id": run_id,
            "experiment_id": actual_experiment_id,
            "status": status.upper(),
            "run_name": run_name,
        }

        click.echo(json.dumps(output, indent=2))

    except MlflowException as e:
        raise click.ClickException(f"Failed to create run: {e.message}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error creating run: {e!s}")


@commands.command("link-traces")
@mlflow_mcp(tool_name="link_traces_to_run")
@click.option(
    "--run-id",
    type=click.STRING,
    required=True,
    help="ID of the run to link traces to.",
)
@click.option(
    "trace_ids",
    "--trace-id",
    "-t",
    multiple=True,
    required=True,
    help="Trace ID to link to the run. Can be specified multiple times (maximum 100 traces).",
)
def link_traces(run_id: str, trace_ids: tuple[str, ...]) -> None:
    """
    Link traces to a run.

    This command links one or more traces to an existing run. Traces can be
    linked to runs to establish relationships between traces and runs.
    Maximum 100 traces can be linked in a single command.
    """
    try:
        client = MlflowClient()
        client.link_traces_to_run(list(trace_ids), run_id)

        # Output success message with count
        click.echo(f"Successfully linked {len(trace_ids)} trace(s) to run '{run_id}'")

    except MlflowException as e:
        raise click.ClickException(f"Failed to link traces: {e.message}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error linking traces: {e!s}")


@commands.command("get-metric-history")
@mlflow_mcp(tool_name="get_metric_history")
@click.option(
    "--run-id",
    type=click.STRING,
    required=True,
    help="ID of the run to get metric history from.",
)
@click.option(
    "--metric-key",
    type=click.STRING,
    required=True,
    help="Name of the metric to retrieve history for.",
)
@click.option(
    "--max-results",
    type=click.INT,
    help="Maximum number of metric history entries to return. If not specified, returns all entries.",
)
def get_metric_history(run_id: str, metric_key: str, max_results: int | None) -> None:
    """
    Get the full time series history of a metric for a run.

    Returns a JSON array of metric history entries, each containing:
    - step: The step number at which the metric was logged
    - value: The metric value
    - timestamp: Unix timestamp (milliseconds) when the metric was logged
    """
    try:
        client = MlflowClient()
        metric_history = client.get_metric_history(run_id, metric_key)

        # Convert to list of dicts for JSON output
        history_data = [
            {
                "step": metric.step,
                "value": metric.value,
                "timestamp": metric.timestamp,
            }
            for metric in metric_history
        ]

        # Apply max_results limit if specified
        if max_results is not None and max_results > 0:
            history_data = history_data[:max_results]

        click.echo(json.dumps(history_data, indent=2))

    except MlflowException as e:
        raise click.ClickException(f"Failed to get metric history: {e.message}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error getting metric history: {e!s}")


@commands.command("search")
@mlflow_mcp(tool_name="search_runs")
@click.option(
    "--experiment-ids",
    type=click.STRING,
    required=True,
    help="Comma-separated list of experiment IDs to search within.",
)
@click.option(
    "--filter",
    "filter_string",
    type=click.STRING,
    default="",
    help="Filter query string using MLflow's search syntax. "
    "Examples: 'metrics.accuracy > 0.9', 'params.lr = \"1e-5\"', "
    "'attributes.status = \"FINISHED\"'.",
)
@click.option(
    "--order-by",
    type=click.STRING,
    multiple=True,
    help="Order results by specified columns. Can be specified multiple times. "
    "Format: 'column_name [ASC|DESC]'. Examples: 'metrics.accuracy DESC', 'start_time ASC'.",
)
@click.option(
    "--max-results",
    type=click.INT,
    default=100,
    help="Maximum number of runs to return. Default is 100.",
)
@click.option(
    "--view-type",
    type=click.Choice(["ACTIVE_ONLY", "DELETED_ONLY", "ALL"], case_sensitive=False),
    default="ACTIVE_ONLY",
    help="Type of runs to return. Options: ACTIVE_ONLY (default), DELETED_ONLY, or ALL.",
)
def search_runs(
    experiment_ids: str,
    filter_string: str,
    order_by: tuple[str, ...],
    max_results: int,
    view_type: str,
) -> None:
    """
    Search for runs matching specified criteria across experiments.

    Returns a JSON array of runs with their parameters, metrics, and metadata.
    Supports filtering, ordering, and pagination.
    """
    try:
        client = MlflowClient()

        # Parse experiment IDs
        exp_ids = [exp_id.strip() for exp_id in experiment_ids.split(",") if exp_id.strip()]
        if not exp_ids:
            raise click.UsageError("At least one experiment ID must be provided.")

        # Convert order_by tuple to list
        order_by_list = list(order_by) if order_by else None

        # Search runs
        runs = client.search_runs(
            experiment_ids=exp_ids,
            filter_string=filter_string,
            order_by=order_by_list,
            max_results=max_results,
            run_view_type=view_type.upper(),
        )

        # Convert runs to JSON-serializable format
        runs_data = []
        for run in runs:
            run_dict = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "run_name": run.info.run_name,
                "metrics": {key: value for key, value in run.data.metrics.items()},
                "params": {key: value for key, value in run.data.params.items()},
                "tags": {key: value for key, value in run.data.tags.items()},
            }
            runs_data.append(run_dict)

        click.echo(json.dumps(runs_data, indent=2))

    except MlflowException as e:
        raise click.ClickException(f"Failed to search runs: {e.message}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error searching runs: {e!s}")


@commands.command("compare")
@mlflow_mcp(tool_name="compare_runs")
@click.option(
    "--run-ids",
    type=click.STRING,
    required=True,
    help="Comma-separated list of run IDs to compare.",
)
@click.option(
    "--metric-keys",
    type=click.STRING,
    help="Comma-separated list of metric keys to include. If not specified, includes all metrics.",
)
@click.option(
    "--param-keys",
    type=click.STRING,
    help="Comma-separated list of parameter keys to include. If not specified, includes all parameters.",
)
def compare_runs(
    run_ids: str,
    metric_keys: str | None,
    param_keys: str | None,
) -> None:
    """
    Compare multiple runs side-by-side.

    Returns a JSON object with run IDs as keys, each containing the run's
    metrics and parameters for easy comparison.
    """
    try:
        client = MlflowClient()

        # Parse run IDs
        run_id_list = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
        if not run_id_list:
            raise click.UsageError("At least one run ID must be provided.")

        # Parse metric and param keys if provided
        metric_key_set = (
            {key.strip() for key in metric_keys.split(",") if key.strip()}
            if metric_keys
            else None
        )
        param_key_set = (
            {key.strip() for key in param_keys.split(",") if key.strip()} if param_keys else None
        )

        # Fetch runs and build comparison
        comparison = {}
        for run_id in run_id_list:
            run = client.get_run(run_id)

            # Filter metrics
            if metric_key_set is not None:
                metrics = {k: v for k, v in run.data.metrics.items() if k in metric_key_set}
            else:
                metrics = dict(run.data.metrics)

            # Filter params
            if param_key_set is not None:
                params = {k: v for k, v in run.data.params.items() if k in param_key_set}
            else:
                params = dict(run.data.params)

            comparison[run_id] = {
                "run_name": run.info.run_name,
                "status": run.info.status,
                "metrics": metrics,
                "params": params,
            }

        click.echo(json.dumps(comparison, indent=2))

    except MlflowException as e:
        raise click.ClickException(f"Failed to compare runs: {e.message}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error comparing runs: {e!s}")

