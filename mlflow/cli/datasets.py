import json
from typing import Any, Literal

import click

from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import get_dataset
from mlflow.tracing.client import TracingClient
from mlflow.utils.string_utils import _create_table
from mlflow.utils.time import conv_longdate_to_str

EXPERIMENT_ID = click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    required=True,
    help="Experiment ID to list datasets for. Can be set via MLFLOW_EXPERIMENT_ID env var.",
)


def _format_datasets_as_json(datasets) -> dict[str, Any]:
    """Format datasets as a JSON-serializable dictionary."""
    return {
        "datasets": [
            {
                "dataset_id": ds.dataset_id,
                "name": ds.name,
                "digest": ds.digest,
                "created_time": ds.created_time,
                "last_update_time": ds.last_update_time,
                "created_by": ds.created_by,
                "last_updated_by": ds.last_updated_by,
                "tags": ds.tags,
            }
            for ds in datasets
        ],
        "next_page_token": datasets.token,
    }


def _format_datasets_as_table(datasets) -> tuple[list[list[str]], list[str]]:
    """Format datasets as table rows with headers."""
    headers = ["Dataset ID", "Name", "Created", "Last Updated", "Created By"]
    rows = []
    for ds in datasets:
        created = conv_longdate_to_str(ds.created_time) if ds.created_time else ""
        updated = conv_longdate_to_str(ds.last_update_time) if ds.last_update_time else ""
        rows.append([ds.dataset_id, ds.name, created, updated, ds.created_by or ""])
    return rows, headers


@click.group("datasets")
def commands():
    """Manage GenAI evaluation datasets."""


@commands.command("list")
@EXPERIMENT_ID
@click.option(
    "--filter-string",
    type=click.STRING,
    help="Filter string (e.g., \"name LIKE 'qa_%'\").",
)
@click.option(
    "--max-results",
    type=click.INT,
    default=50,
    help="Maximum results (default: 50).",
)
@click.option(
    "--order-by",
    type=click.STRING,
    help="Columns to order by (e.g., 'last_update_time DESC').",
)
@click.option(
    "--page-token",
    type=click.STRING,
    help="Pagination token.",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
def list_datasets(
    experiment_id: str,
    filter_string: str | None = None,
    max_results: int = 50,
    order_by: str | None = None,
    page_token: str | None = None,
    output: Literal["table", "json"] = "table",
) -> None:
    """
    List GenAI evaluation datasets associated with an experiment.

    \b
    Examples:
    # List datasets in experiment 1
    mlflow datasets list --experiment-id 1

    \b
    # Using environment variable
    export MLFLOW_EXPERIMENT_ID=1
    mlflow datasets list --max-results 10

    \b
    # Filter datasets by name pattern
    mlflow datasets list --experiment-id 1 --filter-string "name LIKE 'qa_%'"

    \b
    # Order results by last update time
    mlflow datasets list --experiment-id 1 --order-by "last_update_time DESC"

    \b
    # Output as JSON
    mlflow datasets list --experiment-id 1 --output json
    """
    client = MlflowClient()
    order_by_list = [o.strip() for o in order_by.split(",")] if order_by else None

    datasets = client.search_datasets(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by_list,
        page_token=page_token,
    )

    if output == "json":
        result = _format_datasets_as_json(datasets)
        click.echo(json.dumps(result, indent=2))
    else:
        rows, headers = _format_datasets_as_table(datasets)
        click.echo(_create_table(rows, headers=headers))

        if datasets.token:
            click.echo(f"\nNext page token: {datasets.token}")


@commands.command("add-traces")
@click.option(
    "--dataset-id",
    type=click.STRING,
    required=True,
    help="ID of the evaluation dataset to add traces to.",
)
@click.option(
    "--trace-ids",
    type=click.STRING,
    required=True,
    help="Comma-separated list of trace IDs to add.",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
def add_traces(
    dataset_id: str,
    trace_ids: str,
    output: Literal["table", "json"] = "table",
) -> None:
    """
    Add traces to a GenAI evaluation dataset.

    Traces are merged by trace ID, so re-adding a trace already in the dataset
    is a no-op rather than creating a duplicate. Works against both OSS MLflow
    and Databricks-managed datasets.

    \b
    Examples:
    # Add a single trace
    mlflow datasets add-traces --dataset-id d-abc123 --trace-ids tr-1234567890abcdef

    \b
    # Add multiple traces
    mlflow datasets add-traces --dataset-id d-abc123 --trace-ids tr-1,tr-2,tr-3

    \b
    # Output the updated dataset as JSON
    mlflow datasets add-traces --dataset-id d-abc123 --trace-ids tr-1 --output json
    """
    ids = [t.strip() for t in trace_ids.split(",") if t.strip()]
    if not ids:
        raise click.UsageError("--trace-ids must contain at least one trace ID.")

    try:
        dataset = get_dataset(dataset_id=dataset_id)
    except MlflowException as e:
        raise click.UsageError(f"Could not load dataset {dataset_id!r}: {e.message}")

    tracing_client = TracingClient()
    traces = []
    for trace_id in ids:
        try:
            traces.append(tracing_client.get_trace(trace_id))
        except MlflowException as e:
            raise click.UsageError(f"Could not fetch trace {trace_id!r}: {e.message}")

    # merge_records mutates self on the OSS path but returns a new object on the
    # Databricks-managed path, so reassign to report the post-merge digest/name for both.
    dataset = dataset.merge_records(traces)

    if output == "json":
        result = {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "digest": dataset.digest,
            "added_trace_ids": ids,
        }
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Added {len(ids)} trace(s) to dataset {dataset.dataset_id} ({dataset.name}).")
        click.echo(f"Digest: {dataset.digest}")
