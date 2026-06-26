"""CLI for managing GenAI review queues (expert trace-review worklists).

Mirrors the fluent SDK in ``mlflow.genai.review_queues`` so the review-queue
surface of the review UI is fully scriptable, and exposes each command as an MCP
tool via ``@mlflow_mcp``. Item-level operations live under the ``items`` subgroup.
"""

import json
from typing import Any, Literal

import click

from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.exceptions import MlflowException
from mlflow.genai.review_queues import (
    add_items_to_review_queue,
    create_review_queue,
    delete_review_queue,
    get_or_create_user_queue,
    get_review_queue,
    list_review_queue_items,
    list_review_queues,
    remove_items_from_review_queue,
    set_review_queue_item_status,
    update_review_queue,
)
from mlflow.mcp.decorator import mlflow_mcp
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.string_utils import _create_table

_QUEUE_HEADERS = ["Queue ID", "Name", "Type", "Owner", "Users", "Schemas"]
_ITEM_HEADERS = ["Item ID", "Type", "Status", "Completed By"]


def _to_dict(entity) -> dict[str, Any]:
    return json.loads(message_to_json(entity.to_proto()))


def _run(fn, **kwargs):
    try:
        return fn(**kwargs)
    except MlflowException as e:
        raise click.ClickException(str(e))


def _require_experiment_id(experiment_id: str | None) -> str:
    if not experiment_id:
        raise click.UsageError(
            "--experiment-id is required (or set the MLFLOW_EXPERIMENT_ID environment variable)."
        )
    return experiment_id


def _csv(value: str | None) -> list[str] | None:
    """Split a comma-separated flag value; None when the flag was omitted."""
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _require_csv(value: str, param: str) -> list[str]:
    items = _csv(value) or []
    if not items:
        raise click.UsageError(f"{param} must contain at least one value.")
    return items


def _queue_row(d: dict[str, Any]) -> list[str]:
    return [
        d.get("queue_id", ""),
        d.get("name", ""),
        d.get("queue_type", ""),
        d.get("created_by", "") or "",
        ", ".join(d.get("users", [])),
        ", ".join(d.get("schema_ids", [])),
    ]


def _item_row(d: dict[str, Any]) -> list[str]:
    return [
        d.get("item_id", ""),
        d.get("item_type", ""),
        d.get("status", ""),
        d.get("completed_by", "") or "",
    ]


def _emit_queue(queue, output: str) -> None:
    d = _to_dict(queue)
    if output == "json":
        click.echo(json.dumps(d, indent=2))
    else:
        click.echo(_create_table([_queue_row(d)], headers=_QUEUE_HEADERS))


@click.group("review-queues")
def commands():
    """
    Manage GenAI review queues (expert trace-review worklists).

    To target a tracking server, set the MLFLOW_TRACKING_URI environment variable to the
    URL of the desired server.
    """


@commands.command("create")
@mlflow_mcp(tool_name="create_review_queue")
@click.option("--name", required=True, type=click.STRING, help="Queue name, unique per experiment.")
@click.option(
    "--queue-type",
    type=click.Choice(["user", "custom"]),
    required=True,
    help="'user' (personal worklist, all schemas) or 'custom' (chosen users + schemas).",
)
@click.option(
    "--users",
    type=click.STRING,
    default=None,
    help="Comma-separated assigned users (custom only, max 10).",
)
@click.option(
    "--schema-ids",
    type=click.STRING,
    default=None,
    help="Comma-separated label-schema IDs (custom only).",
)
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    default=None,
    help="Parent experiment ID. Can be set via MLFLOW_EXPERIMENT_ID.",
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def create(
    name: str,
    queue_type: Literal["user", "custom"],
    users: str | None,
    schema_ids: str | None,
    experiment_id: str | None,
    output: Literal["table", "json"],
) -> None:
    """
    Create a review queue.

    \b
    Examples:
    # Personal worklist for a reviewer
    mlflow review-queues create --name alice@example.com --queue-type user -x 0

    \b
    # Custom queue with two reviewers and two questions
    mlflow review-queues create --name "weekly triage" --queue-type custom \\
        --users alice@example.com,bob@example.com --schema-ids s1,s2 -x 0
    """
    queue = _run(
        create_review_queue,
        name=name,
        queue_type=queue_type,
        users=_csv(users),
        schema_ids=_csv(schema_ids),
        experiment_id=_require_experiment_id(experiment_id),
    )
    _emit_queue(queue, output)


@commands.command("get")
@mlflow_mcp(tool_name="get_review_queue")
@click.option("--queue-id", type=click.STRING, default=None, help="Queue ID to fetch.")
@click.option(
    "--name", type=click.STRING, default=None, help="Queue name (requires --experiment-id)."
)
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    default=None,
    help="Parent experiment ID (used with --name).",
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def get(
    queue_id: str | None,
    name: str | None,
    experiment_id: str | None,
    output: Literal["table", "json"],
) -> None:
    """Get a review queue by --queue-id, or by --name + --experiment-id."""
    if (queue_id is None) == (name is None):
        raise click.UsageError("Provide exactly one of --queue-id or --name.")
    if name is not None:
        _require_experiment_id(experiment_id)
    queue = _run(get_review_queue, queue_id=queue_id, name=name, experiment_id=experiment_id)
    _emit_queue(queue, output)


@commands.command("list")
@mlflow_mcp(tool_name="list_review_queues")
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    default=None,
    help="Parent experiment ID. Can be set via MLFLOW_EXPERIMENT_ID.",
)
@click.option(
    "--user", type=click.STRING, default=None, help="Only queues this user is assigned to."
)
@click.option("--max-results", type=click.INT, default=100, help="Page size (default 100).")
@click.option(
    "--page-token", type=click.STRING, default=None, help="Continuation token from a prior call."
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def list_(
    experiment_id: str | None,
    user: str | None,
    max_results: int,
    page_token: str | None,
    output: Literal["table", "json"],
) -> None:
    """List an experiment's review queues."""
    queues = _run(
        list_review_queues,
        user=user,
        experiment_id=_require_experiment_id(experiment_id),
        max_results=max_results,
        page_token=page_token,
    )
    dicts = [_to_dict(q) for q in queues]
    if output == "json":
        click.echo(json.dumps({"review_queues": dicts, "next_page_token": queues.token}, indent=2))
    else:
        click.echo(_create_table([_queue_row(d) for d in dicts], headers=_QUEUE_HEADERS))
        if queues.token:
            click.echo(
                f"More results: re-run with --page-token {queues.token} (or raise --max-results).",
                err=True,
            )


@commands.command("update")
@mlflow_mcp(tool_name="update_review_queue")
@click.option("--queue-id", required=True, type=click.STRING, help="Queue ID to update.")
@click.option("--name", type=click.STRING, default=None, help="New name.")
@click.option("--new-owner", type=click.STRING, default=None, help="Reassign owner (needs MANAGE).")
@click.option(
    "--users", type=click.STRING, default=None, help="Comma-separated users (empty string clears)."
)
@click.option(
    "--schema-ids",
    type=click.STRING,
    default=None,
    help="Comma-separated schema IDs (empty string clears).",
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def update(
    queue_id: str,
    name: str | None,
    new_owner: str | None,
    users: str | None,
    schema_ids: str | None,
    output: Literal["table", "json"],
) -> None:
    """Update a custom queue (omitted fields unchanged; empty list clears)."""
    queue = _run(
        update_review_queue,
        queue_id=queue_id,
        name=name,
        new_owner=new_owner,
        users=_csv(users),
        schema_ids=_csv(schema_ids),
    )
    _emit_queue(queue, output)


@commands.command("delete")
@mlflow_mcp(tool_name="delete_review_queue")
@click.option("--queue-id", required=True, type=click.STRING, help="Queue ID to delete.")
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def delete(queue_id: str, output: Literal["table", "json"]) -> None:
    """Delete a review queue by ID (no-op if it doesn't exist)."""
    _run(delete_review_queue, queue_id=queue_id)
    if output == "json":
        click.echo(json.dumps({"queue_id": queue_id, "deleted": True}, indent=2))
    else:
        click.echo(f"Deleted review queue {queue_id}.")


@commands.command("get-or-create-user-queue")
@mlflow_mcp(tool_name="get_or_create_user_review_queue")
@click.option(
    "--user", required=True, type=click.STRING, help="Reviewer identifier (also the queue name)."
)
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    default=None,
    help="Parent experiment ID. Can be set via MLFLOW_EXPERIMENT_ID.",
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def get_or_create_user_queue_cmd(
    user: str, experiment_id: str | None, output: Literal["table", "json"]
) -> None:
    """Get a user's personal review queue, creating it if absent (idempotent)."""
    queue = _run(
        get_or_create_user_queue, user=user, experiment_id=_require_experiment_id(experiment_id)
    )
    _emit_queue(queue, output)


@commands.group("items")
def items():
    """Manage the items (traces) attached to a review queue."""


@items.command("add")
@mlflow_mcp(tool_name="add_review_queue_items")
@click.option("--queue-id", required=True, type=click.STRING, help="Target queue ID.")
@click.option(
    "--item-ids", required=True, type=click.STRING, help="Comma-separated trace IDs to attach."
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def items_add(queue_id: str, item_ids: str, output: Literal["table", "json"]) -> None:
    """Attach items (traces) to a queue (idempotent per item)."""
    ids = _require_csv(item_ids, "--item-ids")
    result = _run(add_items_to_review_queue, queue_id=queue_id, item_ids=ids)
    dicts = [_to_dict(i) for i in result]
    if output == "json":
        click.echo(json.dumps({"queue_id": queue_id, "items": dicts}, indent=2))
    else:
        click.echo(_create_table([_item_row(d) for d in dicts], headers=_ITEM_HEADERS))


@items.command("remove")
@mlflow_mcp(tool_name="remove_review_queue_items")
@click.option("--queue-id", required=True, type=click.STRING, help="Target queue ID.")
@click.option(
    "--item-ids", required=True, type=click.STRING, help="Comma-separated trace IDs to detach."
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def items_remove(queue_id: str, item_ids: str, output: Literal["table", "json"]) -> None:
    """Detach items from a queue (no-op for items not attached)."""
    ids = _require_csv(item_ids, "--item-ids")
    _run(remove_items_from_review_queue, queue_id=queue_id, item_ids=ids)
    if output == "json":
        click.echo(json.dumps({"queue_id": queue_id, "removed_item_ids": ids}, indent=2))
    else:
        click.echo(f"Removed {len(ids)} item(s) from review queue {queue_id}.")


@items.command("list")
@mlflow_mcp(tool_name="list_review_queue_items")
@click.option("--queue-id", required=True, type=click.STRING, help="Queue ID to list items for.")
@click.option(
    "--status",
    type=click.Choice(["pending", "complete", "declined"]),
    default=None,
    help="Filter by shared-pool status.",
)
@click.option("--max-results", type=click.INT, default=100, help="Page size (default 100).")
@click.option(
    "--page-token", type=click.STRING, default=None, help="Continuation token from a prior call."
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def items_list(
    queue_id: str,
    status: str | None,
    max_results: int,
    page_token: str | None,
    output: Literal["table", "json"],
) -> None:
    """List a queue's attached items."""
    result = _run(
        list_review_queue_items,
        queue_id=queue_id,
        status=status,
        max_results=max_results,
        page_token=page_token,
    )
    dicts = [_to_dict(i) for i in result]
    if output == "json":
        click.echo(json.dumps({"items": dicts, "next_page_token": result.token}, indent=2))
    else:
        click.echo(_create_table([_item_row(d) for d in dicts], headers=_ITEM_HEADERS))
        if result.token:
            click.echo(
                f"More results: re-run with --page-token {result.token} (or raise --max-results).",
                err=True,
            )


@items.command("set-status")
@mlflow_mcp(tool_name="set_review_queue_item_status")
@click.option("--queue-id", required=True, type=click.STRING, help="Queue ID.")
@click.option("--item-id", required=True, type=click.STRING, help="Item (trace) ID.")
@click.option(
    "--status",
    type=click.Choice(["pending", "complete", "declined"]),
    required=True,
    help="New status. 'complete'/'declined' require --completed-by; 'pending' rejects it.",
)
@click.option(
    "--completed-by", type=click.STRING, default=None, help="User completing/declining the item."
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def items_set_status(
    queue_id: str,
    item_id: str,
    status: Literal["pending", "complete", "declined"],
    completed_by: str | None,
    output: Literal["table", "json"],
) -> None:
    """Set the shared-pool status of an attached item."""
    item = _run(
        set_review_queue_item_status,
        queue_id=queue_id,
        item_id=item_id,
        status=status,
        completed_by=completed_by,
    )
    d = _to_dict(item)
    if output == "json":
        click.echo(json.dumps(d, indent=2))
    else:
        click.echo(_create_table([_item_row(d)], headers=_ITEM_HEADERS))
