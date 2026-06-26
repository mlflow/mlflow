"""CLI for managing GenAI label schemas (the "review questions" reviewers answer).

Mirrors the fluent SDK in ``mlflow.genai.label_schemas`` so the review-question
surface of the review UI is fully scriptable, and exposes each command as an MCP
tool via ``@mlflow_mcp``. Operates against the MLflow tracking store; the
Databricks ReviewApp routing in the SDK is surfaced as clean errors here.
"""

import json
from typing import Any, Literal

import click

from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.exceptions import MlflowException
from mlflow.genai.label_schemas import (
    InputCategorical,
    InputNumeric,
    InputPassFail,
    InputText,
    create_label_schema,
    delete_label_schema,
    get_label_schema,
    list_label_schemas,
    update_label_schema,
)
from mlflow.mcp.decorator import mlflow_mcp
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.string_utils import _create_table

_INPUT_VARIANTS = ("pass_fail", "categorical", "numeric", "text")
_INPUT_BUILDERS = {
    "pass_fail": InputPassFail,
    "categorical": InputCategorical,
    "numeric": InputNumeric,
    "text": InputText,
}
_INPUT_EXAMPLE = '{"variant": "categorical", "options": ["good", "bad"], "multi_select": false}'
_INPUT_HELP = (
    "Input widget spec as a JSON object with a `variant` key, one of: "
    "pass_fail (positive_label, negative_label), categorical (options, multi_select), "
    "numeric (min_value, max_value), text (max_length). "
    f"Example: '{_INPUT_EXAMPLE}'."
)
_SCHEMA_HEADERS = ["Schema ID", "Name", "Type", "Input", "Comment", "Default"]


class LabelSchemaInputParamType(click.ParamType):
    """Parse ``--input`` JSON into the matching ``Input*`` dataclass."""

    name = "input_json"

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            return value
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            self.fail(
                f"Invalid JSON. Expected a JSON object describing the input widget, "
                f"e.g. '{_INPUT_EXAMPLE}'.",
                param,
                ctx,
            )
        if not isinstance(data, dict):
            self.fail(
                f"Expected a JSON object describing the input widget, e.g. '{_INPUT_EXAMPLE}'.",
                param,
                ctx,
            )
        variant = data.pop("variant", None)
        if variant is None:
            self.fail(
                f"Missing required key 'variant'. Must be one of: {', '.join(_INPUT_VARIANTS)}.",
                param,
                ctx,
            )
        builder = _INPUT_BUILDERS.get(variant)
        if builder is None:
            self.fail(
                f"variant must be one of: {', '.join(_INPUT_VARIANTS)} (got: {variant!r}).",
                param,
                ctx,
            )
        try:
            return builder(**data)
        except TypeError as e:
            self.fail(f"Invalid fields for variant {variant!r}: {e}", param, ctx)


def _schema_to_dict(schema) -> dict[str, Any]:
    return json.loads(message_to_json(schema.to_proto()))


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


def _schema_row(d: dict[str, Any]) -> list[str]:
    variant = next((v for v in _INPUT_VARIANTS if v in d.get("input", {})), "")
    return [
        d.get("schema_id", ""),
        d.get("name", ""),
        d.get("type", ""),
        variant,
        str(d.get("enable_comment", False)),
        str(d.get("is_default", False)),
    ]


def _emit_schema(schema, output: str) -> None:
    d = _schema_to_dict(schema)
    if output == "json":
        click.echo(json.dumps(d, indent=2))
    else:
        click.echo(_create_table([_schema_row(d)], headers=_SCHEMA_HEADERS))


@click.group("label-schemas")
def commands():
    """
    Manage GenAI label schemas (the review questions reviewers answer on traces).

    To target a tracking server, set the MLFLOW_TRACKING_URI environment variable to the
    URL of the desired server.
    """


@commands.command("create")
@mlflow_mcp(tool_name="create_label_schema")
@click.option(
    "--name", required=True, type=click.STRING, help="Schema name, unique per experiment."
)
@click.option(
    "--type",
    "schema_type",
    type=click.Choice(["feedback", "expectation"]),
    required=True,
    help="Schema type.",
)
@click.option(
    "--input", "input_spec", type=LabelSchemaInputParamType(), required=True, help=_INPUT_HELP
)
@click.option("--instruction", type=click.STRING, default=None, help="Guidance shown to reviewers.")
@click.option(
    "--enable-comment/--no-enable-comment",
    default=False,
    help="Whether reviewers can add a free-form rationale.",
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
    schema_type: Literal["feedback", "expectation"],
    input_spec,
    instruction: str | None,
    enable_comment: bool,
    experiment_id: str | None,
    output: Literal["table", "json"],
) -> None:
    """
    Create a label schema.

    \b
    Examples:
    # Pass/fail question
    mlflow label-schemas create --name correctness --type feedback -x 0 \\
        --input '{"variant": "pass_fail", "positive_label": "Pass", "negative_label": "Fail"}'

    \b
    # Categorical question
    mlflow label-schemas create --name tone --type feedback -x 0 \\
        --input '{"variant": "categorical", "options": ["friendly", "neutral", "rude"]}'
    """
    schema = _run(
        create_label_schema,
        name=name,
        type=schema_type,
        input=input_spec,
        instruction=instruction,
        enable_comment=enable_comment,
        experiment_id=_require_experiment_id(experiment_id),
    )
    _emit_schema(schema, output)


@commands.command("get")
@mlflow_mcp(tool_name="get_label_schema")
@click.option("--schema-id", type=click.STRING, default=None, help="Schema ID to fetch.")
@click.option(
    "--name", type=click.STRING, default=None, help="Schema name (requires --experiment-id)."
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
    schema_id: str | None,
    name: str | None,
    experiment_id: str | None,
    output: Literal["table", "json"],
) -> None:
    """Get a label schema by --schema-id, or by --name + --experiment-id."""
    if (schema_id is None) == (name is None):
        raise click.UsageError("Provide exactly one of --schema-id or --name.")
    if name is not None:
        _require_experiment_id(experiment_id)
    schema = _run(get_label_schema, name=name, schema_id=schema_id, experiment_id=experiment_id)
    _emit_schema(schema, output)


@commands.command("list")
@mlflow_mcp(tool_name="list_label_schemas")
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    default=None,
    help="Parent experiment ID. Can be set via MLFLOW_EXPERIMENT_ID.",
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
    max_results: int,
    page_token: str | None,
    output: Literal["table", "json"],
) -> None:
    """List label schemas for an experiment."""
    schemas = _run(
        list_label_schemas,
        experiment_id=_require_experiment_id(experiment_id),
        max_results=max_results,
        page_token=page_token,
    )
    dicts = [_schema_to_dict(s) for s in schemas]
    if output == "json":
        click.echo(json.dumps({"label_schemas": dicts, "next_page_token": schemas.token}, indent=2))
    else:
        click.echo(_create_table([_schema_row(d) for d in dicts], headers=_SCHEMA_HEADERS))
        if schemas.token:
            click.echo(
                f"More results: re-run with --page-token {schemas.token} (or raise --max-results).",
                err=True,
            )


@commands.command("update")
@mlflow_mcp(tool_name="update_label_schema")
@click.option("--schema-id", required=True, type=click.STRING, help="Schema ID to update.")
@click.option("--name", type=click.STRING, default=None, help="New name.")
@click.option("--instruction", type=click.STRING, default=None, help="New reviewer guidance.")
@click.option(
    "--enable-comment/--no-enable-comment",
    "enable_comment",
    default=None,
    help="Toggle free-form rationale (unchanged if omitted).",
)
@click.option(
    "--input",
    "input_spec",
    type=LabelSchemaInputParamType(),
    default=None,
    help=f"New input widget spec (same variant as existing). {_INPUT_HELP}",
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def update(
    schema_id: str,
    name: str | None,
    instruction: str | None,
    enable_comment: bool | None,
    input_spec,
    output: Literal["table", "json"],
) -> None:
    """Sparse-update a label schema (omitted fields are unchanged)."""
    schema = _run(
        update_label_schema,
        schema_id=schema_id,
        name=name,
        instruction=instruction,
        enable_comment=enable_comment,
        input=input_spec,
    )
    _emit_schema(schema, output)


@commands.command("delete")
@mlflow_mcp(tool_name="delete_label_schema")
@click.option("--schema-id", required=True, type=click.STRING, help="Schema ID to delete.")
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def delete(schema_id: str, output: Literal["table", "json"]) -> None:
    """Delete a label schema by ID (no-op if it doesn't exist)."""
    _run(delete_label_schema, schema_id=schema_id)
    if output == "json":
        click.echo(json.dumps({"schema_id": schema_id, "deleted": True}, indent=2))
    else:
        click.echo(f"Deleted label schema {schema_id}.")
