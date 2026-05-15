"""CLI for the agent_playground feature.

``mlflow agent test list|delete`` mirrors the v1 CRUD router endpoints
so headless / scripted workflows have parity with the playground UI.
Other subcommands (``add``, ``run``, ``export``) land in subsequent
stacks.
"""

from __future__ import annotations

import json
from typing import Literal

import click

from mlflow.agent_playground.test_cases import store
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.exceptions import MlflowException
from mlflow.utils.string_utils import _create_table


@click.group("agent")
def commands():
    """Manage the agent_playground feature (test cases, runs, export)."""


@commands.group("test")
def test_commands():
    """Manage regression test cases for the connected agent."""


@test_commands.command("list")
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    required=True,
    help="Experiment whose regression suite to list.",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format. Defaults to a human-readable table.",
)
def list_test_cases(experiment_id: str, output: Literal["table", "json"]) -> None:
    """List regression test cases for an experiment."""
    try:
        cases = store.list_cases(experiment_id)
    except MlflowException as exc:
        raise click.ClickException(exc.message) from exc
    if output == "json":
        click.echo(json.dumps([case.model_dump() for case in cases], indent=2, default=str))
        return

    if not cases:
        click.echo(f"No test cases found in experiment {experiment_id!r}.")
        return

    headers = ["test_case_id", "strategy", "promoted", "rationale_summary"]
    rows = [
        [
            case.test_case_id,
            case.spec.strategy,
            "yes" if case.promoted else "no",
            case.spec.rationale_summary,
        ]
        for case in cases
    ]
    click.echo(_create_table(rows, headers))


@test_commands.command("delete")
@click.argument("test_case_id", type=click.STRING)
@click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    required=True,
    help="Experiment that owns the test case.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the interactive confirmation prompt.",
)
def delete_test_case(test_case_id: str, experiment_id: str, yes: bool) -> None:
    """Hard-delete a regression test case by id."""
    if not yes:
        click.confirm(
            f"Delete test case {test_case_id!r} from experiment {experiment_id!r}?",
            abort=True,
        )

    try:
        deleted = store.delete_case(experiment_id, test_case_id)
    except MlflowException as exc:
        raise click.ClickException(exc.message) from exc
    except NotImplementedError as exc:
        raise click.ClickException(
            f"Delete not supported on the current tracking backend: {exc}"
        ) from exc

    if not deleted:
        raise click.ClickException(
            f"Test case {test_case_id!r} not found in experiment {experiment_id!r}"
        )
    click.echo(f"Deleted test case {test_case_id!r}.")
