"""`mlflow agent` CLI group.

Wires per-subcommand modules under :mod:`mlflow.agent`. To add a new
subcommand, drop a package under ``mlflow/agent/<name>/`` and register it
here with ``commands.add_command``.
"""

from __future__ import annotations

import click

from mlflow.agent.setup.cli import setup


@click.group("agent")
def commands():
    """Coding-agent integrations for MLflow (prototype)."""


commands.add_command(setup)
