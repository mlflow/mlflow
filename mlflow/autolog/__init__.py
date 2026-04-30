"""Shared MLflow ``autolog`` Click command group.

This module hosts the top-level ``mlflow autolog`` Click group so that
multiple integrations (Claude Code, Kiro CLI, and any future AI-agent
integration) can register sibling subcommands on a single shared group
without depending on each other at import time.

Integrations import the group and attach their own subcommand::

    from mlflow.autolog import autolog as commands

    @commands.group("my-integration", invoke_without_command=True)
    def my_integration(...):
        ...

``mlflow/cli/__init__.py`` then registers the shared group exactly once
by importing whichever integration module is installed.
"""

import click


@click.group("autolog")
def autolog():
    """Commands for autologging with MLflow."""


# Re-export for callers that prefer `from mlflow.autolog import commands`
commands = autolog
