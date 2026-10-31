"""Kiro CLI integration for MLflow.

This module provides automatic tracing of Kiro CLI AI coding agent sessions
to MLflow, mirroring the ``mlflow.claude_code`` integration.

Usage:
    mlflow autolog kiro [directory] [options]

After setup, use the regular ``kiro`` command inside the configured directory
and every agent session will be automatically captured as an MLflow trace.

Example::

    # Enable tracing in the current project directory
    mlflow autolog kiro

    # Point at a specific directory with a custom tracking URI
    mlflow autolog kiro -d ~/my-project -u sqlite:///mlflow.db -n "Kiro Sessions"

    # Check status
    mlflow autolog kiro --status

    # Disable tracing
    mlflow autolog kiro --disable
"""
