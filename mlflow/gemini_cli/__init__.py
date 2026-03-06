"""Gemini CLI integration for MLflow.

This module provides automatic tracing of Gemini CLI conversations to MLflow.

Usage:
    mlflow autolog gemini-cli [directory] [options]

After setup, use the regular 'gemini' command and traces will be automatically captured.

Example:

.. code-block:: bash

    # Set up tracing in current directory
    mlflow autolog gemini-cli

    # Use Gemini CLI normally
    gemini "help me refactor this function"

    # View traces
    mlflow server
"""
