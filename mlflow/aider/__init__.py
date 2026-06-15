"""Aider integration for MLflow.

This module provides automatic tracing of Aider coding sessions to MLflow.

Usage::

    import mlflow.aider
    from aider.coders import Coder
    from aider.models import Model

    mlflow.aider.autolog()

    coder = Coder.create(main_model=Model("gpt-4o"))
    coder.run("Add type hints to all functions in utils.py")

Traces will be captured in MLflow for each ``coder.run()`` call, including
the prompt, model, in-chat files, LLM response, and token usage.
"""

from mlflow.aider.autolog import autolog

__all__ = ["autolog"]
