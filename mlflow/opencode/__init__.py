"""Opencode integration for MLflow.

This module provides automatic tracing of Opencode conversations to MLflow.

Usage:
    mlflow autolog opencode [directory] [options]

After setup, use Opencode and traces will be automatically captured.

Opencode is a terminal-based agentic coding tool that supports multiple LLM providers.
This integration captures:
- User prompts and assistant responses
- Tool calls and their results
- Token usage per conversation
- Session metadata

Example:

```bash
# Set up tracing in current directory
mlflow autolog opencode

# Set up tracing with Databricks
mlflow autolog opencode -u databricks -e 123456789

# Disable tracing
mlflow autolog opencode --disable
```
"""
