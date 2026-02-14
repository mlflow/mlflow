"""Cursor integration for MLflow.

This module provides automatic tracing of Cursor AI agent interactions to MLflow.

Usage:
    mlflow autolog cursor [directory] [options]

After setup, your Cursor agent conversations will be automatically traced and
logged to the configured MLflow experiment.

Cursor is an AI-powered code editor built on top of VS Code. It features AI agents
that can help you write, edit, debug, and understand code through natural language
interactions. Cursor agents can execute shell commands, read and edit files, and
use MCP (Model Context Protocol) tools to assist developers.

This integration captures:
    - User prompts and agent responses
    - Agent thinking/reasoning
    - Shell command execution
    - File read and edit operations
    - MCP tool calls and results
    - Session metadata and timing

Example:
    # Set up tracing in your project directory
    mlflow autolog cursor ~/my-project

    # Set up tracing with Databricks backend
    mlflow autolog cursor -u databricks -e 123456789

    # Check tracing status
    mlflow autolog cursor --status

    # Disable tracing
    mlflow autolog cursor --disable
"""
