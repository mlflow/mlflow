"""Core module for managing MLflow commands."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse frontmatter from markdown content.

    Args:
        content: Markdown content with optional YAML frontmatter.

    Returns:
        Tuple of (metadata dict, body content).
    """
    if not content.startswith("---"):
        return {}, content

    match = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
    if not match:
        return {}, content

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        # If YAML parsing fails, return empty metadata
        return {}, content

    body = match.group(2)
    return metadata, body


def list_commands(namespace: str | None = None) -> list[dict[str, Any]]:
    """List all available commands with metadata.

    Args:
        namespace: Optional namespace to filter commands.

    Returns:
        List of command dictionaries with keys: key, namespace, description.
    """
    # We're in mlflow/commands/core.py, so parent is mlflow/commands/
    commands_dir = Path(__file__).parent
    commands = []

    if not commands_dir.exists():
        return commands

    for md_file in commands_dir.glob("**/*.md"):
        try:
            content = md_file.read_text()
            metadata, _ = parse_frontmatter(content)

            # Build command key from path (e.g., genai/analyze_experiment)
            relative_path = md_file.relative_to(commands_dir)
            # Use forward slashes consistently across platforms
            command_key = str(relative_path.with_suffix("")).replace(os.sep, "/")

            # Filter by namespace if specified
            if namespace and not command_key.startswith(f"{namespace}/"):
                continue

            commands.append(
                {
                    "key": command_key,
                    "namespace": metadata.get("namespace", ""),
                    "description": metadata.get("description", "No description"),
                }
            )
        except Exception:
            # Skip files that can't be read or parsed
            continue

    return sorted(commands, key=lambda x: x["key"])


def get_command(key: str) -> str:
    """Get command content by key.

    Args:
        key: Command key (e.g., 'genai/analyze_experiment').

    Returns:
        Full markdown content of the command.

    Raises:
        FileNotFoundError: If command not found.
    """
    # We're in mlflow/commands/core.py, so parent is mlflow/commands/
    commands_dir = Path(__file__).parent
    # Convert forward slashes to OS-specific separators for file path
    key_parts = key.split("/")
    command_path = commands_dir.joinpath(*key_parts).with_suffix(".md")

    if not command_path.exists():
        raise FileNotFoundError(f"Command '{key}' not found")

    return command_path.read_text()


def get_command_body(key: str) -> str:
    """Get command body content without frontmatter.

    Args:
        key: Command key (e.g., 'genai/analyze_experiment').

    Returns:
        Command body content without YAML frontmatter.

    Raises:
        FileNotFoundError: If command not found.
    """
    content = get_command(key)
    _, body = parse_frontmatter(content)
    return body
