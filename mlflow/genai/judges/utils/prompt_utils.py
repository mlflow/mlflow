"""Prompt formatting and manipulation utilities for judge models."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST

if TYPE_CHECKING:
    from mlflow.genai.judges.base import JudgeField
    from mlflow.types.llm import ChatMessage


class DatabricksLLMJudgePrompts(NamedTuple):
    """Result of splitting ChatMessage list for Databricks API."""

    system_prompt: str | None
    user_prompt: str


def format_prompt(prompt: str, **values) -> str:
    """Format double-curly variables in the prompt template."""
    for key, value in values.items():
        # Escape backslashes in the replacement string to prevent re.sub from interpreting
        # them as escape sequences (e.g. \u being treated as Unicode escape)
        replacement = str(value).replace("\\", "\\\\")
        prompt = re.sub(r"\{\{\s*" + key + r"\s*\}\}", replacement, prompt)
    return prompt


def add_output_format_instructions(prompt: str, output_fields: list["JudgeField"]) -> str:
    """
    Add structured output format instructions to a judge prompt.

    This ensures the LLM returns a JSON response with the expected fields,
    matching the expected format for the invoke_judge_model function.

    Args:
        prompt: The formatted prompt with template variables filled in
        output_fields: List of JudgeField objects defining output fields.

    Returns:
        The prompt with output format instructions appended
    """
    json_format_lines = [f'    "{field.name}": "{field.description}"' for field in output_fields]

    json_format = "{\n" + ",\n".join(json_format_lines) + "\n}"

    output_format_instructions = f"""

Please provide your assessment in the following JSON format only (no markdown):

{json_format}"""
    return prompt + output_format_instructions


def _split_messages_for_databricks(messages: list["ChatMessage"]) -> DatabricksLLMJudgePrompts:
    """
    Split a list of ChatMessage objects into system and user prompts for Databricks API.

    Args:
        messages: List of ChatMessage objects to split.

    Returns:
        DatabricksLLMJudgePrompts namedtuple with system_prompt and user_prompt fields.
        The system_prompt may be None.

    Raises:
        MlflowException: If the messages list is empty or invalid.
    """
    from mlflow.types.llm import ChatMessage

    if not messages:
        raise MlflowException(
            "Invalid prompt format: expected non-empty list of ChatMessage",
            error_code=BAD_REQUEST,
        )

    system_prompt = None
    user_parts = []

    for msg in messages:
        if isinstance(msg, ChatMessage):
            if msg.role == "system":
                # Use the first system message as the actual system prompt for the API.
                # Any subsequent system messages are appended to the user prompt to preserve
                # their content and maintain the order in which they appear in the submitted
                # evaluation payload.
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    user_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                user_parts.append(msg.content)
            elif msg.role == "assistant":
                user_parts.append(f"Assistant: {msg.content}")

    user_prompt = "\n\n".join(user_parts) if user_parts else ""

    return DatabricksLLMJudgePrompts(system_prompt=system_prompt, user_prompt=user_prompt)
