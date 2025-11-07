"""Databricks Managed Judge adapter using databricks.agents.evals library."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.judges.utils.prompt_utils import _split_messages_for_databricks
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)


def _check_databricks_agents_installed() -> None:
    """Check if databricks-agents is installed for databricks judge functionality.

    Raises:
        MlflowException: If databricks-agents is not installed.
    """
    try:
        import databricks.agents.evals  # noqa: F401
    except ImportError:
        raise MlflowException(
            f"To use '{_DATABRICKS_DEFAULT_JUDGE_MODEL}' as the judge model, the Databricks "
            "agents library must be installed. Please install it with: "
            "`pip install databricks-agents`",
            error_code=BAD_REQUEST,
        )


def call_chat_completions(
    user_prompt: str, system_prompt: str, session_name: str | None = None
) -> Any:
    """
    Invokes the Databricks chat completions API using the databricks.agents.evals library.

    Args:
        user_prompt (str): The user prompt.
        system_prompt (str): The system prompt.
        session_name (str | None): The session name for tracking. Defaults to "mlflow-v{VERSION}".

    Returns:
        The chat completions result.

    Raises:
        MlflowException: If databricks-agents is not installed.
    """
    _check_databricks_agents_installed()

    from databricks.rag_eval import context, env_vars

    if session_name is None:
        session_name = f"mlflow-v{VERSION}"

    env_vars.RAG_EVAL_EVAL_SESSION_CLIENT_NAME.set(session_name)

    @context.eval_context
    def _call_chat_completions(user_prompt: str, system_prompt: str):
        managed_rag_client = context.get_context().build_managed_rag_client()

        return managed_rag_client.get_chat_completions_result(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

    return _call_chat_completions(user_prompt, system_prompt)


def _parse_databricks_judge_response(
    llm_output: str | None,
    assessment_name: str,
) -> Feedback:
    """
    Parse the response from Databricks judge into a Feedback object.

    Args:
        llm_output: Raw output from the LLM, or None if no response.
        assessment_name: Name of the assessment.

    Returns:
        Feedback object with parsed results or error.
    """
    source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE, source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL
    )
    if not llm_output:
        return Feedback(
            name=assessment_name,
            error="Empty response from Databricks judge",
            source=source,
        )
    try:
        response_data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        return Feedback(
            name=assessment_name,
            error=f"Invalid JSON response from Databricks judge: {e}",
            source=source,
        )
    if "result" not in response_data:
        return Feedback(
            name=assessment_name,
            error=f"Response missing 'result' field: {response_data}",
            source=source,
        )
    return Feedback(
        name=assessment_name,
        value=response_data["result"],
        rationale=response_data.get("rationale", ""),
        source=source,
    )


def _invoke_databricks_default_judge(
    prompt: str | list["ChatMessage"],
    assessment_name: str,
) -> Feedback:
    """
    Invoke the Databricks managed judge using the databricks.agents.evals library.

    Uses the direct chat completions API for clean prompt submission without
    any additional formatting or template requirements.

    Args:
        prompt: The formatted prompt with template variables filled in.
        assessment_name: The name of the assessment.

    Returns:
        Feedback object from the Databricks judge.

    Raises:
        MlflowException: If databricks-agents is not installed.
    """
    try:
        if isinstance(prompt, str):
            system_prompt = None
            user_prompt = prompt
        else:
            prompts = _split_messages_for_databricks(prompt)
            system_prompt = prompts.system_prompt
            user_prompt = prompts.user_prompt

        llm_result = call_chat_completions(user_prompt, system_prompt)
        return _parse_databricks_judge_response(llm_result.output, assessment_name)

    except Exception as e:
        _logger.debug(f"Failed to invoke Databricks judge: {e}", exc_info=True)
        return Feedback(
            name=assessment_name,
            error=f"Failed to invoke Databricks judge: {e}",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL,
            ),
        )
