from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.judges.utils.parsing_utils import _strip_markdown_code_blocks
from mlflow.genai.simulators.prompts import DEFAULT_PERSONA, DISTILL_GOAL_AND_PERSONA_PROMPT
from mlflow.genai.simulators.simulator import _MODEL_API_DOC
from mlflow.genai.simulators.utils import format_history, invoke_model_without_tracing
from mlflow.genai.utils.trace_utils import resolve_conversation_from_session
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

if TYPE_CHECKING:
    from mlflow.entities import Trace

    Session = list[Trace]

_logger = logging.getLogger(__name__)


def _distill_goal_and_persona(
    session: Session,
    model: str,
) -> dict[str, str] | None:
    messages = resolve_conversation_from_session(session)
    if not messages:
        return None

    prompt = DISTILL_GOAL_AND_PERSONA_PROMPT.format(conversation=format_history(messages))

    from mlflow.types.llm import ChatMessage

    response = invoke_model_without_tracing(
        model_uri=model,
        messages=[ChatMessage(role="user", content=prompt)],
    )
    cleaned_response = _strip_markdown_code_blocks(response)
    try:
        result = json.loads(cleaned_response)
        goal = result.get("goal")
        if goal is None:
            _logger.debug(f"Failed to extract goal from response: {cleaned_response}")
            return None
        return {
            "goal": goal,
            "persona": result.get("persona", DEFAULT_PERSONA),
        }
    except json.JSONDecodeError as e:
        _logger.debug(f"Failed to parse response as JSON: {cleaned_response}\nError: {e}")
        return None


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
def generate_seed_conversations(
    sessions: list[Session],
    *,
    model: str | None = None,
) -> list[dict[str, str]]:
    """
    Generate seed test cases by distilling goals and personas from existing sessions.

    This function analyzes sessions and uses an LLM to infer the user's goal and
    persona from each session. This is useful for generating test cases from existing
    conversation data rather than manually writing goals and personas.

    Args:
        sessions: A list of sessions, where each session is a list of traces.
        model: {{ model }}

    Returns:
        A list of dicts with "goal" and "persona" keys, suitable for use with ConversationSimulator.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.simulators import generate_seed_conversations
            from mlflow.genai.simulators import ConversationSimulator

            # Get existing sessions
            sessions = mlflow.search_sessions(...)  # clint: disable=unknown-mlflow-function

            # Generate seed test cases
            test_cases = generate_seed_conversations(sessions)

            # Use the generated test cases with ConversationSimulator
            simulator = ConversationSimulator(test_cases=test_cases)
    """
    model = model or get_default_model()
    results = [_distill_goal_and_persona(session, model=model) for session in sessions]
    return [r for r in results if r is not None]
