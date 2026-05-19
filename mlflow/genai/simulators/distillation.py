from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pydantic

from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.simulators.prompts import DISTILL_GOAL_AND_PERSONA_PROMPT
from mlflow.genai.simulators.simulator import _MODEL_API_DOC, PGBAR_FORMAT
from mlflow.genai.simulators.utils import (
    format_history,
    get_default_simulation_model,
    invoke_model_without_tracing,
)
from mlflow.genai.utils.trace_utils import resolve_conversation_from_session
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

if TYPE_CHECKING:
    from mlflow.entities import Trace
    from mlflow.entities.session import Session

_logger = logging.getLogger(__name__)


class _GoalAndPersona(pydantic.BaseModel):
    goal: str = pydantic.Field(description="The user's underlying goal in the conversation")
    persona: str | None = pydantic.Field(
        default=None,
        description="A description of the user's communication style and personality",
    )
    simulation_guidelines: list[str] | None = pydantic.Field(
        default=None,
        description="List of guidelines for how a simulated user should conduct this conversation",
    )


def _distill_goal_and_persona(
    session: "Session | list[Trace]",
    model: str,
) -> dict[str, str] | None:
    from mlflow.entities.session import Session
    from mlflow.types.llm import ChatMessage

    traces = session.traces if isinstance(session, Session) else session
    messages = resolve_conversation_from_session(traces)
    if not messages:
        return None

    prompt = DISTILL_GOAL_AND_PERSONA_PROMPT.format(conversation=format_history(messages))

    try:
        response = invoke_model_without_tracing(
            model_uri=model,
            messages=[ChatMessage(role="user", content=prompt)],
            response_format=_GoalAndPersona,
        )
        result = _GoalAndPersona.model_validate_json(response)
        if not result.goal:
            _logger.debug(f"Empty goal extracted from response: {response}")
            return None
        test_case = {"goal": result.goal}
        if result.persona:
            test_case["persona"] = result.persona
        if result.simulation_guidelines:
            test_case["simulation_guidelines"] = result.simulation_guidelines
        return test_case
    except pydantic.ValidationError as e:
        _logger.debug(f"Failed to validate response: {e}")
        return None


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
def generate_test_cases(
    sessions: "list[Session] | list[list[Trace]]",
    *,
    model: str | None = None,
) -> list[dict[str, str]]:
    """
    Generate seed test cases by distilling goals and personas from existing sessions.

    This function analyzes sessions and uses an LLM to infer the user's goal and
    persona from each session. This is useful for generating test cases from existing
    conversation data rather than manually writing goals and personas.

    .. note::
        This task benefits from a powerful model. We recommend using ``openai:/gpt-5``
        or a model of similar capability for best results.

    Args:
        sessions: A list of :py:class:`~mlflow.entities.session.Session` objects or
            a list of trace lists (where each inner list contains traces from one session).
        model: {{ model }}

    Returns:
        A list of dicts with "goal", "persona", and "simulation_guidelines" keys,
        suitable for use with :py:class:`~mlflow.genai.simulators.ConversationSimulator`.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.simulators import generate_test_cases
            from mlflow.genai.simulators import ConversationSimulator

            # Get existing sessions
            sessions = mlflow.search_sessions(...)

            # Generate seed test cases
            test_cases = generate_test_cases(sessions)

            # Use the generated test cases with ConversationSimulator
            simulator = ConversationSimulator(test_cases=test_cases)

        To save test cases as an evaluation dataset for reuse:

        .. code-block:: python

            from mlflow.genai.datasets import create_dataset

            # Create a dataset and save the test cases
            dataset = create_dataset(name="my_test_cases")
            dataset.merge_records([{"inputs": tc} for tc in test_cases])
    """
    model = model or get_default_simulation_model()
    num_sessions = len(sessions)
    results: list[dict[str, str] | None] = [None] * num_sessions
    max_workers = min(num_sessions, MLFLOW_GENAI_EVAL_MAX_WORKERS.get())

    progress_bar = (
        tqdm(
            total=num_sessions,
            desc="Generating test cases",
            bar_format=PGBAR_FORMAT,
        )
        if tqdm
        else None
    )

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="MlflowTestCaseGeneration",
    ) as executor:
        futures = {
            executor.submit(_distill_goal_and_persona, session, model): i
            for i, session in enumerate(sessions)
        }
        try:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    _logger.error(f"Failed to distill test case for session {idx}: {e}")
                if progress_bar:
                    progress_bar.update(1)
        finally:
            if progress_bar:
                progress_bar.close()

    return [r for r in results if r is not None]
