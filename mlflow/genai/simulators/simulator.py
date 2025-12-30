import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
import pydantic

import mlflow
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint,
)
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.simulators.prompts import (
    CHECK_GOAL_PROMPT,
    DEFAULT_PERSONA,
    FOLLOWUP_USER_PROMPT,
    INITIAL_USER_PROMPT,
)
from mlflow.genai.utils.trace_utils import parse_outputs_to_str
from mlflow.metrics.genai.model_utils import _parse_model_uri
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.provider import trace_disabled
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

if TYPE_CHECKING:
    from mlflow.types.llm import ChatMessage

_logger = logging.getLogger(__name__)

_MAX_METADATA_LENGTH = 250

_MODEL_API_DOC = {
    "user_model": """User model to use for generating user messages. Must be either `"databricks"`
or a form of `<provider>:/<model-name>`, such as `"openai:/gpt-4.1-mini"`,
`"anthropic:/claude-3.5-sonnet-20240620"`. MLflow natively supports
`["openai", "anthropic", "bedrock", "mistral"]`, and more providers are supported
through `LiteLLM <https://docs.litellm.ai/docs/providers>`_.
Default model depends on the tracking URI setup:

* Databricks: `databricks`
* Otherwise: `openai:/gpt-4.1-mini`.
""",
}


class GoalCheckResult(pydantic.BaseModel):
    """Structured output for goal achievement check."""

    rationale: str = pydantic.Field(
        description="Reason for the assessment explaining whether the goal has been achieved"
    )
    result: str = pydantic.Field(description="'yes' if goal achieved, 'no' otherwise")


@contextmanager
def _suppress_tracing_warnings():
    tracing_logger = logging.getLogger("mlflow.tracing.fluent")
    original_level = tracing_logger.level
    tracing_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        tracing_logger.setLevel(original_level)


@trace_disabled  # Suppress tracing for our LLM (e.g., user message generation, stopping condition)
def _invoke_model(
    model_uri: str,
    messages: list["ChatMessage"],
    num_retries: int = 3,
    inference_params: dict[str, Any] | None = None,
    response_format: type[pydantic.BaseModel] | None = None,
) -> str | pydantic.BaseModel:
    import litellm

    provider, model_name = _parse_model_uri(model_uri)
    if provider in {"databricks", "endpoints"}:
        output = _invoke_databricks_serving_endpoint(
            model_name=model_name,
            prompt=messages,
            num_retries=num_retries,
            inference_params=inference_params,
        )
        content = output.response
        if response_format:
            return response_format.model_validate_json(content)
        return content

    litellm_messages = [litellm.Message(role=msg.role, content=msg.content) for msg in messages]

    kwargs = {
        "model": f"{provider}/{model_name}",
        "messages": litellm_messages,
        "max_retries": num_retries,
    }
    if inference_params:
        kwargs.update(inference_params)
    if response_format:
        kwargs["response_format"] = response_format

    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content

    if response_format:
        return response_format.model_validate_json(content)
    return content


def _get_last_response(conversation_history: list[dict[str, Any]]) -> str | None:
    if not conversation_history:
        return None

    last_msg = conversation_history[-1]
    content = last_msg.get("content")
    if isinstance(content, str) and content:
        return content

    result = parse_outputs_to_str(last_msg)
    if result and result.strip():
        return result

    return _format_history(conversation_history)


def _format_history(history: list[dict[str, Any]]) -> str | None:
    if not history:
        return None
    formatted = []
    for msg in history:
        role = msg.get("role") or "unknown"
        content = msg.get("content") or ""
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


@experimental(version="3.9.0")
class SimulatedUserAgent:
    """
    An LLM-powered agent that simulates user behavior in conversations.

    The agent generates realistic user messages based on a specified goal and persona,
    enabling automated testing of conversational AI systems.

    Args:
        goal: The objective the simulated user is trying to achieve in the conversation.
        persona: Description of the user's personality and background. If None, uses a
            default helpful user persona.
        model: Model URI for generating messages (e.g., "openai:/gpt-4o-mini").
            If None, uses the default model.
        **inference_params: Additional parameters passed to the LLM (e.g., temperature).
    """

    def __init__(
        self,
        goal: str,
        persona: str | None = None,
        model: str | None = None,
        **inference_params,
    ):
        self.goal = goal
        self.persona = persona or DEFAULT_PERSONA
        self.model = model or get_default_model()
        self.inference_params = inference_params

    def generate_message(
        self,
        conversation_history: list[dict[str, Any]],
        turn: int = 0,
    ) -> str:
        from mlflow.types.llm import ChatMessage

        if turn == 0:
            prompt = INITIAL_USER_PROMPT.format(persona=self.persona, goal=self.goal)
        else:
            last_response = _get_last_response(conversation_history)
            history_str = _format_history(conversation_history[:-1])
            prompt = FOLLOWUP_USER_PROMPT.format(
                persona=self.persona,
                goal=self.goal,
                conversation_history=history_str if history_str is not None else "",
                last_response=last_response if last_response is not None else "",
            )

        messages = [ChatMessage(role="user", content=prompt)]
        return _invoke_model(
            model_uri=self.model,
            messages=messages,
            num_retries=3,
            inference_params=self.inference_params,
        )


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.9.0")
class ConversationSimulator:
    """
    Generates multi-turn conversations by simulating user interactions with a target agent.

    The simulator creates a simulated user agent that interacts with your agent's predict function.
    Each conversation is traced in MLflow, allowing you to evaluate how your agent handles
    various user goals and personas.

    Args:
        test_cases: List of test case dictionaries or DataFrame. Each test case must have a
            "goal" field describing what the simulated user wants to achieve. Optional fields:
            - "persona": Custom persona for the simulated user
            - "context": Dict of additional kwargs to pass to predict_fn
        max_turns: Maximum number of conversation turns before stopping. Default is 10.
        user_model: {{ user_model }}
        **user_llm_params: Additional parameters passed to the simulated user's LLM calls.

    Example:
        .. code-block:: python
            from mlflow.genai.simulators import ConversationSimulator

            simulator = ConversationSimulator(
                test_cases=[
                    {"goal": "Learn about MLflow tracking", "persona": "A beginner data scientist"},
                    {
                        "goal": "Debug a model deployment issue",
                        "persona": "An experienced ML engineer",
                    },
                ],
                max_turns=5,
                user_model="openai:/gpt-4o-mini",
            )
    """

    def __init__(
        self,
        test_cases: list[dict[str, Any]] | pd.DataFrame,
        max_turns: int = 10,
        user_model: str | None = None,
        **user_llm_params,
    ):
        self.test_cases = self._normalize_test_cases(test_cases)
        self._validate_test_cases()
        self.max_turns = max_turns
        self.user_model = user_model or get_default_model()
        self.user_llm_params = user_llm_params

    def _normalize_test_cases(
        self, test_cases: list[dict[str, Any]] | pd.DataFrame
    ) -> list[dict[str, Any]]:
        if isinstance(test_cases, pd.DataFrame):
            return test_cases.to_dict("records")
        return test_cases

    def _validate_test_cases(self) -> None:
        if not self.test_cases:
            raise ValueError("test_cases cannot be empty")

        missing_goal_indices = [
            i for i, test_case in enumerate(self.test_cases) if not test_case.get("goal")
        ]
        if missing_goal_indices:
            raise ValueError(f"Test cases at indices {missing_goal_indices} must have 'goal' field")

    def _simulate(self, predict_fn: Callable[..., dict[str, Any]]) -> list[list[str]]:
        num_test_cases = len(self.test_cases)
        all_trace_ids: list[list[str] | None] = [None] * num_test_cases
        max_workers = min(num_test_cases, MLFLOW_GENAI_EVAL_MAX_WORKERS.get())

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="MlflowConversationSimulator",
        ) as executor:
            futures = {
                executor.submit(self._run_conversation, test_case, predict_fn): i
                for i, test_case in enumerate(self.test_cases)
            }

            for future in as_completed(futures):
                idx = futures[future]
                test_case = self.test_cases[idx]
                try:
                    trace_ids = future.result()
                    all_trace_ids[idx] = trace_ids
                except Exception as e:
                    _logger.error(
                        f"Failed to run conversation for test case {test_case.get('goal')}: {e}"
                    )
                    all_trace_ids[idx] = []

        return [ids for ids in all_trace_ids if ids is not None]

    def _run_conversation(
        self, test_case: dict[str, Any], predict_fn: Callable[..., dict[str, Any]]
    ) -> list[str]:
        goal = test_case["goal"]
        persona = test_case.get("persona")
        context = test_case.get("context", {})
        trace_session_id = f"sim-{uuid.uuid4().hex[:16]}"

        user_agent = SimulatedUserAgent(
            goal=goal,
            persona=persona,
            model=self.user_model,
            **self.user_llm_params,
        )

        conversation_history: list[dict[str, Any]] = []
        trace_ids: list[str] = []

        for turn in range(self.max_turns):
            try:
                # Generate a user message using the simulated user agent.
                user_message_content = user_agent.generate_message(conversation_history, turn)
                user_message = {"role": "user", "content": user_message_content}
                conversation_history.append(user_message)

                # Invoke the predict_fn with the user message and context.
                response, trace_id = self._invoke_predict_fn(
                    predict_fn=predict_fn,
                    input_messages=conversation_history,
                    trace_session_id=trace_session_id,
                    goal=goal,
                    persona=persona,
                    turn=turn,
                    **context,
                )

                if trace_id:
                    trace_ids.append(trace_id)

                # Parse the assistant's response and check if the goal has been achieved.
                assistant_content = parse_outputs_to_str(response)
                if not assistant_content or not assistant_content.strip():
                    _logger.debug(f"Stopping conversation: empty response at turn {turn}")
                    break
                conversation_history.append({"role": "assistant", "content": assistant_content})
                if self._check_goal_achieved(assistant_content, goal):
                    _logger.debug(f"Stopping conversation: goal achieved at turn {turn}")
                    break

            except Exception as e:
                _logger.error(f"Error during turn {turn}: {e}")
                break

        return trace_ids

    def _invoke_predict_fn(
        self,
        predict_fn: Callable[..., dict[str, Any]],
        input_messages: list[dict[str, Any]],
        trace_session_id: str,
        goal: str,
        persona: str | None,
        turn: int,
        **context,
    ) -> tuple[dict[str, Any], str | None]:
        # NB: We trace the predict_fn call to add session and simulation metadata to the trace.
        #     This adds a new root span to the trace, with the same inputs and outputs as the
        #     predict_fn call.
        @mlflow.trace(name=f"simulation_turn_{turn}")
        def traced_predict(input: list[dict[str, Any]], **context):
            mlflow.update_current_trace(
                metadata={
                    TraceMetadataKey.TRACE_SESSION: trace_session_id,
                    "mlflow.simulation.synthetic": "true",
                    "mlflow.simulation.goal": goal[:_MAX_METADATA_LENGTH],
                    "mlflow.simulation.persona": (persona or DEFAULT_PERSONA)[
                        :_MAX_METADATA_LENGTH
                    ],
                    "mlflow.simulation.turn": str(turn),
                },
            )
            return predict_fn(input=input, **context)

        response = traced_predict(input=input_messages, **context)
        return response, mlflow.get_last_active_trace_id(thread_local=True)

    def _check_goal_achieved(self, last_response: str, goal: str) -> bool:
        from mlflow.types.llm import ChatMessage

        eval_prompt = CHECK_GOAL_PROMPT.format(goal=goal, last_response=last_response)
        messages = [ChatMessage(role="user", content=eval_prompt)]

        try:
            text_result = _invoke_model(
                model_uri=self.user_model,
                messages=messages,
                num_retries=3,
                inference_params={"temperature": 0.0},
            )
            result = GoalCheckResult.model_validate_json(text_result)
            return result.result.strip().lower() == "yes"
        except pydantic.ValidationError:
            _logger.warning("Goal achievement check: could not parse response")
            return False
        except Exception as e:
            _logger.warning(f"Goal achievement check failed: {e}")
            return False
