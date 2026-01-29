from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable

import pydantic

import mlflow
from mlflow.environment_variables import MLFLOW_GENAI_SIMULATOR_MAX_WORKERS
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import EvaluationDataset
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
    create_litellm_message_from_databricks_response,
    serialize_messages_to_databricks_prompts,
)
from mlflow.genai.judges.constants import (
    _DATABRICKS_AGENTIC_JUDGE_MODEL,
    _DATABRICKS_DEFAULT_JUDGE_MODEL,
)
from mlflow.genai.simulators.prompts import (
    CHECK_GOAL_PROMPT,
    DEFAULT_PERSONA,
    FOLLOWUP_USER_PROMPT,
    INITIAL_USER_PROMPT,
)
from mlflow.genai.simulators.utils import get_default_simulation_model
from mlflow.genai.utils.trace_utils import parse_outputs_to_str
from mlflow.telemetry.events import SimulateConversationEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

if TYPE_CHECKING:
    from pandas import DataFrame

    from mlflow.entities import Trace
    from mlflow.types.llm import ChatMessage

_logger = logging.getLogger(__name__)

_MAX_METADATA_LENGTH = 250
_EXPECTED_TEST_CASE_KEYS = {"goal", "persona", "context", "expectations"}
_REQUIRED_TEST_CASE_KEYS = {"goal"}

PGBAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}, Remaining: {remaining}] {postfix}"
)


@dataclass
class SimulationTimingTracker:
    _lock: Lock = field(default_factory=Lock, repr=False)
    predict_fn_seconds: float = 0.0
    generate_message_seconds: float = 0.0
    check_goal_seconds: float = 0.0

    def add(
        self,
        predict_fn_seconds: float = 0,
        generate_message_seconds: float = 0,
        check_goal_seconds: float = 0,
    ):
        with self._lock:
            self.predict_fn_seconds += predict_fn_seconds
            self.generate_message_seconds += generate_message_seconds
            self.check_goal_seconds += check_goal_seconds

    def format_postfix(self) -> str:
        with self._lock:
            simulator_seconds = self.generate_message_seconds + self.check_goal_seconds
            total = self.predict_fn_seconds + simulator_seconds
            if total == 0:
                return "(predict: 0%, simulator: 0%)"
            predict_pct = 100 * self.predict_fn_seconds / total
            simulator_pct = 100 * simulator_seconds / total
            return f"(predict: {predict_pct:.1f}%, simulator: {simulator_pct:.1f}%)"


_MODEL_API_DOC = {
    "model": """Model to use for generating user messages. Must be one of:

* `"databricks"` - Uses the Databricks managed LLM endpoint
* `"databricks:/<endpoint-name>"` - Uses a Databricks model serving endpoint \
(e.g., `"databricks:/databricks-claude-sonnet-4-5"`)
* `"<provider>:/<model-name>"` - Uses LiteLLM (e.g., `"openai:/gpt-4.1-mini"`, \
`"anthropic:/claude-3.5-sonnet-20240620"`)

MLflow natively supports `["openai", "anthropic", "bedrock", "mistral"]`, and more \
providers are supported through `LiteLLM <https://docs.litellm.ai/docs/providers>`_.

Default model depends on the tracking URI setup:

* Databricks: `"databricks"`
* Otherwise: `"openai:/gpt-4.1-mini"`
""",
}


class GoalCheckResult(pydantic.BaseModel):
    """Structured output for goal achievement check."""

    rationale: str = pydantic.Field(
        description="Reason for the assessment explaining whether the goal has been achieved"
    )
    result: str = pydantic.Field(description="'yes' if goal achieved, 'no' otherwise")


@contextmanager
def _suppress_tracing_logging():
    # Suppress INFO logs when flushing traces from the async trace export queue.
    async_logger = logging.getLogger("mlflow.tracing.export.async_export_queue")
    # Suppress WARNING logs when the tracing provider is automatically traced, but used in a
    # tracing-disabled context (e.g., generating user messages).
    fluent_logger = logging.getLogger("mlflow.tracing.fluent")
    original_async_level = async_logger.level
    original_fluent_level = fluent_logger.level

    async_logger.setLevel(logging.WARNING)
    fluent_logger.setLevel(logging.ERROR)

    try:
        yield
    finally:
        async_logger.setLevel(original_async_level)
        fluent_logger.setLevel(original_fluent_level)


@contextmanager
def _delete_trace_if_created():
    """Delete any trace created within this context to avoid polluting user traces."""
    trace_id_before = mlflow.get_last_active_trace_id(thread_local=True)
    try:
        yield
    finally:
        trace_id_after = mlflow.get_last_active_trace_id(thread_local=True)
        if trace_id_after and trace_id_after != trace_id_before:
            try:
                mlflow.delete_trace(trace_id_after)
            except Exception as e:
                _logger.debug(f"Failed to delete trace {trace_id_after}: {e}")


# TODO: Refactor judges adapters to support returning raw responses, then use them here
#       instead of reimplementing the invocation logic.
def _invoke_model_without_tracing(
    model_uri: str,
    messages: list["ChatMessage"],
    num_retries: int = 3,
    inference_params: dict[str, Any] | None = None,
) -> str:
    """
    Invoke a model without tracing. This method will delete the last trace created by the
    invocation, if any.
    """
    import litellm

    from mlflow.metrics.genai.model_utils import _parse_model_uri

    with _delete_trace_if_created():
        # Use Databricks managed endpoint with agentic model for the default "databricks" URI
        if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
            user_prompt, system_prompt = serialize_messages_to_databricks_prompts(messages)

            result = call_chat_completions(
                user_prompt=user_prompt,
                # NB: We cannot use an empty system prompt here so we use a period.
                system_prompt=system_prompt or ".",
                model=_DATABRICKS_AGENTIC_JUDGE_MODEL,
            )
            if getattr(result, "error_code", None):
                raise MlflowException(
                    f"Failed to get chat completions result from Databricks managed endpoint: "
                    f"[{result.error_code}] {result.error_message}"
                )

            output_json = result.output_json
            if not output_json:
                raise MlflowException("Empty response from Databricks managed endpoint")

            parsed_json = json.loads(output_json) if isinstance(output_json, str) else output_json
            return create_litellm_message_from_databricks_response(parsed_json).content

        provider, model_name = _parse_model_uri(model_uri)

        # Use LiteLLM for other providers (including Databricks served models)
        litellm_messages = [litellm.Message(role=msg.role, content=msg.content) for msg in messages]

        kwargs = {
            "model": f"{provider}/{model_name}",
            "messages": litellm_messages,
            "max_retries": num_retries,
            # Drop unsupported params (e.g., temperature=0 for certain models)
            "drop_params": True,
        }
        if inference_params:
            kwargs.update(inference_params)

        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            # Check if error is about unsupported temperature parameter:
            # "Unsupported value: 'temperature' does not support 0.0 with this model"
            error_str = str(e)
            if inference_params and "Unsupported value: 'temperature'" in error_str:
                kwargs.pop("temperature", None)
                response = litellm.completion(**kwargs)
                return response.choices[0].message.content
            else:
                raise


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

    return str(last_msg)


def _format_history(history: list[dict[str, Any]]) -> str | None:
    if not history:
        return None
    formatted = []
    for msg in history:
        role = msg.get("role") or "unknown"
        content = msg.get("content") or ""
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def _fetch_traces(all_trace_ids: list[list[str]]) -> list[list["Trace"]]:
    from mlflow.tracing.client import TracingClient

    flat_trace_ids = [tid for trace_ids in all_trace_ids for tid in trace_ids]
    if not flat_trace_ids:
        raise MlflowException(
            "Simulation produced no traces. This may indicate that all conversations failed during "
            "simulation. Check the logs above for error details."
        )

    mlflow.flush_trace_async_logging()

    client = TracingClient()
    max_workers = min(len(flat_trace_ids), MLFLOW_GENAI_SIMULATOR_MAX_WORKERS.get())
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="ConversationSimulatorTraceFetcher"
    ) as executor:
        flat_traces = list(executor.map(client.get_trace, flat_trace_ids))

    all_traces: list[list["Trace"]] = []
    idx = 0
    for trace_ids in all_trace_ids:
        all_traces.append(flat_traces[idx : idx + len(trace_ids)])
        idx += len(trace_ids)

    return all_traces


@experimental(version="3.10.0")
@dataclass(frozen=True)
class SimulatorContext:
    """
    Context information passed to simulated user agents for message generation.

    This dataclass bundles all input information needed for a simulated user to
    generate their next message in a conversation.

    Args:
        goal: The objective the simulated user is trying to achieve.
        persona: Description of the user's personality and background.
        conversation_history: The full conversation history as a list of message dicts.
        turn: The current turn number (0-indexed).
    """

    goal: str
    persona: str
    conversation_history: list[dict[str, Any]]
    turn: int

    @property
    def is_first_turn(self) -> bool:
        return self.turn == 0

    @property
    def formatted_history(self) -> str | None:
        return _format_history(self.conversation_history)

    @property
    def last_assistant_response(self) -> str | None:
        if not self.conversation_history:
            return None
        return _get_last_response(self.conversation_history)


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.10.0")
class BaseSimulatedUserAgent(ABC):
    """
    Abstract base class for simulated user agents.

    Subclass this to create custom simulated user implementations with specialized
    behavior. The base class provides common functionality like LLM invocation and
    context construction.

    Args:
        goal: The objective the simulated user is trying to achieve in the conversation.
        persona: Description of the user's personality and background. If None, uses a
            default helpful user persona.
        model: {{ model }}
        **inference_params: Additional parameters passed to the LLM (e.g., temperature).

    Example:
        .. code-block:: python

            from mlflow.genai.simulators import BaseSimulatedUserAgent, SimulatorContext


            class ImpatientUserAgent(BaseSimulatedUserAgent):
                def generate_message(self, context: SimulatorContext) -> str:
                    if context.is_first_turn:
                        return f"I need help NOW with: {context.goal}"
                    return self.invoke_llm(
                        f"Respond impatiently. Goal: {context.goal}. "
                        f"Last response: {context.last_assistant_response}"
                    )
    """

    def __init__(
        self,
        model: str | None = None,
        **inference_params,
    ):
        self.model = model or get_default_simulation_model()
        self.inference_params = inference_params

    @abstractmethod
    def generate_message(self, context: SimulatorContext) -> str:
        """
        Generate a user message based on the provided context.

        Args:
            context: A SimulatorContext containing information like goal, persona,
                     conversation history, and turn.

        Returns:
            The generated user message string.
        """

    def invoke_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        from mlflow.types.llm import ChatMessage

        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        return _invoke_model_without_tracing(
            model_uri=self.model,
            messages=messages,
            num_retries=3,
            inference_params=self.inference_params,
        )


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.10.0")
class SimulatedUserAgent(BaseSimulatedUserAgent):
    """
    An LLM-powered agent that simulates user behavior in conversations.

    The agent generates realistic user messages based on a specified goal and persona,
    enabling automated testing of conversational AI systems.

    Args:
        goal: The objective the simulated user is trying to achieve in the conversation.
        persona: Description of the user's personality and background. If None, uses a
            default helpful user persona.
        model: {{ model }}
        **inference_params: Additional parameters passed to the LLM (e.g., temperature).
    """

    def generate_message(self, context: SimulatorContext) -> str:
        if context.is_first_turn:
            prompt = INITIAL_USER_PROMPT.format(persona=context.persona, goal=context.goal)
        else:
            history_without_last = context.conversation_history[:-1]
            history_str = _format_history(history_without_last)
            prompt = FOLLOWUP_USER_PROMPT.format(
                persona=context.persona,
                goal=context.goal,
                conversation_history=history_str if history_str is not None else "",
                last_response=context.last_assistant_response or "",
            )

        return self.invoke_llm(prompt)


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.9.0")
class ConversationSimulator:
    """
    Generates multi-turn conversations by simulating user interactions with a target agent.

    The simulator creates a simulated user agent that interacts with your agent's predict function.
    Each conversation is traced in MLflow, allowing you to evaluate how your agent handles
    various user goals and personas.

    The predict function passed to the simulator must follow the OpenAI Responses API format
    (https://platform.openai.com/docs/api-reference/responses):

    - It must accept an ``input`` parameter containing the conversation history
      as a list of message dictionaries (e.g., ``[{"role": "user", "content": "..."}]``)
    - It may accept additional keyword arguments from the test case's ``context`` field
    - It receives an ``mlflow_session_id`` parameter that uniquely identifies the conversation
      session. This ID is consistent across all turns in the same conversation, allowing you
      to associate related traces or maintain stateful context (e.g., for thread-based agents).
    - It should return a response (the assistant's message content will be extracted)

    Args:
        test_cases: List of test case dicts, a DataFrame, or an EvaluationDataset,
            with the following fields:

            - "goal": Describing what the simulated user wants to achieve.
            - "persona" (optional): Custom persona for the simulated user.
            - "context" (optional): Dict of additional kwargs to pass to predict_fn.
            - "expectations" (optional): Dict of expected values (ground truth) for
              session-level evaluation. These are logged to the first trace of the
              session with the session ID in metadata, allowing session-level scorers
              to retrieve them.

        max_turns: Maximum number of conversation turns before stopping. Default is 10.
        user_model: {{ model }}
        user_agent_class: Optional custom simulated user agent class. Must be a subclass
            of :py:class:`BaseSimulatedUserAgent`. If not provided, uses the default
            :py:class:`SimulatedUserAgent`.
        **user_llm_params: Additional parameters passed to the simulated user's LLM calls.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.simulators import ConversationSimulator
            from mlflow.genai.scorers import ConversationalSafety, Safety


            def predict_fn(input: list[dict], **kwargs) -> dict:
                # The mlflow_session_id uniquely identifies this conversation session.
                # All turns in the same conversation share the same session ID.
                session_id = kwargs.get("mlflow_session_id")

                # You can use this to maintain state across turns, e.g.:
                # thread = get_or_create_thread(session_id)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=input,
                )
                return response


            # Each test case requires a "goal". "persona", "context", and "expectations"
            # are optional.
            simulator = ConversationSimulator(
                test_cases=[
                    {"goal": "Learn about MLflow tracking"},
                    {"goal": "Debug deployment issue", "persona": "A data scientist"},
                    {
                        "goal": "Set up model registry",
                        "persona": "A beginner",
                        "context": {"user_id": "123"},
                        "expectations": {"expected_topic": "model registry"},
                    },
                ],
                max_turns=5,
            )

            mlflow.genai.evaluate(
                data=simulator,
                predict_fn=predict_fn,
                scorers=[ConversationalSafety(), Safety()],
            )
    """

    def __init__(
        self,
        test_cases: list[dict[str, Any]] | "DataFrame" | EvaluationDataset,
        max_turns: int = 10,
        user_model: str | None = None,
        user_agent_class: type[BaseSimulatedUserAgent] | None = None,
        **user_llm_params,
    ):
        if user_agent_class is not None and not issubclass(
            user_agent_class, BaseSimulatedUserAgent
        ):
            raise TypeError(
                f"user_agent_class must be a subclass of BaseSimulatedUserAgent, "
                f"got {user_agent_class.__name__}"
            )
        self.test_cases = test_cases
        self.max_turns = max_turns
        self.user_model = user_model or get_default_simulation_model()
        self.user_agent_class = user_agent_class or SimulatedUserAgent
        self.user_llm_params = user_llm_params

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "test_cases":
            value = self._normalize_test_cases(value)
            self._validate_test_cases(value)
        super().__setattr__(name, value)

    def _normalize_test_cases(
        self, test_cases: list[dict[str, Any]] | "DataFrame" | EvaluationDataset
    ) -> list[dict[str, Any]]:
        from pandas import DataFrame

        if isinstance(test_cases, EvaluationDataset):
            records = test_cases.to_df()["inputs"].to_list()
            if not records or not (records[0].keys() & _REQUIRED_TEST_CASE_KEYS):
                raise ValueError(
                    "EvaluationDataset passed to ConversationSimulator must contain "
                    "conversational test cases with a 'goal' field in the 'inputs' column"
                )
            return records

        if isinstance(test_cases, DataFrame):
            return test_cases.to_dict("records")

        return test_cases

    def _validate_test_cases(self, test_cases: list[dict[str, Any]]) -> None:
        if not test_cases:
            raise ValueError("test_cases cannot be empty")

        missing_goal_indices = [
            i for i, test_case in enumerate(test_cases) if not test_case.get("goal")
        ]
        if missing_goal_indices:
            raise ValueError(f"Test cases at indices {missing_goal_indices} must have 'goal' field")

        indices_with_extra_keys = [
            i
            for i, test_case in enumerate(test_cases)
            if set(test_case.keys()) - _EXPECTED_TEST_CASE_KEYS
        ]
        if indices_with_extra_keys:
            _logger.warning(
                f"Test cases at indices {indices_with_extra_keys} contain unexpected keys "
                f"which will be ignored. Expected keys: {_EXPECTED_TEST_CASE_KEYS}."
            )

    def _compute_test_case_digest(self) -> str:
        """Compute a digest based on the test cases for consistent dataset identification.

        This ensures the same test cases produce the same digest regardless of
        simulation output variations caused by LLM non-determinism.
        """
        import pandas as pd

        from mlflow.data.digest_utils import compute_pandas_digest

        test_case_df = pd.DataFrame(self.test_cases)
        return compute_pandas_digest(test_case_df)

    @experimental(version="3.10.0")
    @record_usage_event(SimulateConversationEvent)
    def simulate(self, predict_fn: Callable[..., dict[str, Any]]) -> list[list["Trace"]]:
        """
        Run conversation simulations for all test cases.

        Executes the simulated user agent against the provided predict function
        for each test case, generating multi-turn conversations. Each conversation
        is traced in MLflow.

        Args:
            predict_fn: The target function to evaluate. Must accept an ``input``
                parameter containing the conversation history as a list of message
                dicts, and may accept additional kwargs from the test case's context.

        Returns:
            A list of lists containing Trace objects. Each inner list corresponds to
            a test case and contains the traces for each turn in that conversation.
        """
        num_test_cases = len(self.test_cases)
        all_trace_ids: list[list[str]] = [[] for _ in range(num_test_cases)]
        max_workers = min(num_test_cases, MLFLOW_GENAI_SIMULATOR_MAX_WORKERS.get())
        timings = SimulationTimingTracker()

        progress_bar = (
            tqdm(
                total=num_test_cases,
                desc="Simulating conversations",
                bar_format=PGBAR_FORMAT,
                postfix=timings.format_postfix(),
            )
            if tqdm
            else None
        )

        with (
            _suppress_tracing_logging(),
            ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="MlflowConversationSimulator",
            ) as executor,
        ):
            futures = {
                executor.submit(self._run_conversation, test_case, predict_fn, timings): i
                for i, test_case in enumerate(self.test_cases)
            }
            try:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        all_trace_ids[idx] = future.result()
                    except Exception as e:
                        _logger.error(
                            f"Failed to run conversation for test case "
                            f"{self.test_cases[idx].get('goal')}: {e}"
                        )
                    if progress_bar:
                        progress_bar.set_postfix_str(timings.format_postfix(), refresh=False)
                        progress_bar.update(1)
            finally:
                if progress_bar:
                    progress_bar.close()

        return _fetch_traces(all_trace_ids)

    def _run_conversation(
        self,
        test_case: dict[str, Any],
        predict_fn: Callable[..., dict[str, Any]],
        timings: SimulationTimingTracker,
    ) -> list[str]:
        goal = test_case["goal"]
        persona = test_case.get("persona") or DEFAULT_PERSONA
        context = test_case.get("context", {})
        expectations = test_case.get("expectations", {})
        trace_session_id = f"sim-{uuid.uuid4().hex[:16]}"

        user_agent = self.user_agent_class(
            model=self.user_model,
            **self.user_llm_params,
        )

        conversation_history: list[dict[str, Any]] = []
        trace_ids: list[str] = []

        for turn in range(self.max_turns):
            try:
                start_time = time.perf_counter()
                simulator_context = SimulatorContext(
                    goal=goal,
                    persona=persona,
                    conversation_history=conversation_history,
                    turn=turn,
                )
                user_message_content = user_agent.generate_message(simulator_context)
                timings.add(generate_message_seconds=time.perf_counter() - start_time)

                user_message = {"role": "user", "content": user_message_content}
                conversation_history.append(user_message)

                start_time = time.perf_counter()
                response, trace_id = self._invoke_predict_fn(
                    predict_fn=predict_fn,
                    input_messages=conversation_history,
                    trace_session_id=trace_session_id,
                    goal=goal,
                    persona=persona,
                    context=context,
                    expectations=expectations if turn == 0 else None,
                    turn=turn,
                )
                timings.add(predict_fn_seconds=time.perf_counter() - start_time)

                if trace_id:
                    trace_ids.append(trace_id)

                assistant_content = parse_outputs_to_str(response)
                if not assistant_content or not assistant_content.strip():
                    _logger.debug(f"Stopping conversation: empty response at turn {turn}")
                    break
                conversation_history.append({"role": "assistant", "content": assistant_content})

                start_time = time.perf_counter()
                goal_achieved = self._check_goal_achieved(
                    conversation_history, assistant_content, goal
                )
                timings.add(check_goal_seconds=time.perf_counter() - start_time)

                if goal_achieved:
                    _logger.debug(f"Stopping conversation: goal achieved at turn {turn}")
                    break

            except Exception as e:
                _logger.error(f"Error during turn {turn}: {e}", exc_info=True)
                break

        return trace_ids

    def _invoke_predict_fn(
        self,
        predict_fn: Callable[..., dict[str, Any]],
        input_messages: list[dict[str, Any]],
        trace_session_id: str,
        goal: str,
        persona: str | None,
        context: dict[str, Any],
        expectations: dict[str, Any] | None,
        turn: int,
    ) -> tuple[dict[str, Any], str | None]:
        # NB: We trace the predict_fn call to add session and simulation metadata to the trace.
        #     This adds a new root span to the trace, with the same inputs and outputs as the
        #     predict_fn call. The goal/persona/turn metadata is used for trace comparison UI
        #     since message content may differ between simulation runs.
        @mlflow.trace(name=f"simulation_turn_{turn}", span_type="CHAIN")
        def traced_predict(input: list[dict[str, Any]], **ctx):
            mlflow.update_current_trace(
                metadata={
                    TraceMetadataKey.TRACE_SESSION: trace_session_id,
                    "mlflow.simulation.goal": goal[:_MAX_METADATA_LENGTH],
                    "mlflow.simulation.persona": (persona or DEFAULT_PERSONA)[
                        :_MAX_METADATA_LENGTH
                    ],
                    "mlflow.simulation.turn": str(turn),
                },
            )
            if span := mlflow.get_current_active_span():
                span.set_attributes(
                    {
                        "mlflow.simulation.goal": goal,
                        "mlflow.simulation.persona": persona or DEFAULT_PERSONA,
                        "mlflow.simulation.context": context,
                    }
                )
            return predict_fn(input=input, **ctx)

        response = traced_predict(
            input=input_messages, mlflow_session_id=trace_session_id, **context
        )
        trace_id = mlflow.get_last_active_trace_id(thread_local=True)

        # Log expectations to the first trace of the session
        if expectations and trace_id:
            for name, value in expectations.items():
                mlflow.log_expectation(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    metadata={TraceMetadataKey.TRACE_SESSION: trace_session_id},
                )

        return response, trace_id

    def _check_goal_achieved(
        self,
        conversation_history: list[dict[str, Any]],
        last_response: str,
        goal: str,
    ) -> bool:
        from mlflow.types.llm import ChatMessage

        history_str = _format_history(conversation_history)
        eval_prompt = CHECK_GOAL_PROMPT.format(
            goal=goal,
            conversation_history=history_str if history_str is not None else "",
            last_response=last_response,
        )
        messages = [ChatMessage(role="user", content=eval_prompt)]

        try:
            text_result = _invoke_model_without_tracing(
                model_uri=self.user_model,
                messages=messages,
                num_retries=3,
                inference_params={"temperature": 0.0, "response_format": GoalCheckResult},
            )
            result = GoalCheckResult.model_validate_json(text_result)
            return result.result.strip().lower() == "yes"
        except pydantic.ValidationError:
            _logger.warning(f"Could not parse response for goal achievement check: {text_result}")
            return False
        except Exception as e:
            _logger.warning(f"Goal achievement check failed: {e}")
            return False
