import logging
import uuid
from typing import Any, Callable

import pandas as pd

import mlflow
from mlflow.entities.trace import Trace
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.simulators.prompts import (
    CHECK_GOAL_PROMPT,
    DEFAULT_PERSONA,
    FOLLOWUP_USER_PROMPT,
    INITIAL_USER_PROMPT,
)
from mlflow.metrics.genai.model_utils import _parse_model_uri
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.utils.truncation import (
    _get_text_content_from_message,
    _try_extract_messages,
)
from mlflow.types.llm import ChatMessage
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

_MAX_METADATA_LENGTH = 250


def _extract_assistant_content(response: dict[str, Any]) -> str | None:
    if messages := _try_extract_messages(response):
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return _get_text_content_from_message(msg)
        return _get_text_content_from_message(messages[-1])
    return None


def _get_last_response(conversation_history: list[dict[str, Any]]) -> str | None:
    if not conversation_history:
        return None

    last_msg = conversation_history[-1]
    content = last_msg.get("content", "")
    if isinstance(content, str) and content:
        return content

    if result := _extract_assistant_content(last_msg):
        return result

    return _format_history(conversation_history)


def _format_history(history: list[dict[str, Any]]) -> str | None:
    if not history:
        return None
    formatted = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


@experimental(version="3.9.0")
class SimulatedUserAgent:
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
        if turn == 0:
            prompt = INITIAL_USER_PROMPT.format(persona=self.persona, goal=self.goal)
        else:
            last_response = _get_last_response(conversation_history) or ""
            history_str = _format_history(conversation_history[:-1]) or ""
            prompt = FOLLOWUP_USER_PROMPT.format(
                persona=self.persona,
                goal=self.goal,
                conversation_history=history_str,
                last_response=last_response,
            )

        messages = [ChatMessage(role="user", content=prompt)]
        provider, model_name = _parse_model_uri(self.model)
        response, _ = _invoke_litellm_and_handle_tools(
            provider=provider,
            model_name=model_name,
            messages=messages,
            trace=None,
            num_retries=3,
            inference_params=self.inference_params,
        )
        return response


@experimental(version="3.9.0")
class ConversationSimulator:
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

        for i, test_case in enumerate(self.test_cases):
            if not test_case.get("goal"):
                raise ValueError(f"Test case at index {i} must have 'goal' field")

    def _simulate(self, predict_fn: Callable[..., dict[str, Any]]) -> list[list[Trace]]:
        all_traces = []

        for test_case in self.test_cases:
            try:
                traces = self._run_conversation(test_case, predict_fn)
                all_traces.append(traces)
            except Exception as e:
                _logger.error(
                    f"Failed to run conversation for test case {test_case.get('goal')}: {e}"
                )

        return all_traces

    def _run_conversation(
        self, test_case: dict[str, Any], predict_fn: Callable[..., dict[str, Any]]
    ) -> list[Trace]:
        goal = test_case["goal"]
        persona = test_case.get("persona")
        context = test_case.get("context", {})

        session_id = f"sim-{uuid.uuid4().hex[:16]}"

        user_agent = SimulatedUserAgent(
            goal=goal,
            persona=persona,
            model=self.user_model,
            **self.user_llm_params,
        )

        conversation_history: list[dict[str, Any]] = []
        traces: list[Trace] = []

        wrapped_predict = self._wrap_predict_fn(
            predict_fn=predict_fn,
            session_id=session_id,
            goal=goal,
            persona=persona,
        )

        for turn in range(self.max_turns):
            try:
                user_message_content = user_agent.generate_message(conversation_history, turn)
                user_message = {"role": "user", "content": user_message_content}
                conversation_history.append(user_message)

                response, trace = wrapped_predict(
                    input_messages=conversation_history,
                    turn=turn,
                    **context,
                )

                if trace:
                    traces.append(trace)

                assistant_content = _extract_assistant_content(response) or ""
                assistant_message = {"role": "assistant", "content": assistant_content}
                conversation_history.append(assistant_message)

                if not assistant_content.strip():
                    _logger.info(f"Stopping conversation: empty response at turn {turn}")
                    break

                if self._check_goal_achieved(assistant_content, goal):
                    _logger.info(f"Stopping conversation: goal achieved at turn {turn}")
                    break

            except Exception as e:
                _logger.error(f"Error during turn {turn}: {e}")
                break

        return traces

    def _wrap_predict_fn(
        self,
        predict_fn: Callable[..., dict[str, Any]],
        session_id: str,
        goal: str,
        persona: str | None,
    ) -> Callable[..., tuple[dict[str, Any], Trace | None]]:
        def wrapped(
            input_messages: list[dict[str, Any]],
            turn: int,
            **context,
        ) -> tuple[dict[str, Any], Trace | None]:
            @mlflow.trace(name=f"simulation_turn_{turn}")
            def traced_predict():
                mlflow.update_current_trace(
                    metadata={
                        TraceMetadataKey.TRACE_SESSION: session_id,
                        "mlflow.simulation.synthetic": "true",
                        "mlflow.simulation.goal": goal[:_MAX_METADATA_LENGTH],
                        "mlflow.simulation.persona": (persona or DEFAULT_PERSONA)[
                            :_MAX_METADATA_LENGTH
                        ],
                        "mlflow.simulation.turn": str(turn),
                    },
                )
                return predict_fn(input=input_messages, **context)

            response = traced_predict()
            trace_id = mlflow.get_last_active_trace_id()
            trace = mlflow.get_trace(trace_id) if trace_id else None

            return response, trace

        return wrapped

    def _check_goal_achieved(self, last_response: str, goal: str) -> bool:
        if not last_response:
            return False

        eval_prompt = CHECK_GOAL_PROMPT.format(goal=goal, last_response=last_response)

        messages = [ChatMessage(role="user", content=eval_prompt)]

        try:
            provider, model_name = _parse_model_uri(self.user_model)
            response, _ = _invoke_litellm_and_handle_tools(
                provider=provider,
                model_name=model_name,
                messages=messages,
                trace=None,
                num_retries=3,
                inference_params={"temperature": 0.0, "max_tokens": 10},
            )

            return response.strip().lower().startswith("yes")
        except Exception as e:
            _logger.warning(f"Goal achievement check failed: {e}")
            return False
