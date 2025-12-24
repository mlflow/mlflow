import logging
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.types.llm import ChatMessage
from mlflow.types.responses import ResponsesAgentResponse

_logger = logging.getLogger(__name__)

DEFAULT_PERSONA = "You are a helpful user having a natural conversation."


def _parse_model_string(model: str) -> tuple[str, str]:
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return "openai", model


class SimulatedUserAgent:
    INITIAL_USER_PROMPT = (
        "{persona}\n\n"
        "Your goal in this conversation is to: {goal}\n\n"
        "Start a conversation about this topic. Don't ask about your goal directly - "
        "instead start with a broader question and let the conversation develop naturally."
    )

    FOLLOWUP_USER_PROMPT = (
        "{persona}\n\n"
        "Your goal: {goal}\n\n"
        "Conversation so far:\n{conversation_history}\n\n"
        "They just said: {last_response}\n\n"
        "Respond naturally and guide the conversation toward your goal."
    )

    def __init__(
        self,
        goal: str,
        persona: str | None = None,
        model: str = "openai/gpt-4o-mini",
        **inference_params,
    ):
        self.goal = goal
        self.persona = persona or DEFAULT_PERSONA
        self.model = model
        self.inference_params = inference_params

    def generate_message(
        self,
        conversation_history: list[dict[str, Any]],
        turn: int = 0,
    ) -> str:
        if turn == 0:
            prompt = self.INITIAL_USER_PROMPT.format(persona=self.persona, goal=self.goal)
        else:
            last_response = (
                conversation_history[-1].get("content", "") if conversation_history else ""
            )
            history_str = (
                self._format_history(conversation_history[:-1])
                if len(conversation_history) > 1
                else ""
            )

            prompt = self.FOLLOWUP_USER_PROMPT.format(
                persona=self.persona,
                goal=self.goal,
                conversation_history=history_str,
                last_response=last_response,
            )

        messages = [ChatMessage(role="user", content=prompt)]
        provider, model_name = _parse_model_string(self.model)
        response, _ = _invoke_litellm_and_handle_tools(
            provider=provider,
            model_name=model_name,
            messages=messages,
            trace=None,
            num_retries=3,
            inference_params=self.inference_params,
        )
        return response

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return ""
        formatted = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


class ConversationSimulator:
    def __init__(
        self,
        test_cases: list[dict[str, Any]] | pd.DataFrame,
        max_turns: int = 10,
        user_model: str = "openai/gpt-4o-mini",
        **user_llm_params,
    ):
        self.test_cases = self._normalize_test_cases(test_cases)
        self.max_turns = max_turns
        self.user_model = user_model
        self.user_llm_params = user_llm_params

    def _normalize_test_cases(
        self, test_cases: list[dict[str, Any]] | pd.DataFrame
    ) -> list[dict[str, Any]]:
        if isinstance(test_cases, pd.DataFrame):
            return test_cases.to_dict("records")
        return test_cases

    def simulate(self, predict_fn: Callable[..., dict[str, Any]]) -> list[Trace]:
        if not self.test_cases:
            raise ValueError("test_cases cannot be empty")

        all_traces = []

        for test_case in self.test_cases:
            try:
                traces = self._run_conversation(test_case, predict_fn)
                all_traces.extend(traces)
            except Exception as e:
                _logger.error(
                    f"Failed to run conversation for test case {test_case.get('goal')}: {e}"
                )
                continue

        return all_traces

    def _run_conversation(
        self, test_case: dict[str, Any], predict_fn: Callable[..., dict[str, Any]]
    ) -> list[Trace]:
        goal = test_case.get("goal")
        if not goal:
            raise ValueError("Test case must have 'goal' field")

        persona = test_case.get("persona")
        context = test_case.get("context", {})

        session_id = f"sim-{uuid4().hex[:16]}"

        user_agent = SimulatedUserAgent(
            goal=goal,
            persona=persona,
            model=self.user_model,
            **self.user_llm_params,
        )

        conversation_history = []
        traces = []

        for turn in range(self.max_turns):
            try:
                user_message_content = user_agent.generate_message(conversation_history, turn)
                user_message = {"role": "user", "content": user_message_content}
                conversation_history.append(user_message)

                with mlflow.start_span(
                    name=f"conversation_turn_{turn}",
                    span_type=SpanType.AGENT,
                ):
                    response = predict_fn(input=conversation_history, **context)

                    if not isinstance(response, (dict, ResponsesAgentResponse)):
                        raise ValueError(
                            f"predict_fn must return ResponsesAgentResponse or dict, "
                            f"got {type(response)}"
                        )

                    if isinstance(response, dict):
                        response = ResponsesAgentResponse(**response)

                    assistant_message = self._extract_assistant_message(response)
                    conversation_history.append(assistant_message)

                mlflow.update_current_trace(
                    metadata={
                        TraceMetadataKey.TRACE_SESSION: session_id,
                        "mlflow.simulation.synthetic": "true",
                        "mlflow.simulation.goal": goal,
                        "mlflow.simulation.persona": (persona or DEFAULT_PERSONA)[:100],
                        "mlflow.simulation.turn": str(turn),
                    },
                )

                if trace_id := mlflow.get_last_active_trace_id():
                    trace = mlflow.get_trace(trace_id)
                    traces.append(trace)

                assistant_content = assistant_message.get("content", "")
                if not assistant_content or not assistant_content.strip():
                    _logger.info(f"Stopping conversation: empty response at turn {turn}")
                    break

                if self._check_goal_achieved(conversation_history, goal):
                    _logger.info(f"Stopping conversation: goal achieved at turn {turn}")
                    break

            except Exception as e:
                _logger.error(f"Error during turn {turn}: {e}")
                break

        return traces

    def _extract_assistant_message(self, response: ResponsesAgentResponse) -> dict[str, Any]:
        texts = []
        for output_item in response.output:
            output_dict = output_item if isinstance(output_item, dict) else output_item.model_dump()
            if output_dict.get("type") == "message" and output_dict.get("role") == "assistant":
                for content_item in output_dict.get("content", []):
                    content_dict = (
                        content_item
                        if isinstance(content_item, dict)
                        else content_item.model_dump()
                    )
                    if content_dict.get("type") == "output_text":
                        texts.append(content_dict.get("text", ""))

        return {"role": "assistant", "content": "".join(texts)}

    def _check_goal_achieved(self, conversation_history: list[dict[str, Any]], goal: str) -> bool:
        last_assistant_message = None
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                last_assistant_message = msg.get("content", "")
                break

        if not last_assistant_message:
            return False

        eval_prompt = f"""Goal: {goal}

Latest response: {last_assistant_message}

Has the conversation achieved the specified goal? Answer only 'yes' or 'no'."""

        messages = [ChatMessage(role="user", content=eval_prompt)]

        try:
            provider, model_name = _parse_model_string(self.user_model)
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
