from __future__ import annotations

import inspect
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage, ToolDefinition

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.base_adapter import (
    AdapterInvocationInput,
    AdapterInvocationOutput,
    BaseJudgeAdapter,
)
from mlflow.genai.judges.constants import (
    _DATABRICKS_AGENTIC_JUDGE_MODEL,
    _DATABRICKS_DEFAULT_JUDGE_MODEL,
)
from mlflow.genai.judges.utils.tool_calling_utils import _process_tool_calls
from mlflow.protos.databricks_pb2 import BAD_REQUEST, REQUEST_LIMIT_EXCEEDED
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
    user_prompt: str,
    system_prompt: str,
    session_name: str | None = None,
    tools: list[ToolDefinition] | None = None,
    model: str | None = None,
    use_case: str | None = None,
) -> Any:
    """
    Invokes the Databricks chat completions API using the databricks.agents.evals library.

    Args:
        user_prompt: The user prompt.
        system_prompt: The system prompt.
        session_name: The session name for tracking. Defaults to "mlflow-v{VERSION}".
        tools: Optional list of ToolDefinition objects for tool calling.
        model: Optional model to use.
        use_case: The use case for the chat completion. Only used if supported
            by the installed databricks-agents version.

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
    def _call_chat_completions(
        user_prompt: str,
        system_prompt: str,
        tools: list[ToolDefinition] | None,
        model: str | None,
        use_case: str | None,
    ):
        managed_rag_client = context.get_context().build_managed_rag_client()

        # Build kwargs dict starting with required parameters
        kwargs = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
        }

        # Add optional parameters
        if model is not None:
            kwargs["model"] = model

        if tools is not None:
            kwargs["tools"] = tools

        # Check if use_case parameter is supported by checking the method signature
        if use_case is not None:
            get_chat_completions_sig = inspect.signature(
                managed_rag_client.get_chat_completions_result
            )
            if "use_case" in get_chat_completions_sig.parameters:
                kwargs["use_case"] = use_case

        try:
            return managed_rag_client.get_chat_completions_result(**kwargs)
        except Exception:
            _logger.debug("Failed to call chat completions", exc_info=True)
            raise

    return _call_chat_completions(user_prompt, system_prompt, tools, model, use_case)


def _parse_databricks_judge_response(
    llm_output: str | None,
    assessment_name: str,
    trace: "Trace | None" = None,
) -> Feedback:
    """
    Parse the response from Databricks judge into a Feedback object.

    Args:
        llm_output: Raw output from the LLM, or None if no response.
        assessment_name: Name of the assessment.
        trace: Optional trace object to associate with the feedback.

    Returns:
        Feedback object with parsed results or error.
    """
    source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE, source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL
    )
    trace_id = trace.info.trace_id if trace else None

    if not llm_output:
        return Feedback(
            name=assessment_name,
            error="Empty response from Databricks judge",
            source=source,
            trace_id=trace_id,
        )

    try:
        response_data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        _logger.debug(f"Invalid JSON response from Databricks judge: {e}", exc_info=True)
        return Feedback(
            name=assessment_name,
            error=f"Invalid JSON response from Databricks judge: {e}\n\nLLM output: {llm_output}",
            source=source,
            trace_id=trace_id,
        )

    if "result" not in response_data:
        return Feedback(
            name=assessment_name,
            error=f"Response missing 'result' field: {response_data}",
            source=source,
            trace_id=trace_id,
        )

    return Feedback(
        name=assessment_name,
        value=response_data["result"],
        rationale=response_data.get("rationale", ""),
        source=source,
        trace_id=trace_id,
    )


def _create_litellm_message_from_databricks_response(
    response_data: dict[str, Any],
) -> Any:
    """
    Convert Databricks OpenAI-style response to litellm Message.

    Handles both string content and reasoning model outputs.

    Args:
        response_data: Parsed JSON response from Databricks.

    Returns:
        litellm.Message object.

    Raises:
        ValueError: If response format is invalid.
    """
    import litellm

    choices = response_data.get("choices", [])
    if not choices:
        raise ValueError("Invalid response format: missing 'choices' field")

    message_data = choices[0].get("message", {})

    # Create litellm Message with tool calls if present
    tool_calls_data = message_data.get("tool_calls")
    tool_calls = None
    if tool_calls_data:
        tool_calls = [
            litellm.ChatCompletionMessageToolCall(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=litellm.Function(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls_data
        ]

    content = message_data.get("content")
    if isinstance(content, list):
        content_parts = [
            block["text"] for block in content if isinstance(block, dict) and "text" in block
        ]
        content = "\n".join(content_parts) if content_parts else None

    return litellm.Message(
        role=message_data.get("role", "assistant"),
        content=content,
        tool_calls=tool_calls,
    )


def _serialize_messages_to_databricks_prompts(
    messages: list[Any],
) -> tuple[str, str | None]:
    """
    Serialize litellm Messages to user_prompt and system_prompt for Databricks.

    This is needed because call_chat_completions only accepts string prompts.

    Args:
        messages: List of litellm Message objects.

    Returns:
        Tuple of (user_prompt, system_prompt).
    """
    system_prompt = None
    user_parts = []

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_parts.append(msg.content)
        elif msg.role == "assistant":
            if msg.tool_calls:
                user_parts.append("Assistant: [Called tools]")
            elif msg.content:
                user_parts.append(f"Assistant: {msg.content}")
        elif msg.role == "tool":
            user_parts.append(f"Tool {msg.name}: {msg.content}")

    user_prompt = "\n\n".join(user_parts)
    return user_prompt, system_prompt


def _invoke_databricks_default_judge(
    prompt: str | list["ChatMessage"],
    assessment_name: str,
    trace: "Trace | None" = None,
    use_case: str | None = None,
) -> Feedback:
    """
    Invoke the Databricks default judge with agentic tool calling support.

    When a trace is provided, enables an agentic loop where the judge can iteratively
    call tools to analyze the trace data before producing a final assessment.

    Args:
        prompt: The formatted prompt with template variables filled in.
        assessment_name: The name of the assessment.
        trace: Optional trace object for tool-based analysis.
        use_case: The use case for the chat completion. Only used if supported by the
            installed databricks-agents version.

    Returns:
        Feedback object from the Databricks judge.

    Raises:
        MlflowException: If databricks-agents is not installed or max iterations exceeded.
    """
    import litellm

    try:
        # Convert initial prompt to litellm Messages (same pattern as litellm adapter)
        if isinstance(prompt, str):
            messages = [litellm.Message(role="user", content=prompt)]
        else:
            messages = [litellm.Message(role=msg.role, content=msg.content) for msg in prompt]

        # Enable tool calling if trace is provided
        tools = None
        model = None
        if trace is not None:
            from mlflow.genai.judges.tools import list_judge_tools

            tools = [tool.get_definition() for tool in list_judge_tools()]
            model = _DATABRICKS_AGENTIC_JUDGE_MODEL

        # Agentic loop: iteratively call LLM and execute tools until final answer
        max_iterations = MLFLOW_JUDGE_MAX_ITERATIONS.get()
        iteration_count = 0

        while True:
            iteration_count += 1
            if iteration_count > max_iterations:
                raise MlflowException(
                    f"Completion iteration limit of {max_iterations} exceeded. "
                    f"This usually indicates the model is not powerful enough to effectively "
                    f"analyze the trace. Consider using a more intelligent/powerful model. "
                    f"In rare cases, for very complex traces where a large number of completion "
                    f"iterations might be required, you can increase the number of iterations by "
                    f"modifying the {MLFLOW_JUDGE_MAX_ITERATIONS.name} environment variable.",
                    error_code=REQUEST_LIMIT_EXCEEDED,
                )

            try:
                # Serialize messages to Databricks format
                user_prompt, system_prompt = _serialize_messages_to_databricks_prompts(messages)

                llm_result = call_chat_completions(
                    user_prompt, system_prompt, tools=tools, model=model, use_case=use_case
                )

                # Parse response
                output_json = llm_result.output_json
                if not output_json:
                    return Feedback(
                        name=assessment_name,
                        error="Empty response from Databricks judge",
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.LLM_JUDGE,
                            source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL,
                        ),
                        trace_id=trace.info.trace_id if trace else None,
                    )

                parsed_json = (
                    json.loads(output_json) if isinstance(output_json, str) else output_json
                )

                # Convert response to litellm Message
                try:
                    message = _create_litellm_message_from_databricks_response(parsed_json)
                except ValueError as e:
                    return Feedback(
                        name=assessment_name,
                        error=f"Invalid response format from Databricks judge: {e}",
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.LLM_JUDGE,
                            source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL,
                        ),
                        trace_id=trace.info.trace_id if trace else None,
                    )

                # No tool calls means final answer - parse and return
                if not message.tool_calls:
                    return _parse_databricks_judge_response(message.content, assessment_name, trace)

                # Append assistant message and process tool calls (same pattern as litellm)
                messages.append(message)
                tool_response_messages = _process_tool_calls(
                    tool_calls=message.tool_calls,
                    trace=trace,
                )
                messages.extend(tool_response_messages)

            except Exception as e:
                _logger.debug(
                    f"Failed in agentic loop iteration {iteration_count}: {e}", exc_info=True
                )
                raise

    except Exception as e:
        _logger.debug(f"Failed to invoke Databricks judge: {e}", exc_info=True)
        return Feedback(
            name=assessment_name,
            error=f"Failed to invoke Databricks judge: {e}",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL,
            ),
            trace_id=trace.info.trace_id if trace else None,
        )


class DatabricksManagedJudgeAdapter(BaseJudgeAdapter):
    """Adapter for Databricks managed judge using databricks.agents.evals library."""

    @classmethod
    def is_applicable(
        cls,
        model_uri: str,
        prompt: str | list["ChatMessage"],
    ) -> bool:
        return model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL

    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        feedback = _invoke_databricks_default_judge(
            prompt=input_params.prompt,
            assessment_name=input_params.assessment_name,
            trace=input_params.trace,
            use_case=input_params.use_case,
        )
        return AdapterInvocationOutput(feedback=feedback)
