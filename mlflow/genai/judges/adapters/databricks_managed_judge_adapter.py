from __future__ import annotations

import inspect
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage, ToolDefinition

T = TypeVar("T")  # Generic type for agentic loop return value

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
from mlflow.genai.judges.utils.tool_calling_utils import (
    _process_tool_calls,
    _raise_iteration_limit_exceeded,
)
from mlflow.genai.utils.message_utils import serialize_messages_to_databricks_prompts
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

# The judge's rationale is free text that may contain unescaped double quotes (e.g. when it
# quotes JSON from the trace), while the result is a fixed enum, number, or boolean. These
# patterns recover both fields from such malformed output, in either key order. The greedy
# rationale capture can swallow field or object boundaries, so recovery is skipped when the
# captured rationale looks like it spans more than the rationale value (see
# `_is_unambiguous_rationale`).
_RESULT_VALUE_PATTERN = r'"[^"\\]*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null'
_RATIONALE_THEN_RESULT_PATTERN = re.compile(
    r'^\s*\{\s*"rationale"\s*:\s*"(?P<rationale>.*)"\s*,'
    rf'\s*"result"\s*:\s*(?P<result>{_RESULT_VALUE_PATTERN})\s*\}}\s*$',
    re.DOTALL,
)
_RESULT_THEN_RATIONALE_PATTERN = re.compile(
    rf'^\s*\{{\s*"result"\s*:\s*(?P<result>{_RESULT_VALUE_PATTERN})\s*,'
    r'\s*"rationale"\s*:\s*"(?P<rationale>.*)"\s*\}\s*$',
    re.DOTALL,
)
# Matches a double quote that is not escaped (preceded by an even number of backslashes).
_UNESCAPED_QUOTE_PATTERN = re.compile(r'(?<!\\)((?:\\\\)*)"')
# Matches a `", "key":` field boundary, which signals that the greedy rationale capture spans
# extra fields of the outer object.
_FIELD_BOUNDARY_PATTERN = re.compile(r'"\s*,\s*"[^"]*"\s*:')


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


def _is_unambiguous_rationale(raw_rationale: str) -> bool:
    """
    Return False if the captured rationale may span outer object boundaries, i.e. its braces or
    brackets are unbalanced (truncated output or concatenated objects) or it contains a field
    boundary outside any nested braces or brackets (extra fields).
    """
    depth = 0
    top_level_chars = []
    for char in raw_rationale:
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth < 0:
                return False
        elif depth == 0:
            top_level_chars.append(char)
    return depth == 0 and not _FIELD_BOUNDARY_PATTERN.search("".join(top_level_chars))


def _lenient_parse_judge_output(llm_output: str) -> dict[str, Any] | None:
    """
    Recover ``result`` and ``rationale`` from judge output that is not valid JSON because the
    rationale contains unescaped double quotes. Returns None if the output doesn't match the
    expected two-field shape.
    """
    patterns = (_RATIONALE_THEN_RESULT_PATTERN, _RESULT_THEN_RATIONALE_PATTERN)
    match = next((m for p in patterns if (m := p.match(llm_output))), None)
    if match is None:
        return None

    try:
        result = json.loads(match.group("result"))
    except json.JSONDecodeError:
        return None
    raw_rationale = match.group("rationale")
    if not _is_unambiguous_rationale(raw_rationale):
        return None
    escaped_rationale = _UNESCAPED_QUOTE_PATTERN.sub(r'\1\\"', raw_rationale)
    try:
        rationale = json.loads(f'"{escaped_rationale}"', strict=False)
    except json.JSONDecodeError:
        rationale = raw_rationale
    return {"result": result, "rationale": rationale}


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
        response_data = _lenient_parse_judge_output(llm_output)
        if response_data is None:
            return Feedback(
                name=assessment_name,
                error=(
                    f"Invalid JSON response from Databricks judge: {e}\n\nLLM output: {llm_output}"
                ),
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


def _create_message_from_databricks_response(
    response_data: dict[str, Any],
) -> "ChatMessage":
    """
    Convert Databricks OpenAI-style response to a ChatMessage.

    Handles both string content and reasoning model outputs.

    Args:
        response_data: Parsed JSON response from Databricks.

    Returns:
        ChatMessage object.

    Raises:
        ValueError: If response format is invalid.
    """
    from mlflow.types.llm import ChatMessage

    choices = response_data.get("choices", [])
    if not choices:
        raise ValueError("Invalid response format: missing 'choices' field")

    message_data = choices[0].get("message", {})

    # tool_calls come as plain dicts; ChatMessage.__post_init__ auto-converts to ToolCall
    tool_calls = message_data.get("tool_calls")

    content = message_data.get("content")
    if isinstance(content, list):
        content_parts = [
            block["text"] for block in content if isinstance(block, dict) and "text" in block
        ]
        content = "\n".join(content_parts) if content_parts else None

    # ChatMessage requires content when tool_calls is absent; use empty string as fallback
    if content is None and not tool_calls:
        content = ""

    return ChatMessage(
        role=message_data.get("role", "assistant"),
        content=content,
        tool_calls=tool_calls,
    )


def _run_databricks_agentic_loop(
    messages: list["ChatMessage"],
    trace: "Trace | None",
    on_final_answer: Callable[[str | None], T],
    use_case: str | None = None,
) -> T:
    """
    Run an agentic loop with Databricks chat completions.

    This is the shared implementation for all Databricks-based agentic workflows
    (judges, structured output extraction for traces). It handles the iterative
    tool-calling loop until the LLM produces a final answer.

    Args:
        messages: Initial ChatMessage objects for the conversation.
        trace: Optional trace for tool calling. If provided, enables tool use.
        on_final_answer: Callback to process the final LLM response content.
            Receives the content string (or None if empty) and should return
            the appropriate result type or raise an exception.

    Returns:
        Result from on_final_answer callback.

    Raises:
        MlflowException: If max iterations exceeded or other errors occur.
    """
    tools = None
    if trace is not None:
        from mlflow.genai.judges.tools import list_judge_tools

        tools = [tool.get_definition() for tool in list_judge_tools()]

    max_iterations = MLFLOW_JUDGE_MAX_ITERATIONS.get()
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            _raise_iteration_limit_exceeded(max_iterations)

        try:
            user_prompt, system_prompt = serialize_messages_to_databricks_prompts(messages)

            llm_result = call_chat_completions(
                user_prompt,
                system_prompt or "",
                tools=tools,
                model=_DATABRICKS_AGENTIC_JUDGE_MODEL,
                use_case=use_case,
            )

            # Surface API errors from the response before checking output_json,
            # so users see the actual error (e.g. "Model context limit exceeded")
            # instead of a misleading "Empty response" message.
            error_code = getattr(llm_result, "error_code", None)
            error_message = getattr(llm_result, "error_message", None)
            if error_code or error_message:
                raise MlflowException(
                    f"Databricks judge API error (code={error_code}): {error_message}"
                )

            output_json = llm_result.output_json
            if not output_json:
                raise MlflowException("Empty response from Databricks judge")

            parsed_json = json.loads(output_json) if isinstance(output_json, str) else output_json
            message = _create_message_from_databricks_response(parsed_json)

            if not message.tool_calls:
                return on_final_answer(message.content)

            messages.append(message)
            tool_response_messages = _process_tool_calls(
                tool_calls=message.tool_calls,
                trace=trace,
            )
            messages.extend(tool_response_messages)
        except Exception:
            _logger.debug("Failed during Databricks agentic loop iteration", exc_info=True)
            raise


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
    from mlflow.types.llm import ChatMessage

    try:
        if isinstance(prompt, str):
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = [ChatMessage(role=msg.role, content=msg.content) for msg in prompt]

        # Define callback to parse final answer into Feedback
        def parse_judge_response(content: str | None) -> Feedback:
            return _parse_databricks_judge_response(content, assessment_name, trace)

        return _run_databricks_agentic_loop(messages, trace, parse_judge_response, use_case)

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

    def _invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        if input_params.base_url is not None or input_params.extra_headers is not None:
            raise MlflowException(
                "base_url and extra_headers are not supported for Databricks Managed Judge. "
                "The endpoint is determined by Databricks workspace configuration.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        feedback = _invoke_databricks_default_judge(
            prompt=input_params.prompt,
            assessment_name=input_params.assessment_name,
            trace=input_params.trace,
            use_case=input_params.use_case,
        )
        return AdapterInvocationOutput(feedback=feedback)
