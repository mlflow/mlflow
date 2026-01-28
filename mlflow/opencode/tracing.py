"""MLflow tracing integration for Opencode conversations."""

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from mlflow.opencode.config import (
    MLFLOW_TRACING_ENABLED,
    get_env_var,
)
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey, TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager

NANOSECONDS_PER_MS = 1e6
MAX_PREVIEW_LENGTH = 1000

# Message and part types from Opencode SDK
MESSAGE_ROLE_USER = "user"
MESSAGE_ROLE_ASSISTANT = "assistant"
PART_TYPE_TEXT = "text"
PART_TYPE_TOOL = "tool"

# Custom logging level for Opencode tracing
OPENCODE_TRACING_LEVEL = logging.WARNING - 5


def setup_logging() -> logging.Logger:
    log_dir = Path(os.getcwd()) / ".opencode" / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    log_file = log_dir / "opencode_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logging.addLevelName(OPENCODE_TRACING_LEVEL, "OPENCODE_TRACING")
    logger.setLevel(OPENCODE_TRACING_LEVEL)
    logger.propagate = False

    return logger


_MODULE_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _MODULE_LOGGER

    if _MODULE_LOGGER is None:
        _MODULE_LOGGER = setup_logging()
    return _MODULE_LOGGER


def setup_mlflow() -> None:
    if not is_tracing_enabled():
        return

    if tracking_uri := get_env_var(MLFLOW_TRACKING_URI.name):
        mlflow.set_tracking_uri(tracking_uri)

    experiment_id = get_env_var(MLFLOW_EXPERIMENT_ID.name)
    experiment_name = get_env_var(MLFLOW_EXPERIMENT_NAME.name)

    try:
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
        elif experiment_name:
            mlflow.set_experiment(experiment_name)
    except Exception as e:
        get_logger().warning("Failed to set experiment: %s", e)


def is_tracing_enabled() -> bool:
    return get_env_var(MLFLOW_TRACING_ENABLED).lower() in ("true", "1", "yes")


def timestamp_to_ns(timestamp: int | float | None) -> int | None:
    return int(timestamp * NANOSECONDS_PER_MS) if timestamp is not None else None


def extract_user_prompt(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        info = msg.get("info", {})
        if info.get("role") == MESSAGE_ROLE_USER:
            parts = msg.get("parts", [])
            for part in parts:
                if part.get("type") == PART_TYPE_TEXT:
                    return part.get("text", "")
    return ""


def extract_assistant_response(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        info = msg.get("info", {})
        if info.get("role") == MESSAGE_ROLE_ASSISTANT:
            parts = msg.get("parts", [])
            for part in reversed(parts):
                if part.get("type") == PART_TYPE_TEXT:
                    text = part.get("text", "")
                    if text.strip():
                        return text
    return ""


def find_last_user_message_index(messages: list[dict[str, Any]]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        info = messages[i].get("info", {})
        if info.get("role") == MESSAGE_ROLE_USER:
            return i
    return None


def _set_token_usage_attribute(span, tokens: dict[str, Any]) -> None:
    if not tokens:
        return

    input_tokens = tokens.get("input", 0)
    output_tokens = tokens.get("output", 0)
    reasoning_tokens = tokens.get("reasoning", 0)

    # Include cache tokens in the total
    cache = tokens.get("cache", {})
    cache_read = cache.get("read", 0)
    cache_write = cache.get("write", 0)

    usage_dict = {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens + reasoning_tokens,
    }

    # Add cache info if available
    if cache_read or cache_write:
        usage_dict["cache_read_tokens"] = cache_read
        usage_dict["cache_write_tokens"] = cache_write

    span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)


def _reconstruct_conversation_messages(
    messages: list[dict[str, Any]], end_idx: int
) -> list[dict[str, Any]]:
    result = []

    for i in range(end_idx):
        msg = messages[i]
        info = msg.get("info", {})
        role = info.get("role")
        parts = msg.get("parts", [])

        if role == MESSAGE_ROLE_USER:
            # Extract text parts from user message
            if text_parts := [p.get("text", "") for p in parts if p.get("type") == PART_TYPE_TEXT]:
                result.append({"role": "user", "content": "\n".join(text_parts)})

        elif role == MESSAGE_ROLE_ASSISTANT:
            # Extract text parts from assistant message
            if text_parts := [p.get("text", "") for p in parts if p.get("type") == PART_TYPE_TEXT]:
                result.append({"role": "assistant", "content": "\n".join(text_parts)})

            # Also add tool results as tool messages
            for part in parts:
                if part.get("type") == PART_TYPE_TOOL:
                    state = part.get("state", {})
                    if state.get("status") == "completed":
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.get("callID"),
                                "content": state.get("output", ""),
                            }
                        )

    return result


def _create_llm_and_tool_spans(parent_span, messages: list[dict[str, Any]], start_idx: int) -> None:
    llm_call_num = 0

    for i in range(start_idx, len(messages)):
        msg = messages[i]
        info = msg.get("info", {})

        if info.get("role") != MESSAGE_ROLE_ASSISTANT:
            continue

        parts = msg.get("parts", [])
        model_id = info.get("modelID", "unknown")
        provider_id = info.get("providerID", "unknown")
        tokens = info.get("tokens", {})

        # Get timing from message
        time_info = info.get("time", {})
        created_ns = timestamp_to_ns(time_info.get("created"))
        completed_ns = timestamp_to_ns(time_info.get("completed"))

        # Check for text content
        text_parts = [p for p in parts if p.get("type") == PART_TYPE_TEXT]
        tool_parts = [p for p in parts if p.get("type") == PART_TYPE_TOOL]

        # Create LLM span for text responses
        if text_parts:
            llm_call_num += 1
            conversation_messages = _reconstruct_conversation_messages(messages, i)
            text_content = "\n".join(p.get("text", "") for p in text_parts)

            llm_span = mlflow.start_span_no_context(
                name=f"llm_call_{llm_call_num}",
                parent_span=parent_span,
                span_type=SpanType.LLM,
                start_time_ns=created_ns,
                inputs={
                    "model": f"{provider_id}/{model_id}",
                    "messages": conversation_messages,
                },
                attributes={
                    "model": model_id,
                    "provider": provider_id,
                },
            )

            _set_token_usage_attribute(llm_span, tokens)
            llm_span.set_outputs({"response": text_content})
            llm_span.end(end_time_ns=completed_ns)

        # Create tool spans
        for tool_part in tool_parts:
            state = tool_part.get("state", {})
            tool_name = tool_part.get("tool", "unknown")
            call_id = tool_part.get("callID", "")

            # Get tool timing
            tool_time = state.get("time", {})
            tool_start_ns = timestamp_to_ns(tool_time.get("start"))
            tool_end_ns = timestamp_to_ns(tool_time.get("end"))

            tool_span = mlflow.start_span_no_context(
                name=f"tool_{tool_name}",
                parent_span=parent_span,
                span_type=SpanType.TOOL,
                start_time_ns=tool_start_ns,
                inputs=state.get("input", {}),
                attributes={
                    "tool_name": tool_name,
                    "tool_id": call_id,
                    "status": state.get("status", "unknown"),
                },
            )

            # Set output based on status
            if state.get("status") == "completed":
                tool_span.set_outputs(
                    {
                        "result": state.get("output", ""),
                        "title": state.get("title", ""),
                    }
                )
            elif state.get("status") == "error":
                tool_span.set_outputs(
                    {
                        "error": state.get("error", "Unknown error"),
                    }
                )

            tool_span.end(end_time_ns=tool_end_ns)


def process_session(
    session_id: str,
    session_info: dict[str, Any],
    messages: list[dict[str, Any]],
) -> mlflow.entities.Trace | None:
    try:
        if not messages:
            get_logger().warning("Empty messages list, skipping")
            return None

        last_user_idx = find_last_user_message_index(messages)
        if last_user_idx is None:
            get_logger().warning("No user message found in session")
            return None

        user_prompt = extract_user_prompt(messages)
        if not user_prompt:
            get_logger().warning("No user prompt text found")
            return None

        get_logger().log(
            OPENCODE_TRACING_LEVEL, "Creating MLflow trace for session: %s", session_id
        )

        # Get timing from the FIRST message in this batch (for correct ordering)
        first_msg = messages[0]
        first_msg_time = first_msg.get("info", {}).get("time", {})
        created_ns = timestamp_to_ns(first_msg_time.get("created"))

        # Get end time from the LAST message in this batch
        last_msg = messages[-1]
        last_msg_time = last_msg.get("info", {}).get("time", {})
        updated_ns = timestamp_to_ns(last_msg_time.get("completed") or last_msg_time.get("created"))

        # Create parent span for the conversation
        parent_span = mlflow.start_span_no_context(
            name="opencode_conversation",
            inputs={"prompt": user_prompt},
            start_time_ns=created_ns,
            span_type=SpanType.AGENT,
        )

        # Create child spans for LLM calls and tools
        _create_llm_and_tool_spans(parent_span, messages, last_user_idx + 1)

        # Get final response for preview
        final_response = extract_assistant_response(messages)

        # Set trace metadata
        try:
            with InMemoryTraceManager.get_instance().get_trace(
                parent_span.trace_id
            ) as in_memory_trace:
                in_memory_trace.info.request_preview = user_prompt[:MAX_PREVIEW_LENGTH]
                if final_response:
                    in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]
                in_memory_trace.info.trace_metadata = {
                    **in_memory_trace.info.trace_metadata,
                    TraceMetadataKey.TRACE_SESSION: session_id,
                    TraceMetadataKey.TRACE_USER: os.environ.get("USER", ""),
                    "mlflow.trace.working_directory": session_info.get("directory", os.getcwd()),
                    "mlflow.trace.session_title": session_info.get("title", ""),
                }
        except Exception as e:
            get_logger().warning("Failed to update trace metadata: %s", e)

        # End parent span
        parent_span.set_outputs(
            {"response": final_response or "Conversation completed", "status": "completed"}
        )
        parent_span.end(end_time_ns=updated_ns)

        get_logger().log(OPENCODE_TRACING_LEVEL, "Created MLflow trace: %s", parent_span.trace_id)

        return mlflow.get_trace(parent_span.trace_id)

    except Exception as e:
        get_logger().error("Error processing session: %s", e, exc_info=True)
        return None
