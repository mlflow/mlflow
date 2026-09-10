"""MLflow tracing integration for Aider CLI interactions."""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey
from mlflow.tracing.provider import _get_trace_exporter
from mlflow.tracing.trace_manager import InMemoryTraceManager

NANOSECONDS_PER_S = 1e9
MAX_PREVIEW_LENGTH = 1000
DEFAULT_AIDER_HISTORY_FILE = ".aider.llm.history"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


def setup_logging() -> logging.Logger:
    log_dir = Path(os.getcwd()) / ".aider" / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    log_file = log_dir / "aider_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_MODULE_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _MODULE_LOGGER
    if _MODULE_LOGGER is None:
        _MODULE_LOGGER = setup_logging()
    return _MODULE_LOGGER


def read_aider_history(history_file: str) -> list[dict[str, Any]]:
    path = Path(history_file)
    if not path.exists():
        raise FileNotFoundError(f"Aider history file not found: {history_file}")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    messages = []
    blocks = re.split(r"\n(?=#{1,2} )", content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("# aider"):
            continue
        if block.startswith("## "):
            lines = block.split("\n", 1)
            header = lines[0].strip("# ").strip().lower()
            content_text = lines[1].strip() if len(lines) > 1 else ""
            if not content_text:
                continue
            if "user" in header or "human" in header:
                messages.append({"role": ROLE_USER, "content": content_text})
            elif "assistant" in header or "llm" in header or "response" in header:
                messages.append({"role": ROLE_ASSISTANT, "content": content_text})
    return messages


def find_last_user_message(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == ROLE_USER:
            return msg.get("content", "")
    return None


def find_final_assistant_response(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == ROLE_ASSISTANT:
            return msg.get("content", "")
    return None


def _create_conversation_spans(parent_span, messages: list[dict[str, Any]]) -> None:
    pending_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == ROLE_USER:
            pending_messages.append({"role": "user", "content": content})
        elif role == ROLE_ASSISTANT and content.strip():
            llm_span = mlflow.start_span_no_context(
                name="llm",
                parent_span=parent_span,
                span_type=SpanType.LLM,
                inputs={"messages": pending_messages},
                attributes={SpanAttributeKey.MESSAGE_FORMAT: "aider"},
            )
            llm_span.set_outputs({"role": "assistant", "content": content})
            llm_span.end()
            pending_messages = []


def _flush_trace_async_logging() -> None:
    try:
        if hasattr(_get_trace_exporter(), "_async_queue"):
            mlflow.flush_trace_async_logging()
    except Exception as e:
        get_logger().debug("Failed to flush trace async logging: %s", e)


def process_aider_history(
    history_file: str | None = None,
    session_id: str | None = None,
) -> mlflow.entities.Trace | None:
    if history_file is None:
        history_file = str(Path(os.getcwd()) / DEFAULT_AIDER_HISTORY_FILE)
    try:
        messages = read_aider_history(history_file)
        if not messages:
            get_logger().warning("No messages found in Aider history file")
            return None
        user_prompt = find_last_user_message(messages)
        if not user_prompt:
            get_logger().warning("No user message found in Aider history")
            return None
        if not session_id:
            session_id = f"aider-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        get_logger().info("Creating MLflow trace for Aider session: %s", session_id)
        parent_span = mlflow.start_span_no_context(
            name="aider_conversation",
            inputs={"prompt": user_prompt[:MAX_PREVIEW_LENGTH]},
            span_type=SpanType.AGENT,
        )
        _create_conversation_spans(parent_span, messages)
        final_response = find_final_assistant_response(messages)
        try:
            with InMemoryTraceManager.get_instance().get_trace(
                parent_span.trace_id
            ) as in_memory_trace:
                if user_prompt:
                    in_memory_trace.info.request_preview = user_prompt[:MAX_PREVIEW_LENGTH]
                if final_response:
                    in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]
                in_memory_trace.info.trace_metadata = {
                    **in_memory_trace.info.trace_metadata,
                    TraceMetadataKey.TRACE_USER: os.environ.get("USER", ""),
                    TraceMetadataKey.TRACE_SESSION: session_id,
                    "mlflow.trace.working_directory": os.getcwd(),
                }
        except Exception as e:
            get_logger().warning("Failed to update trace metadata: %s", e)
        outputs = {"status": "completed"}
        if final_response:
            outputs["response"] = final_response
        parent_span.set_outputs(outputs)
        parent_span.end()
        _flush_trace_async_logging()
        get_logger().info("Created MLflow trace: %s", parent_span.trace_id)
        return mlflow.get_trace(parent_span.trace_id)
    except Exception as e:
        get_logger().error("Error processing Aider history: %s", e, exc_info=True)
        return None
