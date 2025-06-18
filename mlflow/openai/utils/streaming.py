"""
Utilities for handling OpenAI streaming responses in MLflow tracing.
"""

import logging
from typing import Any

_logger = logging.getLogger(__name__)


def reconstruct_chat_completion_from_chunks(chunks: list[Any]) -> Any:
    """
    Reconstruct a ChatCompletion object from a list of ChatCompletionChunk objects.

    This function combines streaming chunks back into a complete ChatCompletion object
    that matches the structure of non-streaming responses, enabling consistent
    tracing and logging behavior.

    Args:
        chunks: List of ChatCompletionChunk objects from streaming response

    Returns:
        A reconstructed ChatCompletion object, or fallback representation if reconstruction fails
    """
    if not chunks:
        return chunks

    try:
        from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

        # Check if we're dealing with ChatCompletionChunk objects
        first_chunk = chunks[0]
        if not isinstance(first_chunk, ChatCompletionChunk):
            return _fallback_to_string_concatenation(chunks)

        # Extract base metadata from first chunk
        base_metadata = _extract_base_metadata(first_chunk)

        # Accumulate message content and metadata from all chunks
        message_data = _accumulate_message_data(chunks)

        # Build the reconstructed ChatCompletion structure
        reconstructed = _build_chat_completion_structure(base_metadata, message_data)

        # Convert to proper ChatCompletion object
        return _create_chat_completion_object(reconstructed)

    except ImportError:
        _logger.debug("OpenAI types not available, falling back to string concatenation")
        return _fallback_to_string_concatenation(chunks)
    except Exception as e:
        _logger.debug(f"Failed to reconstruct ChatCompletion from chunks: {e}")
        return _fallback_to_string_concatenation(chunks)


def _extract_base_metadata(first_chunk: Any) -> dict[str, Any]:
    """Extract base metadata from the first chunk."""
    return {
        "id": first_chunk.id,
        "object": "chat.completion",  # Convert from "chat.completion.chunk"
        "created": first_chunk.created,
        "model": first_chunk.model,
        "system_fingerprint": getattr(first_chunk, "system_fingerprint", None),
    }


def _accumulate_message_data(chunks: list[Any]) -> dict[str, Any]:
    """Accumulate message content and metadata from all chunks."""
    accumulated_content = ""
    accumulated_tool_calls = []
    accumulated_function_call = None
    finish_reason = None
    usage = None

    for chunk in chunks:
        if not (hasattr(chunk, "choices") and chunk.choices):
            continue

        choice = chunk.choices[0]

        # Process delta content
        if hasattr(choice, "delta") and choice.delta:
            accumulated_content += _extract_content_from_delta(choice.delta)
            accumulated_tool_calls.extend(_extract_tool_calls_from_delta(choice.delta))

            # Handle legacy function calls
            if choice.delta.function_call:
                accumulated_function_call = choice.delta.function_call

        # Extract finish reason
        if hasattr(choice, "finish_reason") and choice.finish_reason:
            finish_reason = choice.finish_reason

        # Extract usage from chunks that have it
        if hasattr(chunk, "usage") and chunk.usage:
            usage = chunk.usage

    return {
        "content": accumulated_content if accumulated_content else None,
        "tool_calls": accumulated_tool_calls if accumulated_tool_calls else None,
        "function_call": accumulated_function_call,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def _extract_content_from_delta(delta: Any) -> str:
    """Extract text content from a delta object."""
    return delta.content or ""


def _extract_tool_calls_from_delta(delta: Any) -> list[Any]:
    """Extract tool calls from a delta object."""
    return delta.tool_calls or []


def _build_chat_completion_structure(
    base_metadata: dict[str, Any], message_data: dict[str, Any]
) -> dict[str, Any]:
    """Build the final ChatCompletion structure."""
    # Build message object
    message = {
        "role": "assistant",
        "content": message_data["content"],
        "tool_calls": message_data["tool_calls"],
        "function_call": message_data["function_call"],
    }

    # Remove None values from message
    message = {k: v for k, v in message.items() if v is not None}

    # Build usage object
    usage = None
    if message_data["usage"]:
        usage_obj = message_data["usage"]
        usage = {
            "prompt_tokens": usage_obj.prompt_tokens,
            "completion_tokens": usage_obj.completion_tokens,
            "total_tokens": usage_obj.total_tokens,
        }

    # Build complete structure
    reconstructed = {
        **base_metadata,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": message_data["finish_reason"],
            }
        ],
        "usage": usage,
    }

    # Remove None values from top level
    return {k: v for k, v in reconstructed.items() if v is not None}


def _create_chat_completion_object(reconstructed: dict[str, Any]) -> Any:
    """Convert reconstructed dict to ChatCompletion object."""
    try:
        from openai.types.chat import ChatCompletion

        return ChatCompletion.model_validate(reconstructed)
    except Exception as e:
        _logger.debug(f"Failed to create ChatCompletion object, returning dict: {e}")
        return reconstructed


def _fallback_to_string_concatenation(chunks: list[Any]) -> str:
    """Fallback method to concatenate chunks as strings."""
    return "".join([str(chunk) for chunk in chunks])


def is_streaming_chat_completion(chunks: list[Any]) -> bool:
    """
    Check if the given chunks represent a streaming ChatCompletion response.

    Args:
        chunks: List of potential ChatCompletionChunk objects

    Returns:
        True if chunks represent streaming ChatCompletion, False otherwise
    """
    if not chunks:
        return False

    try:
        from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

        return isinstance(chunks[0], ChatCompletionChunk)
    except ImportError:
        return False
