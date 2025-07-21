from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.telemetry.model_diagnostics import decorators

from mlflow.semantic_kernel.autolog import (
    _semantic_kernel_chat_completion_error_wrapper,
    _semantic_kernel_chat_completion_input_wrapper,
    _semantic_kernel_chat_completion_response_wrapper,
    setup_semantic_kernel_tracing,
)
from mlflow.semantic_kernel.tracing_utils import (
    _create_trace_wrapper,
    _parse_chat_inputs,
    _parse_embedding_inputs,
    _parse_kernel_invoke_inputs,
    _parse_kernel_invoke_prompt_inputs,
    _parse_text_inputs,
    _serialize_chat_output,
    _serialize_kernel_output,
    _serialize_text_output,
    _streaming_not_supported_wrapper,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "semantic_kernel"


@experimental(version="3.0.1")
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Anthropic to MLflow.
    Only synchronous calls are supported. Asynchnorous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for Anthropic models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Anthropic autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Anthropic
            autologging. If ``False``, show all events and warnings.
    """

    setup_semantic_kernel_tracing()

    streaming_methods = [
        "get_streaming_chat_message_content",
        "get_streaming_chat_message_contents",
        "_inner_get_streaming_chat_message_contents",
        "get_streaming_text_content",
        "get_streaming_text_contents",
        "_inner_get_streaming_text_contents",
        "invoke_stream",
        "invoke_prompt_stream",
    ]

    # Method to parser mapping
    method_parsers = {
        # Chat completion methods
        "get_chat_message_content": _parse_chat_inputs,
        "get_chat_message_contents": _parse_chat_inputs,
        "get_streaming_chat_message_content": _parse_chat_inputs,
        "get_streaming_chat_message_contents": _parse_chat_inputs,
        "_inner_get_chat_message_contents": _parse_chat_inputs,
        "_inner_get_streaming_chat_message_contents": _parse_chat_inputs,
        # Text completion methods
        "get_text_content": _parse_text_inputs,
        "get_text_contents": _parse_text_inputs,
        "get_streaming_text_content": _parse_text_inputs,
        "get_streaming_text_contents": _parse_text_inputs,
        "_inner_get_text_contents": _parse_text_inputs,
        "_inner_get_streaming_text_contents": _parse_text_inputs,
        # Embedding methods
        "generate_embeddings": _parse_embedding_inputs,
        "generate_raw_embeddings": _parse_embedding_inputs,
        # Kernel methods
        "invoke": _parse_kernel_invoke_inputs,
        "invoke_stream": _parse_kernel_invoke_inputs,
        "invoke_prompt": _parse_kernel_invoke_prompt_inputs,
        "invoke_prompt_stream": _parse_kernel_invoke_prompt_inputs,
    }

    # Method to serializer mapping
    method_serializers = {
        # Chat completion methods
        "get_chat_message_content": _serialize_chat_output,
        "get_chat_message_contents": _serialize_chat_output,
        "get_streaming_chat_message_content": _serialize_chat_output,
        "get_streaming_chat_message_contents": _serialize_chat_output,
        "_inner_get_chat_message_contents": _serialize_chat_output,
        "_inner_get_streaming_chat_message_contents": _serialize_chat_output,
        # Text completion methods
        "get_text_content": _serialize_text_output,
        "get_text_contents": _serialize_text_output,
        "get_streaming_text_content": _serialize_text_output,
        "get_streaming_text_contents": _serialize_text_output,
        "_inner_get_text_contents": _serialize_text_output,
        "_inner_get_streaming_text_contents": _serialize_text_output,
        # Kernel methods
        "invoke": _serialize_kernel_output,
        "invoke_stream": _serialize_kernel_output,
        "invoke_prompt": _serialize_kernel_output,
        "invoke_prompt_stream": _serialize_kernel_output,
        # Note: Embedding methods excluded as they don't generate traces
    }

    entry_point_patches = [
        (
            ChatCompletionClientBase,
            [
                "get_chat_message_content",
                "get_chat_message_contents",
                "get_streaming_chat_message_content",
                "get_streaming_chat_message_contents",
                "_inner_get_chat_message_contents",
                "_inner_get_streaming_chat_message_contents",
            ],
        ),
        (
            TextCompletionClientBase,
            [
                "get_text_content",
                "get_text_contents",
                "get_streaming_text_content",
                "get_streaming_text_contents",
                "_inner_get_text_contents",
                "_inner_get_streaming_text_contents",
            ],
        ),
        (EmbeddingGeneratorBase, ["generate_embeddings", "generate_raw_embeddings"]),
        (Kernel, ["invoke", "invoke_stream", "invoke_prompt", "invoke_prompt_stream"]),
    ]

    for cls, methods in entry_point_patches:
        for method in methods:
            if hasattr(cls, method):
                # Use streaming wrapper for streaming methods
                if method in streaming_methods:
                    safe_patch(FLAVOR_NAME, cls, method, _streaming_not_supported_wrapper)
                else:
                    # Create wrapper with the appropriate parser and serializer
                    parser = method_parsers.get(method)
                    serializer = method_serializers.get(method)
                    safe_patch(FLAVOR_NAME, cls, method, _create_trace_wrapper(parser, serializer))

    patches = [
        ("_set_completion_input", _semantic_kernel_chat_completion_input_wrapper),
        ("_set_completion_response", _semantic_kernel_chat_completion_response_wrapper),
        ("_set_completion_error", _semantic_kernel_chat_completion_error_wrapper),
    ]

    for method_name, wrapper in patches:
        safe_patch(
            FLAVOR_NAME,
            decorators,
            method_name,
            wrapper,
        )
