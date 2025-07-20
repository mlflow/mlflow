from mlflow.semantic_kernel.autolog import (
    _semantic_kernel_chat_completion_error_wrapper,
    _semantic_kernel_chat_completion_input_wrapper,
    _semantic_kernel_chat_completion_response_wrapper,
    _streaming_not_supported_wrapper,
    _trace_wrapper,
    setup_semantic_kernel_tracing,
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

    from semantic_kernel.utils.telemetry.model_diagnostics import decorators

    try:
        from semantic_kernel.connectors.ai.chat_completion_client_base import (
            ChatCompletionClientBase,
        )
        from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
        from semantic_kernel.connectors.ai.text_completion_client_base import (
            TextCompletionClientBase,
        )
        from semantic_kernel.kernel import Kernel

        # Separate streaming and non-streaming methods
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
                        safe_patch(FLAVOR_NAME, cls, method, _trace_wrapper)

    except ImportError:
        pass

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
