from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.telemetry.model_diagnostics import decorators

from mlflow.entities import SpanType
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
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "semantic_kernel"


@experimental(version="3.2.0")
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

    # NB: Patch all semantic kernel methods that are not covered by the chat decorators listed
    # below.
    chat_entry_points = ["get_chat_message_content", "get_chat_message_contents"]
    for method in chat_entry_points:
        safe_patch(
            FLAVOR_NAME,
            ChatCompletionClientBase,
            method,
            _create_trace_wrapper(SpanType.CHAT_MODEL, _parse_chat_inputs, _serialize_chat_output),
        )

    text_entry_points = ["get_text_content", "get_text_contents"]
    for method in text_entry_points:
        safe_patch(
            FLAVOR_NAME,
            TextCompletionClientBase,
            method,
            _create_trace_wrapper(SpanType.LLM, _parse_text_inputs, _serialize_text_output),
        )

    embedding_entry_points = ["generate_embeddings", "generate_raw_embeddings"]
    for method in embedding_entry_points:
        safe_patch(
            FLAVOR_NAME,
            EmbeddingGeneratorBase,
            method,
            _create_trace_wrapper(SpanType.EMBEDDING, _parse_embedding_inputs, None),
        )

    safe_patch(
        FLAVOR_NAME,
        Kernel,
        "invoke",
        _create_trace_wrapper(None, _parse_kernel_invoke_inputs, _serialize_kernel_output),
    )
    safe_patch(
        FLAVOR_NAME,
        Kernel,
        "invoke_prompt",
        _create_trace_wrapper(None, _parse_kernel_invoke_prompt_inputs, _serialize_kernel_output),
    )

    # NB: Semantic Kernel uses logging to serialize inputs/outputs. These parsers are used by their
    # tracing decorators to log the inputs/outputs. These patches give coverage for many additional
    # methods that are not covered by above entry point patches.
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
