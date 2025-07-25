from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.telemetry.model_diagnostics import decorators

from mlflow.entities import SpanType
from mlflow.semantic_kernel.autolog import setup_semantic_kernel_tracing
from mlflow.semantic_kernel.tracing_utils import (
    create_trace_wrapper,
    semantic_kernel_diagnostics_wrapper,
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
            create_trace_wrapper(SpanType.CHAT_MODEL),
        )

    text_entry_points = ["get_text_content", "get_text_contents"]
    for method in text_entry_points:
        safe_patch(
            FLAVOR_NAME,
            TextCompletionClientBase,
            method,
            create_trace_wrapper(SpanType.LLM),
        )

    # NOTE: Semantic Kernel currently does not instrument embeddings with OpenTelemetry
    # embedding_entry_points = ["generate_embeddings", "generate_raw_embeddings"]
    kernel_entry_points = ["invoke", "invoke_prompt"]
    for method in kernel_entry_points:
        safe_patch(
            FLAVOR_NAME,
            Kernel,
            method,
            create_trace_wrapper(SpanType.AGENT),
        )

    # NB: Semantic Kernel uses logging to serialize inputs/outputs. These parsers are used by their
    # tracing decorators to log the inputs/outputs. These patches give coverage for many additional
    # methods that are not covered by above entry point patches.
    for method_name in [
        "_set_completion_input",
        "_set_completion_response",
        "_set_completion_error",
    ]:
        safe_patch(
            FLAVOR_NAME,
            decorators,
            method_name,
            semantic_kernel_diagnostics_wrapper,
        )
