from mlflow.semantic_kernel.autolog import (
    _semantic_kernel_chat_completion_error_wrapper,
    _semantic_kernel_chat_completion_input_wrapper,
    _semantic_kernel_chat_completion_response_wrapper,
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
