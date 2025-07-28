from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.telemetry.model_diagnostics import decorators

from mlflow.semantic_kernel.autolog import setup_semantic_kernel_tracing
from mlflow.semantic_kernel.tracing_utils import (
    patched_kernel_entry_point,
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
    Enables (or disables) and configures autologging from Semantic Kernel to MLflow.
    Only synchronous calls are supported. Asynchnorous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for Semantic Kernel.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Semantic Kernel  autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Anthropic
            autologging. If ``False``, show all events and warnings.
    """

    setup_semantic_kernel_tracing()

    # Create root spans for the kernel entry points.
    for method in ["invoke", "invoke_prompt"]:
        safe_patch(
            FLAVOR_NAME,
            Kernel,
            method,
            patched_kernel_entry_point,
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
