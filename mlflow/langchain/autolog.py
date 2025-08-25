import logging

from mlflow.langchain.constant import FLAVOR_NAME
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import safe_patch

logger = logging.getLogger(__name__)


@autologging_integration(FLAVOR_NAME)
def autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    """
    Enables (or disables) and configures autologging from Langchain to MLflow.

    Args:
        disable: If ``True``, disables the Langchain autologging integration. If ``False``,
            enables the Langchain autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            langchain that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Langchain
            autologging. If ``False``, show all events and warnings during Langchain
            autologging.
        log_traces: If ``True``, traces are logged for Langchain models by using
            MlflowLangchainTracer as a callback during inference. If ``False``, no traces are
            collected during inference. Default to ``True``.
    """
    try:
        from langchain_core.callbacks import BaseCallbackManager

        safe_patch(
            FLAVOR_NAME,
            BaseCallbackManager,
            "__init__",
            _patched_callback_manager_init,
        )
    except Exception as e:
        logger.warning(f"Failed to enable tracing for LangChain. Error: {e}")

    # Special handlings for edge cases.
    try:
        from langchain_core.callbacks import BaseCallbackManager
        from langchain_core.runnables import RunnableSequence

        safe_patch(
            FLAVOR_NAME,
            RunnableSequence,
            "batch",
            _patched_runnable_sequence_batch,
        )

        safe_patch(
            FLAVOR_NAME,
            BaseCallbackManager,
            "merge",
            _patched_callback_manager_merge,
        )
    except Exception:
        logger.debug("Failed to patch RunnableSequence or BaseCallbackManager.", exc_info=True)


def _patched_callback_manager_init(original, self, *args, **kwargs):
    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

    original(self, *args, **kwargs)

    if not AutoLoggingConfig.init(FLAVOR_NAME).log_traces:
        return

    for handler in self.inheritable_handlers:
        if isinstance(handler, MlflowLangchainTracer):
            return

    _handler = MlflowLangchainTracer()
    self.add_handler(_handler, inherit=True)


def _patched_callback_manager_merge(original, self, *args, **kwargs):
    """
    Patch BaseCallbackManager.merge to avoid a duplicated callback issue.

    In the above patched __init__, we check `inheritable_handlers` to see if the MLflow tracer
    is already propagated. This works when the `inheritable_handlers` is specified as constructor
    arguments. However, in the `merge` method, LangChain does not use constructor but set
    callbacks via the setter method. This causes duplicated callbacks injection.
    https://github.com/langchain-ai/langchain/blob/d9a069c414a321e7a3f3638a32ecf8a37ec2d188/libs/core/langchain_core/callbacks/base.py#L962-L982
    """
    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

    # Get the MLflow callback inherited from parent
    inherited = self.inheritable_handlers + args[0].inheritable_handlers
    inherited_mlflow_cb = next(
        (cb for cb in inherited if isinstance(cb, MlflowLangchainTracer)), None
    )

    if not inherited_mlflow_cb:
        return original(self, *args, **kwargs)

    merged = original(self, *args, **kwargs)
    # If a new MLflow callback is generated inside __init__, remove it
    duplicate_mlflow_cbs = [
        cb
        for cb in merged.inheritable_handlers
        if isinstance(cb, MlflowLangchainTracer) and cb != inherited_mlflow_cb
    ]
    for cb in duplicate_mlflow_cbs:
        merged.remove_handler(cb)

    return merged


def _patched_runnable_sequence_batch(original, self, *args, **kwargs):
    """
    Patch to terminate span context attachment during batch execution.

    RunnableSequence's batch() methods are implemented in a peculiar way
    that iterates on steps->items sequentially within the same thread. For example, if a
    sequence has 2 steps and the batch size is 3, the execution flow will be:
      - Step 1 for item 1
      - Step 1 for item 2
      - Step 1 for item 3
      - Step 2 for item 1
      - Step 2 for item 2
      - Step 2 for item 3
    Due to this behavior, we cannot attach the span to the context for this particular
    API, otherwise spans for different inputs will be mixed up.
    """
    from mlflow.langchain.langchain_tracer import _should_attach_span_to_context

    original_state = _should_attach_span_to_context.get()
    _should_attach_span_to_context.set(False)
    try:
        return original(self, *args, **kwargs)
    finally:
        _should_attach_span_to_context.set(original_state)
