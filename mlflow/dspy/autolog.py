from mlflow.dspy.save import FLAVOR_NAME
from mlflow.tracing.provider import trace_disabled
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch


@experimental
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from DSPy to MLflow. Currently, the
    MLflow DSPy flavor only supports autologging for tracing.

    Args:
        log_traces: If ``True``, traces are logged for DSPy models by using. If ``False``,
            no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the DSPy autologging integration. If ``False``,
            enables the DSPy autologging integration.
        silent: If ``True``, suppress all event logs and warnings from MLflow during DSPy
            autologging. If ``False``, show all events and warnings.
    """
    # NB: The @autologging_integration annotation is used for adding shared logic. However, one
    # caveat is that the wrapped function is NOT executed when disable=True is passed. This prevents
    # us from running cleaning up logging when autologging is turned off. To workaround this, we
    # annotate _autolog() instead of this entrypoint, and define the cleanup logic outside it.
    # This needs to be called before doing any safe-patching (otherwise safe-patch will be no-op).
    # TODO: since this implementation is inconsistent, explore a universal way to solve the issue.
    _autolog(log_traces=log_traces, disable=disable, silent=silent)

    import dspy

    from mlflow.dspy.callback import MlflowCallback

    # Enable tracing by setting the MlflowCallback
    if log_traces and not disable:
        if not any(isinstance(c, MlflowCallback) for c in dspy.settings.callbacks):
            dspy.settings.configure(callbacks=[*dspy.settings.callbacks, MlflowCallback()])
    else:
        dspy.settings.configure(
            callbacks=[c for c in dspy.settings.callbacks if not isinstance(c, MlflowCallback)]
        )

    # Patch teleprompter and evaluate not to generate traces
    def trace_disabled_fn(original, self, *args, **kwargs):
        @trace_disabled
        def _fn(self, *args, **kwargs):
            return original(self, *args, **kwargs)

        return _fn(self, *args, **kwargs)

    from dspy.evaluate import Evaluate
    from dspy.teleprompt import Teleprompter

    compile_patch = "compile"
    for cls in Teleprompter.__subclasses__():
        # NB: This is to avoid the abstraction inheritance of superclasses that are defined
        # only for the purposes of abstraction. The recursion behavior of the
        # __subclasses__ dunder method will target the appropriate subclasses we need to patch.
        if hasattr(cls, compile_patch):
            safe_patch(
                FLAVOR_NAME,
                cls,
                compile_patch,
                trace_disabled_fn,
            )

    call_patch = "__call__"
    if hasattr(Evaluate, call_patch):
        safe_patch(
            FLAVOR_NAME,
            Evaluate,
            call_patch,
            trace_disabled_fn,
        )


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_traces: bool,
    disable: bool = False,
    silent: bool = False,
):
    """
    TODO: Implement patching logic for autologging models and artifacts.
    """
