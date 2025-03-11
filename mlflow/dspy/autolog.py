import logging

from mlflow.dspy.save import FLAVOR_NAME, log_model
from mlflow.dspy.util import save_dspy_module_state
from mlflow.tracing.provider import trace_disabled
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
    safe_patch,
)

_logger = logging.getLogger(__name__)


@experimental
def autolog(
    log_traces: bool = True,
    log_traces_from_compile: bool = False,
    log_traces_from_eval: bool = True,
    log_compiles: bool = False,
    log_models: bool = False,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from DSPy to MLflow. Currently, the
    MLflow DSPy flavor only supports autologging for tracing.

    Args:
        log_traces: If ``True``, traces are logged for DSPy models by using. If ``False``,
            no traces are collected during inference. Default to ``True``.
        log_traces_from_compile: If ``True``, traces are logged when compiling (optimizing)
            DSPy programs. If ``False``, traces are only logged from normal model inference and
            disabled when compiling. Default to ``False``.
        log_traces_from_eval: If ``True``, traces are logged for DSPy models when running DSPy's
            `built-in evaluator <https://dspy.ai/learn/evaluation/metrics/#evaluation>`_.
            If ``False``, traces are only logged from normal model inference and disabled when
            running the evaluator. Default to ``True``.
        log_compiles: If ``True``, information about the optimization process is logged when
            `Teleprompter.compile()` is called.
        log_models: If ``True``, optimized models are logged as MLflow model artifacts.
            If ``False``, optimized models are not logged.
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
    _autolog(
        log_traces=log_traces,
        log_traces_from_compile=log_traces_from_compile,
        log_traces_from_eval=log_traces_from_eval,
        log_compiles=log_compiles,
        log_models=log_models,
        disable=disable,
        silent=silent,
    )

    if log_models and (not log_compiles):
        _logger.warning(
            "`log_model=True` is not effective without `log_compiles=True`. "
            "Consider setting `log_compiles=True` to log models."
        )

    import dspy

    from mlflow.dspy.callback import MlflowCallback

    # Enable tracing by setting the MlflowCallback
    if not disable:
        if not any(isinstance(c, MlflowCallback) for c in dspy.settings.callbacks):
            dspy.settings.configure(callbacks=[*dspy.settings.callbacks, MlflowCallback()])
    else:
        dspy.settings.configure(
            callbacks=[c for c in dspy.settings.callbacks if not isinstance(c, MlflowCallback)]
        )

    # Patch teleprompter/evaluator not to generate traces by default
    def trace_disabled_fn(original, self, *args, **kwargs):
        # NB: Since calling mlflow.dspy.autolog() again does not unpatch a function, we need to
        # check this flag at runtime to determine if we should generate traces.
        @trace_disabled
        def _fn(self, *args, **kwargs):
            return original(self, *args, **kwargs)

        def _compile_fn(self, *args, **kwargs):
            if callback := _active_callback():
                callback._within_compile = True
            try:
                if get_autologging_config(FLAVOR_NAME, "log_traces_from_compile"):
                    result = original(self, *args, **kwargs)
                else:
                    result = _fn(self, *args, **kwargs)
                return result
            finally:
                if callback:
                    callback._within_compile = False

        if isinstance(self, Teleprompter):
            if not get_autologging_config(FLAVOR_NAME, "log_compiles"):
                return _compile_fn(self, *args, **kwargs)

            # TODO: we may want to save the trainset with mlflow.log_input()
            program = _compile_fn(self, *args, **kwargs)
            # Save the state of the best model in json format
            # so that users can see the demonstrations and instructions.
            save_dspy_module_state(program, "best_model.json")
            if get_autologging_config(FLAVOR_NAME, "log_models"):
                log_model(program, "model")

            return program

        if isinstance(self, Teleprompter) and get_autologging_config(
            FLAVOR_NAME, "log_traces_from_compile"
        ):
            return original(self, *args, **kwargs)

        if isinstance(self, Evaluate) and get_autologging_config(
            FLAVOR_NAME, "log_traces_from_eval"
        ):
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
                manage_run=get_autologging_config(FLAVOR_NAME, "log_compiles"),
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
    log_traces_from_compile: bool,
    log_traces_from_eval: bool,
    log_compiles: bool,
    log_models: bool,
    disable: bool = False,
    silent: bool = False,
):
    """
    TODO: Implement patching logic for autologging artifacts.
    """


def _active_callback():
    import dspy

    from mlflow.dspy.callback import MlflowCallback

    for callback in dspy.settings.callbacks:
        if isinstance(callback, MlflowCallback):
            return callback
