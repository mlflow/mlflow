import importlib
import logging

from packaging.version import Version

import mlflow
from mlflow.dspy.constant import FLAVOR_NAME
from mlflow.tracing.provider import trace_disabled
from mlflow.tracing.utils import construct_full_inputs
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
    safe_patch,
)
from mlflow.utils.autologging_utils.safety import exception_safe_function_for_class

_logger = logging.getLogger(__name__)


def autolog(
    log_traces: bool = True,
    log_traces_from_compile: bool = False,
    log_traces_from_eval: bool = True,
    log_compiles: bool = False,
    log_evals: bool = False,
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
        log_evals: If ``True``, information about the evaluation call is logged when
            `Evaluate.__call__()` is called.
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
        log_evals=log_evals,
        disable=disable,
        silent=silent,
    )

    import dspy

    from mlflow.dspy.callback import MlflowCallback

    # Enable tracing by setting the MlflowCallback
    if not disable:
        if not any(isinstance(c, MlflowCallback) for c in dspy.settings.callbacks):
            dspy.settings.configure(callbacks=[*dspy.settings.callbacks, MlflowCallback()])
        # DSPy token tracking has an issue before 3.0.4: https://github.com/stanfordnlp/dspy/pull/8831
        if Version(importlib.metadata.version("dspy")) >= Version("3.0.4"):
            dspy.settings.configure(track_usage=True)

    else:
        dspy.settings.configure(
            callbacks=[c for c in dspy.settings.callbacks if not isinstance(c, MlflowCallback)]
        )

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
                _patched_compile,
                manage_run=get_autologging_config(FLAVOR_NAME, "log_compiles"),
            )

    from dspy.evaluate import Evaluate

    call_patch = "__call__"
    if hasattr(Evaluate, call_patch):
        safe_patch(
            FLAVOR_NAME,
            Evaluate,
            call_patch,
            _patched_evaluate,
        )


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_traces: bool,
    log_traces_from_compile: bool,
    log_traces_from_eval: bool,
    log_compiles: bool,
    log_evals: bool,
    disable: bool = False,
    silent: bool = False,
):
    pass


def _active_callback():
    import dspy

    from mlflow.dspy.callback import MlflowCallback

    for callback in dspy.settings.callbacks:
        if isinstance(callback, MlflowCallback):
            return callback


def _patched_compile(original, self, *args, **kwargs):
    from mlflow.dspy.util import (
        log_dspy_dataset,
        log_dspy_lm_state,
        log_dummy_model_outputs,
        save_dspy_module_state,
    )

    # NB: Since calling mlflow.dspy.autolog() again does not unpatch a function, we need to
    # check this flag at runtime to determine if we should generate traces.
    # method to disable tracing for compile and evaluate by default
    @trace_disabled
    def _trace_disabled_fn(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    def _compile_fn(self, *args, **kwargs):
        if callback := _active_callback():
            callback.optimizer_stack_level += 1
        try:
            if get_autologging_config(FLAVOR_NAME, "log_traces_from_compile"):
                result = original(self, *args, **kwargs)
            else:
                result = _trace_disabled_fn(self, *args, **kwargs)
            return result
        finally:
            if callback:
                callback.optimizer_stack_level -= 1
                if callback.optimizer_stack_level == 0:
                    # Reset the callback state after the completion of root compile
                    callback.reset()

    if not get_autologging_config(FLAVOR_NAME, "log_compiles"):
        return _compile_fn(self, *args, **kwargs)

    # NB: Log a dummy run outputs such that "Run" tab is shown in the UI. Currently, the
    # GenAI experiment does not show the "Run" tab without this, which is critical gap for
    # DSPy users. This should be done BEFORE the compile call, because Run page is used
    # for tracking the compile progress, not only after finishing the compile.
    log_dummy_model_outputs()

    program = _compile_fn(self, *args, **kwargs)
    # Save the state of the best model in json format
    # so that users can see the demonstrations and instructions.
    save_dspy_module_state(program, "best_model.json")

    # Teleprompter.get_params is introduced in dspy 2.6.15
    params = (
        self.get_params()
        if Version(importlib.metadata.version("dspy")) >= Version("2.6.15")
        else {}
    )
    # Construct the dict of arguments passed to the compile call
    inputs = construct_full_inputs(original, self, *args, **kwargs)
    # Update params with the arguments passed to the compile call
    params.update(inputs)
    mlflow.log_params({k: v for k, v in inputs.items() if isinstance(v, (int, float, str, bool))})

    # Log the current DSPy LM state
    log_dspy_lm_state()

    if trainset := inputs.get("trainset"):
        log_dspy_dataset(trainset, "trainset.json")
    if valset := inputs.get("valset"):
        log_dspy_dataset(valset, "valset.json")
    return program

    if get_autologging_config(FLAVOR_NAME, "log_traces_from_compile"):
        return original(self, *args, **kwargs)
    else:
        return _trace_disabled_fn(self, *args, **kwargs)


def _patched_evaluate(original, self, *args, **kwargs):
    # NB: Since calling mlflow.dspy.autolog() again does not unpatch a function, we need to
    # check this flag at runtime to determine if we should generate traces.
    # method to disable tracing for compile and evaluate by default
    @trace_disabled
    def _trace_disabled_fn(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    if not get_autologging_config(FLAVOR_NAME, "log_traces_from_eval"):
        return _trace_disabled_fn(self, *args, **kwargs)

    # Patch metric call to log assessment results on the prediction traces
    new_kwargs = construct_full_inputs(original, self, *args, **kwargs)
    metric = new_kwargs.get("metric") or self.metric
    new_kwargs["metric"] = _patch_metric(metric)

    args_passed_positional = list(new_kwargs.keys())[: len(args)]
    new_args = []
    for arg in args_passed_positional:
        new_args.append(new_kwargs.pop(arg))

    return original(self, *new_args, **new_kwargs)


def _patch_metric(metric):
    """Patch the metric call to log assessment results on the prediction traces."""
    import dspy

    # NB: This patch MUST not raise an exception, otherwise may interrupt the evaluation call.
    @exception_safe_function_for_class
    def _patched(*args, **kwargs):
        # NB: DSPy runs prediction and the metric call in the same thread, so we can retrieve
        # the prediction trace ID using the last active trace ID.
        # https://github.com/stanfordnlp/dspy/blob/8224a99ca6402863540aae5aa3bc5eddbd2947c4/dspy/evaluate/evaluate.py#L170-L173
        pred_trace_id = mlflow.get_last_active_trace_id(thread_local=True)
        if not pred_trace_id:
            _logger.debug("Tracing during evaluation is enabled, but no prediction trace found.")
            return metric(*args, **kwargs)

        try:
            score = metric(*args, **kwargs)
        except Exception as e:
            _logger.debug("Metric call failed, logging an assessment with error")
            mlflow.log_feedback(trace_id=pred_trace_id, name=metric.__name__, error=e)
            raise

        try:
            if isinstance(score, dspy.Prediction):
                # GEPA metric returns a Prediction object with score and feedback attributes.
                # https://dspy.ai/tutorials/gepa_aime/
                value = getattr(score, "score", None)
                rationale = getattr(score, "feedback", None)
            else:
                value = score
                rationale = None

            mlflow.log_feedback(
                trace_id=pred_trace_id,
                name=metric.__name__,
                value=value,
                rationale=rationale,
            )
        except Exception as e:
            _logger.debug(f"Failed to log feedback for metric on prediction trace: {e}")

        return score

    return _patched
