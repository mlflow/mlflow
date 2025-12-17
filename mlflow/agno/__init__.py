import inspect
import logging

from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "agno"
_logger = logging.getLogger(__name__)


@experimental(version="3.3.0")
def autolog(*, log_traces: bool = True, disable: bool = False, silent: bool = False) -> None:
    """
    Enables (or disables) and configures autologging from Agno to MLflow.

    For Agno V2 (>= 2.0.0), this uses OpenTelemetry instrumentation via OpenInference.

    Args:
        log_traces: If ``True``, traces are logged for Agno Agents.
        disable: If ``True``, disables Agno autologging.
        silent: If ``True``, suppresses all MLflow event logs and warnings.
    """
    from mlflow.agno.autolog_v1 import patched_async_class_call, patched_class_call
    from mlflow.agno.autolog_v2 import _is_agno_v2, _setup_otel_instrumentation, _uninstrument_otel

    # NB: The @autologging_integration annotation is used for adding shared logic. However, one
    # caveat is that the wrapped function is NOT executed when disable=True is passed. This prevents
    # us from running cleaning up logging when autologging is turned off. To workaround this, we
    # annotate _autolog() instead of this entrypoint, and define the cleanup logic outside it.
    # This needs to be called before doing any safe-patching (otherwise safe-patch will be no-op).
    _autolog(log_traces=log_traces, disable=disable, silent=silent)

    # Check if Agno V2 is installed
    if _is_agno_v2():
        _logger.debug("Detected Agno V2, using OpenTelemetry instrumentation")
        if disable or not log_traces:
            _uninstrument_otel()
        else:
            _setup_otel_instrumentation()
        _record_event(
            AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
        )
        return

    # For Agno V1, use the existing patching method
    from mlflow.agno.utils import discover_storage_backends, find_model_subclasses

    class_map = {
        "agno.agent.Agent": ["run", "arun"],
        "agno.team.Team": ["run", "arun"],
        "agno.tools.function.FunctionCall": ["execute", "aexecute"],
    }

    if storages := discover_storage_backends():
        class_map.update(
            {
                cls.__module__ + "." + cls.__name__: [
                    "create",
                    "read",
                    "upsert",
                    "drop",
                    "upgrade_schema",
                ]
                for cls in storages
            }
        )

    if models := find_model_subclasses():
        class_map.update(
            {
                # TODO: Support streaming
                cls.__module__ + "." + cls.__name__: ["invoke", "ainvoke"]
                for cls in models
            }
        )

    for cls_path, methods in class_map.items():
        mod_name, cls_name = cls_path.rsplit(".", 1)
        try:
            module = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError) as exc:
            _logger.debug("Agno autologging: failed to import %s – %s", cls_path, exc)
            continue

        for method_name in methods:
            try:
                original = getattr(cls, method_name)
                wrapper = (
                    patched_async_class_call
                    if inspect.iscoroutinefunction(original)
                    else patched_class_call
                )
                safe_patch(FLAVOR_NAME, cls, method_name, wrapper)
            except AttributeError as exc:
                _logger.debug(
                    "Agno autologging: cannot patch %s.%s – %s", cls_path, method_name, exc
                )

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_traces: bool,
    disable: bool = False,
    silent: bool = False,
):
    pass
