"""
The ``mlflow.smolagents`` module provides an API for tracing Smolagents AI agents.
"""

import logging

from mlflow.smolagents.autolog import (
    patched_class_call,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "smolagents"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Smolagents to MLflow.
    Note that asynchronous APIs and Tool calling are not recorded now.

    Args:
        log_traces: If ``True``, traces are logged for Smolagents agents.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Smolagents autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Smolagents
            autologging. If ``False``, show all events and warnings.
    """
    import smolagents
    from smolagents import models

    class_method_map = {
        "MultiStepAgent": ["run"],
        "CodeAgent": ["step"],
        "ToolCallingAgent": ["step"],
        "Tool": ["__call__"],
    }

    try:
        for _, attr in vars(smolagents).items():
            if isinstance(attr, type) and issubclass(attr, models.Model):
                class_method_map.setdefault(attr.__name__, []).append("__call__")
    except Exception as e:
        _logger.warn("the error happens while registering models to class_method_map: %s", e)

    try:
        for class_name, methods in class_method_map.items():
            cls = getattr(smolagents, class_name)
            for method in methods:
                safe_patch(
                    FLAVOR_NAME,
                    cls,
                    method,
                    patched_class_call,
                )
    except (AttributeError, ModuleNotFoundError) as e:
        _logger.error(
            "An exception happens when applying auto-tracing to smolagents. Exception: %s", e
        )

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )
