import contextlib
import inspect
import logging
from typing import Any

import mlflow
from mlflow.ml_package_versions import FLAVOR_TO_MODULE_NAME
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS, autologging_conf_lock

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
@autologging_conf_lock
def configure_autologging_for_evaluation(enable_tracing: bool = True):
    """
    Temporarily override the autologging configuration for all flavors during the model evaluation.
    For example, model auto-logging must be disabled during the evaluation. After the evaluation
    is done, the original autologging configurations are restored.

    Args:
        enable_tracing (bool): Whether to enable tracing for the supported flavors during eval.
    """
    flavor_to_original_config = {}
    trace_enabled_flavors = []

    # AUTOLOGGING_INTEGRATIONS can change during we iterate over flavors and enable/disable
    # autologging, therefore, we snapshot the current configuration to restore it later.
    global_config_snapshot = AUTOLOGGING_INTEGRATIONS.copy()

    for flavor in FLAVOR_TO_MODULE_NAME:
        # TODO: Remove this once Gluon autologging is actually removed
        if flavor == "gluon":
            continue

        try:
            if autolog := _get_autolog_function(flavor):
                original_config = global_config_snapshot.get(flavor, {}).copy()

                # If autologging is explicitly disabled, do nothing.
                if original_config.get("disable", False):
                    continue

                elif enable_tracing and _is_trace_autologging_supported(flavor):
                    # set all log_xyz params to False except log_traces
                    new_config = {
                        k: False if k.startswith("log_") else v for k, v in original_config.items()
                    }
                    new_config |= {"log_traces": True, "silent": True}
                    _kwargs_safe_invoke(autolog, new_config)
                    trace_enabled_flavors.append(flavor)
                elif flavor in AUTOLOGGING_INTEGRATIONS:
                    # For flavors that does not support tracing, disable autologging
                    autolog(disable=True)

                flavor_to_original_config[flavor] = original_config

        except Exception as e:
            if isinstance(e, ImportError):
                _logger.debug(f"Failed to import {flavor}. Skip updating autologging.")
            else:
                _logger.info(f"Failed to update autologging configuration for flavor {flavor}. {e}")

            # Autologging configuration might be updated before the exception is raised,
            # which needs to be reverted.
            AUTOLOGGING_INTEGRATIONS.pop(flavor, None)

    if trace_enabled_flavors:
        _logger.info(
            "Tracing is temporarily enabled during the model evaluation for computing some "
            "metrics and debugging. To disable tracing, call `mlflow.autolog(disable=True)`."
        )

    try:
        yield
    finally:
        # Restore original autologging configurations.
        for flavor, original_config in flavor_to_original_config.items():
            autolog = _get_autolog_function(flavor)
            try:
                if original_config:
                    _kwargs_safe_invoke(autolog, original_config)
                    AUTOLOGGING_INTEGRATIONS[flavor] = original_config
                else:
                    # If the original configuration is empty, autologging was not enabled before
                    autolog(disable=True)
                    # We also need to remove the configuration entry from AUTOLOGGING_INTEGRATIONS,
                    # so as not to confuse with the case user explicitly disabled autologging.
                    AUTOLOGGING_INTEGRATIONS.pop(flavor, None)
            except ImportError:
                pass


def _kwargs_safe_invoke(func: callable, kwargs: dict[str, Any]):
    """
    Invoke the function with the given dictionary as keyword arguments, but only include the
    arguments that are present in the function's signature.

    This is particularly used for calling autolog() function with the configuration dictionary
    stored in AUTOLOGGING_INTEGRATIONS. While the config keys mostly align with the autolog()'s
    signature by design, some keys are not present in autolog(), such as "globally_configured".
    """
    sig = inspect.signature(func)
    return func(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _get_autolog_function(flavor_name: str):
    """Get the autolog() function for the specified flavor."""
    flavor_module = getattr(mlflow, flavor_name, None)
    return getattr(flavor_module, "autolog", None)


def _is_trace_autologging_supported(flavor_name: str) -> bool:
    """Check if the given flavor supports trace autologging."""
    if autolog_func := _get_autolog_function(flavor_name):
        return "log_traces" in inspect.signature(autolog_func).parameters
    return False
