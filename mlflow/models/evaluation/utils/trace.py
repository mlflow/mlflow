import contextlib
import inspect
import logging

import mlflow
from mlflow.ml_package_versions import FLAVOR_TO_MODULE_NAME
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def configure_autologging_for_evaluation(enable_tracing: bool = True):
    flavor_to_original_config = {}
    trace_enabled_flavors = []
    for flavor in FLAVOR_TO_MODULE_NAME:
        try:
            if autolog := _get_autolog_function(flavor):
                original_config = AUTOLOGGING_INTEGRATIONS.get(flavor, {})

                # If autologging is explicitly disabled, do nothing.
                if original_config.get("disable", False):
                    continue

                elif enable_tracing and _is_trace_autologging_supported(flavor):
                    # set all log_xyz params to False except log_traces
                    new_config = {
                        k: False if k.startswith("log_") else v for k, v in original_config.items()
                    }
                    new_config = {**new_config, "log_traces": True, "silent": True}
                    autolog(**new_config)
                    trace_enabled_flavors.append(flavor)
                else:
                    # For flavors that does not support tracing, disable autologging
                    autolog(disable=True)

                flavor_to_original_config[flavor] = original_config

        except ImportError:
            _logger.debug(
                f"Flavor {flavor} is not installed. Skip updating autologging configuration."
            )
            # Global autologging configuration might be updated before the exception is raised,
            # which needs to be reverted.
            if flavor in AUTOLOGGING_INTEGRATIONS:
                del AUTOLOGGING_INTEGRATIONS[flavor]

        except Exception as e:
            _logger.info(f"Failed to update autologging configuration for flavor {flavor}. {e}")

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
            if original_config:
                autolog(**original_config)
            else:
                # If the original configuration is empty, autologging was not enabled before
                autolog(disable=True)
                # We also need to remove the configuration entry from AUTOLOGGING_INTEGRATIONS,
                # so as not to confuse with the case user explicitly disabled autologging.
                del AUTOLOGGING_INTEGRATIONS[flavor]


def _get_autolog_function(flavor_name: str):
    """Get the autolog() function for the specified flavor."""
    flavor_module = getattr(mlflow, flavor_name, None)
    return getattr(flavor_module, "autolog", None)


def _is_trace_autologging_supported(flavor_name: str):
    """Check if the given flavor supports trace autologging."""
    if autolog_func := _get_autolog_function(flavor_name):
        return "log_traces" in inspect.signature(autolog_func).parameters
    return False
