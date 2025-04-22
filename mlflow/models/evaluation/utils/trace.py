import contextlib
import inspect
import logging
from typing import Any, Callable

from mlflow.ml_package_versions import FLAVOR_TO_MODULE_NAME
from mlflow.utils.autologging_utils import (
    AUTOLOGGING_INTEGRATIONS,
    autologging_conf_lock,
    get_autolog_function,
    is_autolog_supported,
)
from mlflow.utils.autologging_utils.safety import revert_patches
from mlflow.utils.import_hooks import (
    _post_import_hooks,
    get_post_import_hooks,
    register_post_import_hook,
)

_logger = logging.getLogger(__name__)


# This flag is used to display the message only once when tracing is enabled during the evaluation.
_SHOWN_TRACE_MESSAGE_BEFORE = False


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
    original_import_hooks = {}
    new_import_hooks = {}

    # AUTOLOGGING_INTEGRATIONS can change during we iterate over flavors and enable/disable
    # autologging, therefore, we snapshot the current configuration to restore it later.
    global_config_snapshot = AUTOLOGGING_INTEGRATIONS.copy()

    for flavor in FLAVOR_TO_MODULE_NAME:
        if not is_autolog_supported(flavor):
            continue

        original_config = global_config_snapshot.get(flavor, {}).copy()

        # If autologging is explicitly disabled, do nothing.
        if original_config.get("disable", False):
            continue

        # NB: Using post-import hook to configure the autologging lazily when the target
        # flavor's module is imported, rather than configuring it immediately. This is
        # because the evaluation code usually only uses a subset of the supported flavors,
        # hence we want to avoid unnecessary overhead of configuring all flavors.
        @autologging_conf_lock
        def _setup_autolog(module):
            try:
                autolog = get_autolog_function(flavor)

                # If tracing is supported and not explicitly disabled, enable it.
                if enable_tracing and _should_enable_tracing(flavor, global_config_snapshot):
                    new_config = {
                        k: False if k.startswith("log_") else v for k, v in original_config.items()
                    }
                    new_config |= {"log_traces": True, "silent": True}
                    _kwargs_safe_invoke(autolog, new_config)

                    global _SHOWN_TRACE_MESSAGE_BEFORE
                    if not _SHOWN_TRACE_MESSAGE_BEFORE:
                        _logger.info(
                            "Auto tracing is temporarily enabled during the model evaluation "
                            "for computing some metrics and debugging. To disable tracing, call "
                            "`mlflow.autolog(disable=True)`."
                        )
                        _SHOWN_TRACE_MESSAGE_BEFORE = True
                else:
                    autolog(disable=True)

            except Exception:
                _logger.debug(f"Failed to update autologging config for {flavor}.", exc_info=True)

        module = FLAVOR_TO_MODULE_NAME[flavor]
        try:
            original_import_hooks[module] = get_post_import_hooks(module)
            new_import_hooks[module] = _setup_autolog
            register_post_import_hook(_setup_autolog, module, overwrite=True)
        except Exception:
            _logger.debug(f"Failed to register post-import hook for {flavor}.", exc_info=True)

    try:
        yield
    finally:
        # Remove post-import hooks and patches the are registered during the evaluation.
        for module, hooks in new_import_hooks.items():
            # Restore original post-import hooks if any. Note that we don't use
            # register_post_import_hook method to bypass some pre-checks and just
            # restore the original state.
            if hooks is None:
                _post_import_hooks.pop(module, None)
            else:
                _post_import_hooks[module] = original_import_hooks[module]

        # If any autologging configuration is updated, restore original autologging configurations.
        for flavor, new_config in AUTOLOGGING_INTEGRATIONS.copy().items():
            original_config = global_config_snapshot.get(flavor)
            if original_config != new_config:
                try:
                    autolog = get_autolog_function(flavor)
                    if original_config:
                        _kwargs_safe_invoke(autolog, original_config)
                        AUTOLOGGING_INTEGRATIONS[flavor] = original_config
                    else:
                        # If the original configuration is empty, autologging was not enabled before
                        autolog(disable=True)
                        # Remove all safe_patch applied by autologging
                        revert_patches(flavor)
                        # We also need to remove the config entry from AUTOLOGGING_INTEGRATIONS,
                        # so as not to confuse with the case user explicitly disabled autologging.
                        AUTOLOGGING_INTEGRATIONS.pop(flavor, None)
                except ImportError:
                    pass
                except Exception as e:
                    if original_config is None or (
                        not original_config.get("disable", False)
                        and not original_config.get("silent", False)
                    ):
                        _logger.warning(
                            f"Exception raised while calling autologging for {flavor}: {e}"
                        )


def _should_enable_tracing(flavor: str, autologging_config: dict[str, Any]) -> bool:
    """
    Check if tracing should be enabled for the given flavor during the model evaluation.
    """
    # 1. Check if the autologging or tracing is globally disabled
    # TODO: This check should not take precedence over the flavor-specific configuration
    # set by the explicit mlflow.<flavor>.autolog() call by users.
    # However, in Databricks, sometimes mlflow.<flavor>.autolog() is automatically
    # called in the kernel startup, which is confused with the user's action. In
    # such cases, even when user disables autologging globally, the flavor-specific
    # autologging remains enabled. We are going to fix the Databricks side issue,
    # and after that, we should move this check down after the flavor-specific check.
    global_config = autologging_config.get("mlflow", {})
    if global_config.get("disable", False) or (not global_config.get("log_traces", True)):
        return False

    if not _is_trace_autologging_supported(flavor):
        return False

    # 3. Check if tracing is explicitly disabled for the flavor
    flavor_config = autologging_config.get(flavor, {})
    return flavor_config.get("log_traces", True)


def _kwargs_safe_invoke(func: Callable[..., Any], kwargs: dict[str, Any]):
    """
    Invoke the function with the given dictionary as keyword arguments, but only include the
    arguments that are present in the function's signature.

    This is particularly used for calling autolog() function with the configuration dictionary
    stored in AUTOLOGGING_INTEGRATIONS. While the config keys mostly align with the autolog()'s
    signature by design, some keys are not present in autolog(), such as "globally_configured".
    """
    sig = inspect.signature(func)
    return func(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _is_trace_autologging_supported(flavor_name: str) -> bool:
    """Check if the given flavor supports trace autologging."""
    if autolog_func := get_autolog_function(flavor_name):
        return "log_traces" in inspect.signature(autolog_func).parameters
    return False
