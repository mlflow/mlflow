import logging
import importlib
import sys
from mlflow.exceptions import MlflowException, BAD_REQUEST

_logger = logging.getLogger(__name__)


_BUILTIN_METRIC_TO_GREATER_IS_BETTER = {
    # metric_name: greater_is_better
    "mean_absolute_error": False,
    "mean_squared_error": False,
    "root_mean_squared_error": False,
    "max_error": False,
    "mean_absolute_percentage_error": False,
}


def _get_custom_metrics(step_config):
    return (step_config.get("metrics") or {}).get("custom")


def _load_custom_metric_functions(pipeline_root, step_config):
    custom_metrics = _get_custom_metrics(step_config)
    if not custom_metrics:
        return None
    try:
        sys.path.append(pipeline_root)
        custom_metrics_mod = importlib.import_module("steps.custom_metrics")
        return [getattr(custom_metrics_mod, cm["function"]) for cm in custom_metrics]
    except Exception as e:
        raise MlflowException(
            message="Failed to load custom metric functions",
            error_code=BAD_REQUEST,
        ) from e


def _get_primary_metric(step_config):
    return (step_config.get("metrics") or {}).get("primary", "root_mean_squared_error")


def _get_metric_greater_is_better(step_config):
    custom_metrics = _get_custom_metrics(step_config)
    custom_metric_greater_is_better = (
        {cm["name"]: cm["greater_is_better"] for cm in custom_metrics} if custom_metrics else {}
    )
    overridden_builtin_metrics = set(custom_metric_greater_is_better.keys()).intersection(
        _BUILTIN_METRIC_TO_GREATER_IS_BETTER.keys()
    )
    if overridden_builtin_metrics:
        _logger.warning(
            "Custom metrics override the following built-in metrics: %s",
            sorted(overridden_builtin_metrics),
        )
    metric_to_greater_is_better = {
        **_BUILTIN_METRIC_TO_GREATER_IS_BETTER,
        **custom_metric_greater_is_better,
    }
    return metric_to_greater_is_better
