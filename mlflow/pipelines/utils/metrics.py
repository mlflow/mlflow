import logging
import importlib
import sys
from typing import List, Dict

from mlflow.exceptions import MlflowException, BAD_REQUEST
from mlflow.pipelines.helpers import PipelineHelper, PipelineMetric

_logger = logging.getLogger(__name__)


def _get_custom_metrics(step_config: Dict, pipeline_helper: PipelineHelper) -> List[Dict]:
    """
    :param: Configuration dictionary for the train or evaluate step.
    :return: A list of custom metrics defined in the specified configuration dictionary,
             or an empty list of the configuration dictionary does not define any custom metrics.
    """
    custom_metric_dicts = (step_config.get("metrics") or {}).get("custom", [])
    custom_metrics = [
        PipelineMetric.from_custom_metric_dict(metric_dict) for metric_dict in custom_metric_dicts
    ]
    custom_metric_names = {metric.name for metric in custom_metrics}
    builtin_metric_names = {metric.name for metric in pipeline_helper.builtin_metrics()}
    overridden_builtin_metrics = custom_metric_names.intersection(builtin_metric_names)
    if overridden_builtin_metrics:
        _logger.warning(
            "Custom metrics override the following built-in metrics: %s",
            sorted(overridden_builtin_metrics),
        )
    return custom_metrics


def _load_custom_metric_functions(
    pipeline_root: str, metrics: List[PipelineMetric]
) -> List[callable]:
    custom_metric_function_names = [
        metric.custom_function for metric in metrics if metric.custom_function is not None
    ]
    if not custom_metric_function_names:
        return None

    try:
        sys.path.append(pipeline_root)
        custom_metrics_mod = importlib.import_module("steps.custom_metrics")
        return [
            getattr(custom_metrics_mod, custom_metric_function_name)
            for custom_metric_function_name in custom_metric_function_names
        ]
    except Exception as e:
        raise MlflowException(
            message="Failed to load custom metric functions",
            error_code=BAD_REQUEST,
        ) from e


def _get_primary_metric(step_config):
    return (step_config.get("metrics") or {}).get("primary", "root_mean_squared_error")
