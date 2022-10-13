import logging
import importlib
import sys
from typing import List, Dict, Optional

from mlflow.exceptions import MlflowException, BAD_REQUEST
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)


class PipelineMetric:

    _KEY_METRIC_NAME = "name"
    _KEY_METRIC_GREATER_IS_BETTER = "greater_is_better"
    _KEY_CUSTOM_FUNCTION = "function"

    def __init__(self, name: str, greater_is_better: bool, custom_function: Optional[str] = None):
        self.name = name
        self.greater_is_better = greater_is_better
        self.custom_function = custom_function

    @classmethod
    def from_custom_metric_dict(cls, custom_metric_dict):
        metric_name = custom_metric_dict.get(PipelineMetric._KEY_METRIC_NAME)
        greater_is_better = custom_metric_dict.get(PipelineMetric._KEY_METRIC_GREATER_IS_BETTER)
        custom_function = custom_metric_dict.get(PipelineMetric._KEY_CUSTOM_FUNCTION)
        if (metric_name, greater_is_better, custom_function).count(None) > 0:
            raise MlflowException(
                f"Invalid custom metric definition: {custom_metric_dict}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return cls(
            name=metric_name, greater_is_better=greater_is_better, custom_function=custom_function
        )


BUILTIN_CLASSIFICATION_PIPELINE_METRICS = [
    PipelineMetric(name="true_negatives", greater_is_better=True),
    PipelineMetric(name="false_positives", greater_is_better=False),
    PipelineMetric(name="false_negatives", greater_is_better=False),
    PipelineMetric(name="true_positives", greater_is_better=True),
    PipelineMetric(name="recall", greater_is_better=True),
    PipelineMetric(name="precision", greater_is_better=True),
    PipelineMetric(name="f1_score", greater_is_better=True),
    PipelineMetric(name="accuracy_score", greater_is_better=True),
    PipelineMetric(name="log_loss", greater_is_better=False),
    PipelineMetric(name="roc_auc", greater_is_better=True),
    PipelineMetric(name="precision_recall_auc", greater_is_better=True),
]

BUILTIN_REGRESSION_PIPELINE_METRICS = [
    PipelineMetric(name="mean_absolute_error", greater_is_better=False),
    PipelineMetric(name="mean_squared_error", greater_is_better=False),
    PipelineMetric(name="root_mean_squared_error", greater_is_better=False),
    PipelineMetric(name="max_error", greater_is_better=False),
    PipelineMetric(name="mean_absolute_percentage_error", greater_is_better=False),
]


def _get_error_fn(tmpl: str):
    """
    :param tmpl: The template kind, e.g. `regression/v1`.
    :return: The error function for the provided template.
    """
    if tmpl == "regression/v1":
        return lambda predictions, targets: predictions - targets
    raise MlflowException(
        f"No error function for template kind {tmpl}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _get_model_type_from_template(tmpl: str) -> str:
    """
    :param tmpl: The template kind, e.g. `regression/v1`.
    :return: A model type literal compatible with the mlflow evaluation service, e.g. regressor.
    """
    if tmpl == "regression/v1":
        return "regressor"
    raise MlflowException(
        f"No model type for template kind {tmpl}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _get_builtin_metrics(tmpl: str) -> str:
    """
    :param tmpl: The template kind, e.g. `regression/v1`.
    :return: The builtin metrics for the mlflow evaluation service for the model type for
    this template.
    """
    if tmpl == "regression/v1":
        return BUILTIN_REGRESSION_PIPELINE_METRICS
    elif tmpl == "classification/v1":
        return BUILTIN_CLASSIFICATION_PIPELINE_METRICS
    raise MlflowException(
        f"No builtin metrics for template kind {tmpl}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _get_custom_metrics(step_config: Dict) -> List[Dict]:
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
    builtin_metric_names = {
        metric.name for metric in _get_builtin_metrics(step_config.get("template_name"))
    }
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
