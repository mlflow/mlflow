import logging
import importlib
import sys
from typing import List, Dict, Optional, Any

from mlflow.models import EvaluationMetric, make_metric
from mlflow.exceptions import MlflowException, BAD_REQUEST
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)


class RecipeMetric:
    _KEY_METRIC_NAME = "name"
    _KEY_METRIC_GREATER_IS_BETTER = "greater_is_better"
    _KEY_CUSTOM_FUNCTION = "function"

    def __init__(self, name: str, greater_is_better: bool, custom_function: Optional[str] = None):
        self.name = name
        self.greater_is_better = greater_is_better
        self.custom_function = custom_function

    @classmethod
    def from_custom_metric_dict(cls, custom_metric_dict):
        metric_name = custom_metric_dict.get(RecipeMetric._KEY_METRIC_NAME)
        greater_is_better = custom_metric_dict.get(RecipeMetric._KEY_METRIC_GREATER_IS_BETTER)
        custom_function = custom_metric_dict.get(RecipeMetric._KEY_CUSTOM_FUNCTION)
        if (metric_name, greater_is_better, custom_function).count(None) > 0:
            raise MlflowException(
                f"Invalid custom metric definition: {custom_metric_dict}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return cls(
            name=metric_name, greater_is_better=greater_is_better, custom_function=custom_function
        )


BUILTIN_BINARY_CLASSIFICATION_RECIPE_METRICS = [
    RecipeMetric(name="true_negatives", greater_is_better=True),
    RecipeMetric(name="false_positives", greater_is_better=False),
    RecipeMetric(name="false_negatives", greater_is_better=False),
    RecipeMetric(name="true_positives", greater_is_better=True),
    RecipeMetric(name="recall_score", greater_is_better=True),
    RecipeMetric(name="precision_score", greater_is_better=True),
    RecipeMetric(name="f1_score", greater_is_better=True),
    RecipeMetric(name="accuracy_score", greater_is_better=True),
    RecipeMetric(name="roc_auc", greater_is_better=True),
    RecipeMetric(name="log_loss", greater_is_better=False),
]

BUILTIN_MULTICLASS_CLASSIFICATION_RECIPE_METRICS = [
    RecipeMetric(name="recall_score", greater_is_better=True),
    RecipeMetric(name="precision_score", greater_is_better=True),
    RecipeMetric(name="f1_score_macro", greater_is_better=True),
    RecipeMetric(name="f1_score_micro", greater_is_better=True),
    RecipeMetric(name="accuracy_score", greater_is_better=True),
    RecipeMetric(name="roc_auc", greater_is_better=True),
    RecipeMetric(name="log_loss", greater_is_better=False),
]

BUILTIN_REGRESSION_RECIPE_METRICS = [
    RecipeMetric(name="mean_absolute_error", greater_is_better=False),
    RecipeMetric(name="mean_squared_error", greater_is_better=False),
    RecipeMetric(name="root_mean_squared_error", greater_is_better=False),
    RecipeMetric(name="max_error", greater_is_better=False),
    RecipeMetric(name="mean_absolute_percentage_error", greater_is_better=False),
]

DEFAULT_METRICS = {
    "regression": "root_mean_squared_error",
    "classification/binary": "f1_score",
    "classification/multiclass": "f1_score_macro",
}


def _get_error_fn(tmpl: str, use_probability: bool = False, positive_class: Optional[str] = None):
    """
    :param tmpl: The template kind, e.g. `regression/v1`.
    :return: The error function for the provided template.
    """
    if tmpl == "regression/v1":
        return lambda predictions, targets: predictions - targets
    if tmpl == "classification/v1":
        if use_probability:
            # It computes error rate for binary classification since
            # positive class only exists in binary classification.
            def error_rate(true_label, predicted_positive_class_proba):
                if true_label == positive_class:
                    # if true_label == positive_class then the probability is
                    # predicted_positive_class_proba but the error rate is
                    # 1 - predicted_positive_class_proba
                    return 1 - predicted_positive_class_proba
                else:
                    # if true_label != positive_class then the probability is
                    # 1 - predicted_positive_class_proba but the error rate is
                    # predicted_positive_class_proba
                    return predicted_positive_class_proba

            return lambda predictions, targets: [
                error_rate(x, y) for (x, y) in zip(targets, predictions)
            ]
        else:
            return lambda predictions, targets: predictions != targets
    raise MlflowException(
        f"No error function for template kind {tmpl}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _get_extended_task(recipe: str, positive_class: str) -> str:
    """
    :param step_config: Step config
    :return: Extended type string. Currently supported types are: "regression",
    "binary_classification", "multiclass_classification"
    """
    if "regression" in recipe:
        return "regression"
    elif "classification" in recipe:
        if positive_class is not None:
            return "classification/binary"
        else:
            return "classification/multiclass"
    raise MlflowException(
        f"No model type for template kind {recipe}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _get_model_type_from_template(tmpl: str) -> str:
    """
    :param tmpl: The template kind, e.g. `regression/v1`.
    :return: A model type literal compatible with the mlflow evaluation service, e.g. regressor.
    """
    if tmpl == "regression/v1":
        return "regressor"
    if tmpl == "classification/v1":
        return "classifier"
    raise MlflowException(
        f"No model type for template kind {tmpl}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _get_builtin_metrics(ext_task: str) -> Dict[str, str]:
    """
    :param tmpl: The template kind, e.g. `regression/v1`.
    :return: The builtin metrics for the mlflow evaluation service for the model type for
    this template.
    """
    if ext_task == "regression":
        return BUILTIN_REGRESSION_RECIPE_METRICS
    elif ext_task == "classification/binary":
        return BUILTIN_BINARY_CLASSIFICATION_RECIPE_METRICS
    elif ext_task == "classification/multiclass":
        return BUILTIN_MULTICLASS_CLASSIFICATION_RECIPE_METRICS
    raise MlflowException(
        f"No builtin metrics for template kind {ext_task}",
        error_code=INVALID_PARAMETER_VALUE,
    )


def transform_multiclass_metric(metric_name: str, ext_task: str) -> str:
    if ext_task == "classification/multiclass":
        for m in BUILTIN_MULTICLASS_CLASSIFICATION_RECIPE_METRICS:
            if metric_name in m.name:
                return m.name
    return metric_name


def transform_multiclass_metrics_dict(eval_metrics: Dict[str, Any], ext_task) -> Dict[str, Any]:
    return {transform_multiclass_metric(k, ext_task): v for k, v in eval_metrics.items()}


def _get_custom_metrics(step_config: Dict, ext_task: str) -> List[Dict]:
    """
    :param: Configuration dictionary for the train or evaluate step.
    :return: A list of custom metrics defined in the specified configuration dictionary,
             or an empty list of the configuration dictionary does not define any custom metrics.
    """
    custom_metric_dicts = step_config.get("custom_metrics", [])
    custom_metrics = [
        RecipeMetric.from_custom_metric_dict(metric_dict) for metric_dict in custom_metric_dicts
    ]
    custom_metric_names = {metric.name for metric in custom_metrics}
    builtin_metric_names = {metric.name for metric in _get_builtin_metrics(ext_task)}
    overridden_builtin_metrics = custom_metric_names.intersection(builtin_metric_names)
    if overridden_builtin_metrics:
        _logger.warning(
            "Custom metrics override the following built-in metrics: %s",
            sorted(overridden_builtin_metrics),
        )
    return custom_metrics


def _load_custom_metrics(recipe_root: str, metrics: List[RecipeMetric]) -> List[EvaluationMetric]:
    custom_metrics = [metric for metric in metrics if metric.custom_function is not None]
    if not custom_metrics:
        return None

    try:
        sys.path.append(recipe_root)
        custom_metrics_mod = importlib.import_module("steps.custom_metrics")
        return [
            make_metric(
                eval_fn=getattr(custom_metrics_mod, custom_metric.custom_function),
                name=custom_metric.name,
                greater_is_better=custom_metric.greater_is_better,
            )
            for custom_metric in custom_metrics
        ]
    except Exception as e:
        raise MlflowException(
            message="Failed to load custom metric functions",
            error_code=BAD_REQUEST,
        ) from e


def _get_primary_metric(configured_metric: str, ext_task: str):
    if configured_metric is not None:
        return configured_metric
    else:
        return DEFAULT_METRICS[ext_task]
