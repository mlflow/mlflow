from __future__ import annotations

from mlflow.exceptions import MlflowException

_SUPPORTED_METRICS = {
    "MissingValueCount": "evidently.metrics.MissingValueCount",
    "UniqueValueCount": "evidently.metrics.UniqueValueCount",
}


def get_metric_class(metric_name: str):
    """Get an Evidently metric class by name.

    Args:
        metric_name: Name of the Evidently metric (e.g., "ValueDrift", "MissingValueCount")

    Returns:
        The Evidently metric class

    Raises:
        MlflowException: If the metric is not in the supported list
    """
    available = ", ".join(sorted(_SUPPORTED_METRICS))
    if metric_name not in _SUPPORTED_METRICS:
        raise MlflowException.invalid_parameter_value(
            f"Unknown Evidently metric: '{metric_name}'. Available metrics: {available}"
        )

    from evidently import metrics as evidently_metrics

    return getattr(evidently_metrics, metric_name)
