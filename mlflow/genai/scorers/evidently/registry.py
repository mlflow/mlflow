from __future__ import annotations

from mlflow.exceptions import MlflowException

_SUPPORTED_METRICS = {
    "ValueDrift": "evidently.metrics.ValueDrift",
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
        MlflowException: If the metric is not found
    """
    from evidently import metrics as evidently_metrics

    try:
        return getattr(evidently_metrics, metric_name)
    except AttributeError:
        available = ", ".join(sorted(_SUPPORTED_METRICS))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Evidently metric: '{metric_name}'. Could not find "
            f"'{metric_name}' in 'evidently.metrics'. "
            f"Available pre-configured metrics: {available}"
        )
