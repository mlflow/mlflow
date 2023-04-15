from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import deprecated


class MetricThreshold:
    """
    This class allows you to define metric thresholds for model validation.
    Allowed thresholds are: threshold, min_absolute_change, min_relative_change.

    :param threshold: (Optional) A number representing the value threshold for the metric.

                      - If higher is better for the metric, the metric value has to be
                        >= threshold to pass validation.
                      - Otherwise, the metric value has to be <= threshold to pass the validation.

    :param min_absolute_change: (Optional) A positive number representing the minimum absolute
                                change required for candidate model to pass validation with
                                the baseline model.

                                - If higher is better for the metric, metric value has to be
                                  >= baseline model metric value + min_absolute_change
                                  to pass the validation.
                                - Otherwise, metric value has to be
                                  <= baseline model metric value - min_absolute_change
                                  to pass the validation.

    :param min_relative_change: (Optional) A floating point number between 0 and 1 representing
                                the minimum relative change (in percentage of
                                baseline model metric value) for candidate model
                                to pass the comparison with the baseline model.

                                - If higher is better for the metric, metric value has to be
                                  >= baseline model metric value * (1 + min_relative_change)
                                - Otherwise, metric value has to be
                                  <= baseline model metric value * (1 - min_relative_change)
                                - Note that if the baseline model metric value is equal to 0, the
                                  threshold falls back performing a simple verification that the
                                  candidate metric value is better than the baseline metric value,
                                  i.e. metric value >= baseline model metric value + 1e-10 if higher
                                  is better; metric value <= baseline model metric value - 1e-10 if
                                  lower is better.

    :param greater_is_better: A required boolean representing whether higher value is
                              better for the metric.

    :param higher_is_better:
        .. deprecated:: 2.3.0
            Use ``greater_is_better`` instead.

        A required boolean representing whether higher value is better for the metric.
    """

    def __init__(
        self,
        threshold=None,
        min_absolute_change=None,
        min_relative_change=None,
        greater_is_better=None,
        higher_is_better=None,
    ):
        if threshold is not None and not type(threshold) in {int, float}:
            raise MetricThresholdClassException("`threshold` parameter must be a number.")
        if min_absolute_change is not None and (
            not type(min_absolute_change) in {int, float} or min_absolute_change <= 0
        ):
            raise MetricThresholdClassException(
                "`min_absolute_change` parameter must be a positive number."
            )
        if min_relative_change is not None:
            if not isinstance(min_relative_change, float):
                raise MetricThresholdClassException(
                    "`min_relative_change` parameter must be a floating point number."
                )
            if min_relative_change < 0 or min_relative_change > 1:
                raise MetricThresholdClassException(
                    "`min_relative_change` parameter must be between 0 and 1."
                )
        if higher_is_better is None and greater_is_better is None:
            raise MetricThresholdClassException("`greater_is_better` parameter must be defined.")
        if higher_is_better is not None and greater_is_better is not None:
            raise MetricThresholdClassException(
                "`higher_is_better` parameter must be None when `greater_is_better` is defined."
            )
        if greater_is_better is None:
            greater_is_better = higher_is_better
        if not isinstance(greater_is_better, bool):
            raise MetricThresholdClassException("`greater_is_better` parameter must be a boolean.")
        if threshold is None and min_absolute_change is None and min_relative_change is None:
            raise MetricThresholdClassException("no threshold was specified.")
        self._threshold = threshold
        self._min_absolute_change = min_absolute_change
        self._min_relative_change = min_relative_change
        self._greater_is_better = greater_is_better

    @property
    def threshold(self):
        """
        Value of the threshold.
        """
        return self._threshold

    @property
    def min_absolute_change(self):
        """
        Value of the minimum absolute change required to pass model comparison with baseline model.
        """
        return self._min_absolute_change

    @property
    def min_relative_change(self):
        """
        Float value of the minimum relative change required to pass model comparison with
        baseline model.
        """
        return self._min_relative_change

    @property
    @deprecated("The attribute `higher_is_better` is deprecated. Use `greater_is_better` instead.")
    def higher_is_better(self):
        """
        Boolean value representing whether higher value is better for the metric.
        """
        return self._greater_is_better

    @property
    def greater_is_better(self):
        """
        Boolean value representing whether higher value is better for the metric.
        """
        return self._greater_is_better

    def __str__(self):
        """
        Returns a human-readable string consisting of all specified thresholds.
        """
        threshold_strs = []
        if self._threshold is not None:
            threshold_strs.append(f"Threshold: {self._threshold}.")
        if self._min_absolute_change is not None:
            threshold_strs.append(f"Minimum Absolute Change: {self._min_absolute_change}.")
        if self._min_relative_change is not None:
            threshold_strs.append(f"Minimum Relative Change: {self._min_relative_change}.")
        if self._greater_is_better is not None:
            if self._greater_is_better:
                threshold_strs.append("Higher value is better.")
            else:
                threshold_strs.append("Lower value is better.")
        return " ".join(threshold_strs)


class MetricThresholdClassException(MlflowException):
    def __init__(self, _message, **kwargs):
        message = "Could not instantiate MetricThreshold class: " + _message
        super().__init__(message, error_code=INVALID_PARAMETER_VALUE, **kwargs)


class _MetricValidationResult:
    """
    Internal class for representing validation result per metric.
    Not user facing, used for organizing metric failures and generating failure message
    more conveniently.
    :param metric_name: String representing the metric name
    :param candidate_metric_value: value of metric for candidate model
    :param metric_threshold: :py:class: `MetricThreshold<mlflow.models.validation.MetricThreshold>`
                             The MetricThreshold for the metric.
    :param baseline_metric_value: value of metric for baseline model
    """

    missing_candidate = False
    missing_baseline = False
    threshold_failed = False
    min_absolute_change_failed = False
    min_relative_change_failed = False

    def __init__(
        self,
        metric_name,
        candidate_metric_value,
        metric_threshold,
        baseline_metric_value=None,
    ):
        self.metric_name = metric_name
        self.candidate_metric_value = candidate_metric_value
        self.baseline_metric_value = baseline_metric_value
        self.metric_threshold = metric_threshold

    def __str__(self):
        """
        Returns a human-readable string representing the validation result for the metric.
        """
        if self.is_success():
            return f"Metric {self.metric_name} passed the validation."

        if self.missing_candidate:
            return (
                f"Metric validation failed: metric {self.metric_name} was missing from the "
                f"evaluation result of the candidate model."
            )

        result_strs = []
        if self.threshold_failed:
            result_strs.append(
                f"Metric {self.metric_name} value threshold check failed: "
                f"candidate model {self.metric_name} = {self.candidate_metric_value}, "
                f"{self.metric_name} threshold = {self.metric_threshold.threshold}."
            )
        if self.missing_baseline:
            result_strs.append(
                f"Model comparison failed: metric {self.metric_name} was missing from "
                f"the evaluation result of the baseline model."
            )
        else:
            if self.min_absolute_change_failed:
                result_strs.append(
                    f"Metric {self.metric_name} minimum absolute change check failed: "
                    f"candidate model {self.metric_name} = {self.candidate_metric_value}, "
                    f"baseline model {self.metric_name} = {self.baseline_metric_value}, "
                    f"{self.metric_name} minimum absolute change threshold = "
                    f"{self.metric_threshold.min_absolute_change}."
                )
            if self.min_relative_change_failed:
                result_strs.append(
                    f"Metric {self.metric_name} minimum relative change check failed: "
                    f"candidate model {self.metric_name} = {self.candidate_metric_value}, "
                    f"baseline model {self.metric_name} = {self.baseline_metric_value}, "
                    f"{self.metric_name} minimum relative change threshold = "
                    f"{self.metric_threshold.min_relative_change}."
                )
        return " ".join(result_strs)

    def is_success(self):
        return (
            not self.missing_candidate
            and not self.missing_baseline
            and not self.threshold_failed
            and not self.min_absolute_change_failed
            and not self.min_relative_change_failed
        )


class ModelValidationFailedException(MlflowException):
    def __init__(self, message, **kwargs):
        super().__init__(message, error_code=BAD_REQUEST, **kwargs)
