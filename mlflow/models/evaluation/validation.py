from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST


class MetricThreshold:
    """
    This class allows you to define metric thresholds for model validation.
    Allowed thresholds are: threshold, min_absolute_change, min_relative_change.

    :param threshold: A floating number representing the value threshold for the metric.

                      - If higher is better for the metric, the metric value has to be
                        >= threshold to pass validation.
                      - Otherwise, the metric value has to be <= threshold to pass the validation.

    :param min_absolute_change: A floating point number representing the minimum absolute change
                                required for candidate model to pass the comparison with
                                the baseline model.

                                - If higher is better for the metric, metric value has to be
                                  >= baseline model metric value + min_absolute_change
                                  to pass the validation.
                                - Otherwise, metric value has to be
                                  <= baseline model metric value + min_absolute_change
                                  to pass the validation.

    :param min_relative_change: A floating point number between 0 and 1 representing
                                the minimum relative change (in percentage of
                                baseline model metric value) for candidate model
                                to pass the comparison with the baseline model.

                                - If higher is better for the metric, metric value has to be
                                  >= baseline model metric value * (1 + min_relative_change)
                                - Otherwise, metric value has to be
                                  <= baseline model metric value * (1 - min_relative_change)

    :param higher_is_better: A boolean representing whether higher value is better for the metric.
    """

    def __init__(
        self,
        threshold=None,
        min_absolute_change=None,
        min_relative_change=None,
        higher_is_better=None,
    ):
        self._threshold = threshold
        self._min_absolute_change = min_absolute_change
        self._min_relative_change = min_relative_change
        self._higher_is_better = higher_is_better
        if self._min_relative_change is not None and (
            self._min_relative_change < 0 or self._min_relative_change > 1
        ):
            raise ValueError("The min_relative_change argument must be in [0, 1]")
        if self._higher_is_better is None or not isinstance(self._higher_is_better, bool):
            raise ValueError(
                "The higher_is_better argument must be present and \
                a bool indicating whether higher value is preferred"
            )

    @property
    def threshold(self):
        """
        Value threshold.
        :rtype: float
        """
        return self._threshold

    @property
    def min_absolute_change(self):
        """
        Minimum absolute change required to pass model comparison with baseline model
        :rtype: float
        """
        return self._min_absolute_change

    @property
    def min_relative_change(self):
        """
        Minimum relative change required to pass model comparison wwith baseline model
        :rtype: float
        """
        return self._min_relative_change

    @property
    def higher_is_better(self):
        """
        Whether higher value is better for the metric.
        :rtype: bool
        """
        return self._higher_is_better

    def is_empty(self):
        """
        Return True if there is no threshold specified, False otherwise.
        """
        return (
            self._threshold is None
            and self._min_absolute_change is None
            and self._min_relative_change is None
        )

    def __str__(self):
        """
        Return a human-readable string consisting of all specified thresholds.
        """
        threshold_strs = []
        if self._threshold is not None:
            threshold_strs.append(f"Threshold: {self._threshold}.")
        if self._min_absolute_change is not None:
            threshold_strs.append(f"Minimum Absolute Change: {self._min_absolute_change}.")
        if self._min_relative_change is not None:
            threshold_strs.append(f"Minimum Relative Change: {self._min_relative_change}.")
        if self._higher_is_better is not None:
            if self._higher_is_better:
                threshold_strs.append("Higher value is better.")
            else:
                threshold_strs.append("Lower value is better.")
        return " ".join(threshold_strs)


class _MetricValidationResult:
    """
    Internal class for representing validation result per metric.
    Not user facing, used for organizing metric failures and generating failure message
    more conveniently.
    :param metric_name: String representing the metric name
    :param metric_threshold: :py:class: `MetricThreshold<mlflow.models.validation.MetricThreshold>`
                             The MetricThreshold for the metric.
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
        Return a human-readable string representing the validation result for the metric.
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
