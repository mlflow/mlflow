class MetricThreshold:
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
        if self._min_relative_change < 0 or self._min_relative_change > 1:
            raise ValueError("The min_relative_change argument must be in (0, 1)")
        if not self.is_empty():
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
        return self.threshold

    @property
    def min_absolute_change(self):
        """
        Minimum absolute change required to pass model comparsion with baseline model
        """

    def is_empty(self):
        """
        Return True if there is no threshold specified, False otherwise.
        """
        return (
            self.threshold is None
            and self.min_absolute_change is None
            and self.min_relative_change is None
        )

    def __str__(self):
        """
        Return a string consisting of all sepcified thresholds.
        """
        threshold_strs = []
        if self.threshold is not None:
            threshold_strs.append(f"Threshold: {self.threshold}.")
        if self.min_absolute_change is not None:
            threshold_strs.append(f"Minimum Absolute Change: {self.min_absolute_change}.")
        if self.min_relative_change is not None:
            threshold_strs.append(f"Minimum Relative Change: {self.min_relative_change}.")
        if self.higher_is_better is not None:
            if self.higher_is_better:
                threshold_strs.append("Higher value is better.")
            else:
                threshold_strs.append("Lower value is better.")
        return " ".join(threshold_strs)


class MetricValidationResult:
    """
    ValidationResult per metric, not user facing, used internally for generating validation
    failure message.
    """

    missing = False
    threshold_failed = False
    min_absolute_change_failed = False
    min_relative_change_failed = False

    def __init__(self, metric_name, threshold):
        self.metric_name = metric_name
        self.threshold = threshold

    def __str__(self):
        if self.missing:
            return f"Metric {self.metric_name} is missing from the evaluation result"
        if self.min_absolute_change_failed and self.min_relative_change_failed:
            return "Threshold is not met"

    def is_success(self):
        return (
            not self.missing
            and not self.threshold_failed
            and not self.min_absolute_change_failed
            and not self.min_relative_change_failed
        )
