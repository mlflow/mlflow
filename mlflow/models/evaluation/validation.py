class MetricThreshold:
    def __init__(
        self,
        threshold=None,
        min_absolute_change=None,
        min_relative_change=None,
        higher_is_better=None,
    ):
        self.threshold = threshold
        self.min_absolute_change = min_absolute_change
        self.min_relative_change = min_relative_change
        self.higher_is_better = higher_is_better
        if not self.is_empty():
            if self.higher_is_better is None or not isinstance(self.higher_is_better, bool):
                raise ValueError(
                    "The higher_is_better argument must be present and \
                    a bool indicating whether higher value is preferred"
                )

    def is_empty(self):
        return (
            self.threshold is None
            and self.min_absolute_change is None
            and self.min_relative_change is None
        )

    def __str__(self):
        result_strs = []
        if self.threshold is not None:
            result_strs.append(f"Threshold: {self.threshold}.")
        if self.min_absolute_change is not None:
            result_strs.append(f"Minimum Absolute Change: {self.min_absolute_change}.")
        if self.min_relative_change is not None:
            result_strs.append(f"Minimum Relative Change: {self.min_relative_change}.")
        if self.higher_is_better is not None:
            if self.higher_is_better:
                result_strs.append("Higher value is better.")
            else:
                result_strs.append("Lower value is better.")
        return " ".join(result_strs)


class MetricValidationResult:
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


class ModelValidator:
    def __init__(self, validation_thresholds, candidate_metrics, baseline_metrics=None):
        self.validation_thresholds = validation_thresholds
        self.candidate_metrics = candidate_metrics
        self.baseline_metrics = baseline_metrics
        self.validation_result = {
            metric_name: MetricValidationResult(metric_name, threshold)
            for (metric_name, threshold) in validation_thresholds
        }

    def validate(self):
        for metric_name in self.validation_thresholds.keys():
            metric_threshold, validation_result = (
                self.validation_thresholds[metric_name],
                self.validation_result[metric_name],
            )
            candidate_metric_value, baseline_metric_value = (
                self.candidate_metrics[metric_name],
                self.baseline_metrics[metric_name],
            )
            if metric_name not in self.candidate_metrics:
                validation_result.missing = True
            if metric_threshold.higher_is_better:
                if metric_threshold.threshold is not None:
                    validation_result.threshold_failed = (
                        candidate_metric_value < metric_threshold.threshold
                    )
                if not self.baseline_metrics or metric_name not in self.baseline_metrics:
                    pass
                if metric_threshold.min_absolute_change is not None:
                    validation_result.min_absolute_change_failed = (
                        candidate_metric_value
                        < baseline_metric_value + metric_threshold.min_absolute_change
                    )
                if metric_threshold.min_relative_change is not None:
                    validation_result.min_relative_change_failed = (
                        candidate_metric_value - baseline_metric_value
                    ) / baseline_metric_value < metric_threshold.min_relative_change_failed
            else:
                if metric_threshold.threshold is not None:
                    validation_result.threshold_failed = (
                        candidate_metric_value > metric_threshold.threshold
                    )
                if not self.baseline_metrics or metric_name not in self.baseline_metrics:
                    pass
                if metric_threshold.min_absolute_change is not None:
                    validation_result.min_absolute_change_failed = (
                        candidate_metric_value
                        > baseline_metric_value + metric_threshold.min_absolute_change
                    )
                if metric_threshold.min_relative_change is not None:
                    validation_result.min_relative_change_failed = candidate_metric_value
