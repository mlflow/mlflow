from abc import ABC, abstractmethod

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

from typing import Optional


class PipelineHelper(ABC):
    @abstractmethod
    def error_fn(self):
        """
        :return: The error function for the template type.
        """
        pass

    @abstractmethod
    def builtin_metrics(self):
        """
        :return: The builtin metrics for the mlflow evaluation service for the template type.
        """
        pass

    @abstractmethod
    def model_type(self):
        """
        :return: A model type literal compatible with the mlflow evaluation service."""
        pass

    @classmethod
    def from_template(cls, tmpl):
        """
        Factory method for initializing pipeline helpers based on the template type.

        :return: A `PipelineHelper` instance for the given template type.
        """
        if tmpl == "regression/v1":
            return RegressionHelper()
        if tmpl == "classification/v1":
            return ClassificationHelper()
        raise MlflowException(
            f"No such for template {tmpl}",
            error_code=INVALID_PARAMETER_VALUE,
        )


class RegressionHelper(PipelineHelper):
    def error_fn(self):
        return lambda predictions, targets: predictions - targets

    def builtin_metrics(self):
        return [
            PipelineMetric(name="mean_absolute_error", greater_is_better=False),
            PipelineMetric(name="mean_squared_error", greater_is_better=False),
            PipelineMetric(name="root_mean_squared_error", greater_is_better=False),
            PipelineMetric(name="max_error", greater_is_better=False),
            PipelineMetric(name="mean_absolute_percentage_error", greater_is_better=False),
        ]

    def model_type(self):
        return "regressor"


class ClassificationHelper(PipelineHelper):
    def error_fn(self):
        return lambda predictions, targets: predictions != targets

    def builtin_metrics(self):
        return [
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

    def model_type(self):
        return "classifier"


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
