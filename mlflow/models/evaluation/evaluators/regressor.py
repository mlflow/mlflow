from typing import Optional

import numpy as np
from sklearn import metrics as sk_metrics

import mlflow
from mlflow.models.evaluation.base import EvaluationMetric, EvaluationResult, _ModelType
from mlflow.models.evaluation.default_evaluator import (
    BuiltInEvaluator,
    _extract_output_and_other_columns,
    _extract_predict_fn,
    _get_aggregate_metrics_values,
)


class RegressorEvaluator(BuiltInEvaluator):
    """
    A built-in evaluator for regressor models.
    """

    name = "regressor"

    @classmethod
    def can_evaluate(cls, *, model_type, evaluator_config, **kwargs):
        return model_type == _ModelType.REGRESSOR

    def _evaluate(
        self,
        model: Optional["mlflow.pyfunc.PyFuncModel"],
        extra_metrics: list[EvaluationMetric],
        custom_artifacts=None,
        **kwargs,
    ) -> Optional[EvaluationResult]:
        self.y_true = self.dataset.labels_data
        self.sample_weights = self.evaluator_config.get("sample_weights", None)

        input_df = self.X.copy_to_avoid_mutation()
        self.y_pred = self._generate_model_predictions(model, input_df)
        self._compute_buildin_metrics(model)

        self.evaluate_metrics(extra_metrics, prediction=self.y_pred, target=self.y_true)
        self.evaluate_and_log_custom_artifacts(
            custom_artifacts, prediction=self.y_pred, target=self.y_true
        )

        self.log_metrics()
        self.log_eval_table(self.y_pred)

        return EvaluationResult(
            metrics=self.aggregate_metrics, artifacts=self.artifacts, run_id=self.run_id
        )

    def _generate_model_predictions(self, model, input_df):
        if predict_fn := _extract_predict_fn(model):
            preds = predict_fn(input_df)
            y_pred, _, _ = _extract_output_and_other_columns(preds, self.predictions)
            return y_pred
        else:
            return self.dataset.predictions_data

    def _compute_buildin_metrics(self, model):
        self._evaluate_sklearn_model_score_if_scorable(model, self.y_true, self.sample_weights)
        self.metrics_values.update(
            _get_aggregate_metrics_values(
                _get_regressor_metrics(self.y_true, self.y_pred, self.sample_weights)
            )
        )


def _get_regressor_metrics(y, y_pred, sample_weights):
    from mlflow.metrics.metric_definitions import _root_mean_squared_error

    sum_on_target = (
        (np.array(y) * np.array(sample_weights)).sum() if sample_weights is not None else sum(y)
    )
    return {
        "example_count": len(y),
        "mean_absolute_error": sk_metrics.mean_absolute_error(
            y, y_pred, sample_weight=sample_weights
        ),
        "mean_squared_error": sk_metrics.mean_squared_error(
            y, y_pred, sample_weight=sample_weights
        ),
        "root_mean_squared_error": _root_mean_squared_error(
            y_true=y,
            y_pred=y_pred,
            sample_weight=sample_weights,
        ),
        "sum_on_target": sum_on_target,
        "mean_on_target": sum_on_target / len(y),
        "r2_score": sk_metrics.r2_score(y, y_pred, sample_weight=sample_weights),
        "max_error": sk_metrics.max_error(y, y_pred),
        "mean_absolute_percentage_error": sk_metrics.mean_absolute_percentage_error(
            y, y_pred, sample_weight=sample_weights
        ),
    }
