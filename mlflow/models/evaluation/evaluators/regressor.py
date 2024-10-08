from typing import List, Optional

import numpy as np
from sklearn import metrics as sk_metrics

from mlflow.models.evaluation.base import _ModelType, EvaluationMetric, EvaluationResult
from mlflow.models.evaluation.default_evaluator import BuiltInEvaluator, _extract_predict_fn, _get_aggregate_metrics_values



class RegressorEvaluator(BuiltInEvaluator):
    name = "regressor"


    def can_evaluate(self, *, model_type, evaluator_config, **kwargs):
        return model_type == _ModelType.REGRESSOR

    def _evaluate(
        self,
        model: Optional["mlflow.pyfunc.PyFuncModel"],
        extra_metrics: List[EvaluationMetric],
        custom_artifacts = None,
        **kwargs,
    ) -> EvaluationResult:
        y_true = self.dataset.labels_data
        sample_weights = self.evaluator_config.get("sample_weights", None)

        input_df = self.X.copy_to_avoid_mutation()
        y_pred = self._generate_model_predictions(model, input_df)
        self._compute_buildin_metrics(model, y_true, y_pred, sample_weights)

        self.evaluate_metrics(extra_metrics, prediction=y_pred, target=y_true)
        self.evaluate_and_log_custom_artifacts(custom_artifacts, prediction=y_pred, target=y_true)

        self.log_metrics()
        self.log_eval_table(y_pred)

        return EvaluationResult(
            metrics=self.aggregate_metrics, artifacts=self.artifacts, run_id=self.run_id
        )

    def _generate_model_predictions(self, model, input_df):
        predict_fn = _extract_predict_fn(model)
        # Regressor model should output single column of predictions
        if model is not None:
            y_pred = predict_fn(input_df)
        else:
            y_pred = self.dataset.predictions_data
        return y_pred

    def _compute_buildin_metrics(self, model, y_true, y_pred, sample_weights):
        self._evaluate_sklearn_model_score_if_scorable(model, y_true, sample_weights)

        self.metrics_values.update(
            _get_aggregate_metrics_values(
                _get_regressor_metrics(y_true, y_pred, sample_weights)
            )
        )


def _get_regressor_metrics(y, y_pred, sample_weights):
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
        "root_mean_squared_error": sk_metrics.mean_squared_error(
            y, y_pred, sample_weight=sample_weights, squared=False
        ),
        "sum_on_target": sum_on_target,
        "mean_on_target": sum_on_target / len(y),
        "r2_score": sk_metrics.r2_score(y, y_pred, sample_weight=sample_weights),
        "max_error": sk_metrics.max_error(y, y_pred),
        "mean_absolute_percentage_error": sk_metrics.mean_absolute_percentage_error(
            y, y_pred, sample_weight=sample_weights
        ),
    }