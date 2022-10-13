import logging
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.base import BaseEstimator

import mlflow
from mlflow import MlflowException
from mlflow.models.evaluation.default_evaluator import _get_regressor_metrics
from mlflow.pipelines.utils.metrics import PipelineMetric, _load_custom_metric_functions

_logger = logging.getLogger(__name__)

_AUTOML_DEFAULT_TIME_BUDGET = 10
_MLFLOW_TO_FLAML_METRICS = {
    "mean_absolute_error": "mae",
    "mean_squared_error": "mse",
    "root_mean_squared_error": "rmse",
    "r2_score": "r2",
    "mean_absolute_percentage_error": "mape",
}


def get_estimator_and_best_params(
    X,
    y,
    task: str,
    step_config: Dict[str, Any],
    pipeline_root: str,
    evaluation_metrics: Dict[str, PipelineMetric],
    primary_metric: str,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    return _create_model_automl(
        X, y, task, step_config, pipeline_root, evaluation_metrics, primary_metric
    )


def _create_custom_metric_flaml(metric_name: str, coeff: int, custom_func: callable) -> callable:
    def add_suffix(metrics: Dict[str, float], suffix: str) -> Dict[str, float]:
        return {f"{k}_{suffix}": v for k, v in metrics.items()}

    def calc_metric(X, y, estimator) -> Dict[str, float]:
        y_pred = estimator.predict(X)
        builtin_metrics = _get_regressor_metrics(y, y_pred, sample_weights=None)
        res_df = pd.DataFrame()
        res_df["prediction"] = y_pred
        res_df["target"] = y.values
        return custom_func(res_df, builtin_metrics)

    # pylint: disable=keyword-arg-before-vararg
    # pylint: disable=unused-argument
    def custom_metric(
        X_val,
        y_val,
        estimator,
        labels,
        X_train,
        y_train,
        weight_val=None,
        weight_train=None,
        *args,
    ):
        val_metrics = calc_metric(X_val, y_val, estimator)
        if metric_name not in val_metrics:
            raise MlflowException(
                f"User-defined function has not calculated expected primary metric {metric_name}.\n"
                f"The function has returned the following metrics: {val_metrics}"
            )
        main_metric = coeff * val_metrics[metric_name]
        val_metrics = add_suffix(val_metrics, "val")
        train_metrics = add_suffix(calc_metric(X_train, y_train, estimator), "train")
        return main_metric, {**val_metrics, **train_metrics}

    return custom_metric


def _create_model_automl(
    X,
    y,
    task: str,
    step_config: Dict[str, Any],
    pipeline_root: str,
    evaluation_metrics: Dict[str, PipelineMetric],
    primary_metric: str,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    try:
        from flaml import AutoML
    except ImportError:
        raise MlflowException("Please install FLAML to use AutoML!")

    try:
        if primary_metric in _MLFLOW_TO_FLAML_METRICS and primary_metric in evaluation_metrics:
            metric = _MLFLOW_TO_FLAML_METRICS[primary_metric]
        elif primary_metric in evaluation_metrics:
            metric = _create_custom_metric_flaml(
                primary_metric,
                -1 if evaluation_metrics[primary_metric].greater_is_better else 1,
                _load_custom_metric_functions(pipeline_root, [evaluation_metrics[primary_metric]])[
                    0
                ],
            )
        else:
            raise MlflowException(
                f"There is no FLAML alternative or custom metric for {primary_metric} metric."
            )

        automl_settings = step_config.get("flaml_params", {})
        automl_settings["time_budget"] = step_config.get(
            "time_budget_secs", _AUTOML_DEFAULT_TIME_BUDGET
        )
        automl_settings["metric"] = metric
        automl_settings["task"] = task
        # Disabled Autologging, because during the hyperparameter search
        # it tries to log the same parameters multiple times.
        mlflow.autolog(disable=True)
        automl = AutoML()
        automl.fit(X, y, **automl_settings)
        mlflow.autolog(disable=False, log_models=False)
        return automl.model.estimator, automl.best_config
    except Exception as e:
        raise MlflowException(
            f"Error has occurred during training of AutoML model using FLAML: {repr(e)}"
        )
