import logging
import importlib
from typing import Dict, Any, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

import mlflow
from mlflow import MlflowException
from mlflow.models import EvaluationMetric
from mlflow.models.evaluation.default_evaluator import (
    _get_regressor_metrics,
    _get_binary_classifier_metrics,
)
from mlflow.recipes.utils.metrics import RecipeMetric, _load_custom_metrics

_logger = logging.getLogger(__name__)

_AUTOML_DEFAULT_TIME_BUDGET = 600
_MLFLOW_TO_FLAML_METRICS = {
    "mean_absolute_error": "mae",
    "mean_squared_error": "mse",
    "root_mean_squared_error": "rmse",
    "r2_score": "r2",
    "mean_absolute_percentage_error": "mape",
    "f1_score": "f1",
    "f1_score_micro": "micro_f1",
    "f1_score_macro": "macro_f1",
    "accuracy_score": "accuracy",
    "roc_auc": "roc_auc",
    "roc_auc_ovr": "roc_auc_ovr",
    "roc_auc_ovo": "roc_auc_ovo",
    "log_loss": "log_loss",
}

# metrics that are not supported natively in FLAML
_SKLEARN_METRICS = ["recall_score", "precision_score"]


def get_estimator_and_best_params(
    X,
    y,
    task: str,
    extended_task: str,
    step_config: Dict[str, Any],
    recipe_root: str,
    evaluation_metrics: Dict[str, RecipeMetric],
    primary_metric: str,
) -> Tuple["BaseEstimator", Dict[str, Any]]:
    return _create_model_automl(
        X, y, task, extended_task, step_config, recipe_root, evaluation_metrics, primary_metric
    )


def _create_custom_metric_flaml(
    task: str, metric_name: str, coeff: int, eval_metric: EvaluationMetric
) -> callable:
    def calc_metric(X, y, estimator) -> Dict[str, float]:
        y_pred = estimator.predict(X)
        builtin_metrics = (
            _get_regressor_metrics(y, y_pred, sample_weights=None)
            if task == "regression"
            else _get_binary_classifier_metrics(y_true=y, y_pred=y_pred)
        )
        res_df = pd.DataFrame()
        res_df["prediction"] = y_pred
        res_df["target"] = y if task == "classification" else y.values
        return eval_metric.eval_fn(res_df, builtin_metrics)

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
        val_metric = coeff * calc_metric(X_val, y_val, estimator)
        train_metric = calc_metric(X_train, y_train, estimator)
        main_metric = coeff * val_metric
        return main_metric, {
            f"{metric_name}_train": train_metric,
            f"{metric_name}_val": val_metric,
        }

    return custom_metric


def _create_sklearn_metric_flaml(metric_name: str, coeff: int, avg: str = "binary") -> callable:
    # pylint: disable=keyword-arg-before-vararg
    # pylint: disable=unused-argument
    def sklearn_metric(
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
        custom_metrics_mod = importlib.import_module("sklearn.metrics")
        eval_fn = getattr(custom_metrics_mod, metric_name)
        val_metric = coeff * eval_fn(y_val, estimator.predict(X_val), average=avg)
        train_metric = coeff * eval_fn(y_train, estimator.predict(X_train), average=avg)
        return val_metric, {
            f"{metric_name}_train": train_metric,
            f"{metric_name}_val": val_metric,
        }

    return sklearn_metric


def _create_model_automl(
    X,
    y,
    task: str,
    extended_task: str,
    step_config: Dict[str, Any],
    recipe_root: str,
    evaluation_metrics: Dict[str, RecipeMetric],
    primary_metric: str,
) -> Tuple["BaseEstimator", Dict[str, Any]]:
    try:
        from flaml import AutoML
    except ImportError:
        raise MlflowException("Please install FLAML to use AutoML!")

    try:
        if primary_metric in _MLFLOW_TO_FLAML_METRICS and primary_metric in evaluation_metrics:
            metric = _MLFLOW_TO_FLAML_METRICS[primary_metric]
            if primary_metric == "roc_auc" and extended_task == "classification/multiclass":
                metric = "roc_auc_ovr"
        elif primary_metric in _SKLEARN_METRICS and primary_metric in evaluation_metrics:
            metric = _create_sklearn_metric_flaml(
                primary_metric,
                -1 if evaluation_metrics[primary_metric].greater_is_better else 1,
                "macro" if extended_task in ["classification/multiclass"] else "binary",
            )
        elif primary_metric in evaluation_metrics:
            metric = _create_custom_metric_flaml(
                task,
                primary_metric,
                -1 if evaluation_metrics[primary_metric].greater_is_better else 1,
                _load_custom_metrics(recipe_root, [evaluation_metrics[primary_metric]])[0],
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
        if automl.model is None:
            raise MlflowException(
                "AutoML (FLAML) could not train a suitable algorithm. "
                "Maybe you should increase `time_budget_secs`parameter "
                "to give AutoML process more time."
            )
        return automl.model.estimator, automl.best_config
    except Exception as e:
        _logger.warning(e, exc_info=e, stack_info=True)
        raise MlflowException(
            f"Error has occurred during training of AutoML model using FLAML: {repr(e)}"
        )
