import copy
import functools
import inspect
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import namedtuple
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline as sk_Pipeline

import mlflow
from mlflow import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
    EvaluationMetric,
    MetricValue,
    ari_grade_level,
    exact_match,
    flesch_kincaid_grade_level,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge1,
    rouge2,
    rougeL,
    rougeLsum,
    token_count,
    toxicity,
)
from mlflow.models.evaluation.artifacts import (
    CsvEvaluationArtifact,
    ImageEvaluationArtifact,
    JsonEvaluationArtifact,
    NumpyEvaluationArtifact,
    _infer_artifact_type_and_ext,
)
from mlflow.models.evaluation.base import (
    EvaluationResult,
    ModelEvaluator,
    _ModelType,
)
from mlflow.models.utils import plot_lines
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.sklearn import _SklearnModelWrapper
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_ROWS_FOR_SHAP = 2000
_EVAL_TABLE_FILE_NAME = "eval_results_table.json"
_TOKEN_COUNT_METRIC_NAME = "token_count"
_LATENCY_METRIC_NAME = "latency"


def _is_categorical(values):
    """
    Infer whether input values are categorical on best effort.
    Return True represent they are categorical, return False represent we cannot determine result.
    """
    dtype_name = pd.Series(values).convert_dtypes().dtype.name.lower()
    return dtype_name in ["category", "string", "boolean"]


def _is_continuous(values):
    """
    Infer whether input values is continuous on best effort.
    Return True represent they are continous, return False represent we cannot determine result.
    """
    dtype_name = pd.Series(values).convert_dtypes().dtype.name.lower()
    return dtype_name.startswith("float")


def _infer_model_type_by_labels(labels):
    """
    Infer model type by target values.
    """
    if _is_categorical(labels):
        return _ModelType.CLASSIFIER
    elif _is_continuous(labels):
        return _ModelType.REGRESSOR
    else:
        return None  # Unknown


def _extract_raw_model(model):
    model_loader_module = model.metadata.flavors["python_function"]["loader_module"]
    if model_loader_module == "mlflow.sklearn" and not isinstance(model, _ServedPyFuncModel):
        # If we load a sklearn model with mlflow.pyfunc.load_model, the model will be wrapped
        # with _SklearnModelWrapper, we need to extract the raw model from it.
        if isinstance(model._model_impl, _SklearnModelWrapper):
            return model_loader_module, model._model_impl.sklearn_model
        return model_loader_module, model._model_impl
    else:
        return model_loader_module, None


def _extract_predict_fn(model, raw_model):
    predict_fn = model.predict if model is not None else None
    predict_proba_fn = None

    if raw_model is not None:
        predict_fn = raw_model.predict
        predict_proba_fn = getattr(raw_model, "predict_proba", None)

        try:
            import xgboost

            if isinstance(raw_model, xgboost.XGBModel):
                # Because shap evaluation will pass evaluation data in ndarray format
                # (without feature names), if set validate_features=True it will raise error.
                predict_fn = partial(predict_fn, validate_features=False)
                if predict_proba_fn is not None:
                    predict_proba_fn = partial(predict_proba_fn, validate_features=False)
        except ImportError:
            pass

    return predict_fn, predict_proba_fn


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


def _get_binary_sum_up_label_pred_prob(positive_class_index, positive_class, y, y_pred, y_probs):
    y = np.array(y)
    y_bin = np.where(y == positive_class, 1, 0)
    y_pred_bin = None
    y_prob_bin = None
    if y_pred is not None:
        y_pred = np.array(y_pred)
        y_pred_bin = np.where(y_pred == positive_class, 1, 0)

    if y_probs is not None:
        y_probs = np.array(y_probs)
        y_prob_bin = y_probs[:, positive_class_index]

    return y_bin, y_pred_bin, y_prob_bin


def _get_common_classifier_metrics(
    *, y_true, y_pred, y_proba, labels, average, pos_label, sample_weights
):
    metrics = {
        "example_count": len(y_true),
        "accuracy_score": sk_metrics.accuracy_score(y_true, y_pred, sample_weight=sample_weights),
        "recall_score": sk_metrics.recall_score(
            y_true,
            y_pred,
            average=average,
            pos_label=pos_label,
            sample_weight=sample_weights,
        ),
        "precision_score": sk_metrics.precision_score(
            y_true,
            y_pred,
            average=average,
            pos_label=pos_label,
            sample_weight=sample_weights,
        ),
        "f1_score": sk_metrics.f1_score(
            y_true,
            y_pred,
            average=average,
            pos_label=pos_label,
            sample_weight=sample_weights,
        ),
    }
    if y_proba is not None:
        metrics["log_loss"] = sk_metrics.log_loss(
            y_true, y_proba, labels=labels, sample_weight=sample_weights
        )

    return metrics


def _get_binary_classifier_metrics(
    *, y_true, y_pred, y_proba=None, labels=None, pos_label=1, sample_weights=None
):
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        **_get_common_classifier_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            labels=labels,
            average="binary",
            pos_label=pos_label,
            sample_weights=sample_weights,
        ),
    }


def _get_multiclass_classifier_metrics(
    *,
    y_true,
    y_pred,
    y_proba=None,
    labels=None,
    average="weighted",
    sample_weights=None,
):
    metrics = _get_common_classifier_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels,
        average=average,
        pos_label=None,
        sample_weights=sample_weights,
    )
    if average in ("macro", "weighted") and y_proba is not None:
        metrics.update(
            roc_auc=sk_metrics.roc_auc_score(
                y_true=y_true,
                y_score=y_proba,
                sample_weight=sample_weights,
                average=average,
                multi_class="ovr",
            )
        )
    return metrics


def _get_classifier_per_class_metrics_collection_df(y, y_pred, labels, sample_weights):
    per_class_metrics_list = []
    for positive_class_index, positive_class in enumerate(labels):
        (
            y_bin,
            y_pred_bin,
            _,
        ) = _get_binary_sum_up_label_pred_prob(
            positive_class_index, positive_class, y, y_pred, None
        )
        per_class_metrics = {"positive_class": positive_class}
        per_class_metrics.update(
            _get_binary_classifier_metrics(
                y_true=y_bin,
                y_pred=y_pred_bin,
                pos_label=1,
                sample_weights=sample_weights,
            )
        )
        per_class_metrics_list.append(per_class_metrics)

    return pd.DataFrame(per_class_metrics_list)


def _get_dataframe_with_renamed_columns(x, new_column_names):
    """
    Downstream inference functions may expect a pd.DataFrame to be created from x. However,
    if x is already a pd.DataFrame, and new_column_names != x.columns, we cannot simply call
    pd.DataFrame(x, columns=new_column_names) because the resulting pd.DataFrame will contain
    NaNs for every column in new_column_names that does not exist in x.columns. This function
    instead creates a new pd.DataFrame object from x, and then explicitly renames the columns
    to avoid NaNs.

    Args:
        x: A data object, such as a Pandas DataFrame, numPy array, or list
        new_column_names: Column names for the output Pandas DataFrame

    Returns:
        A pd.DataFrame with x as data, with columns new_column_names
    """
    df = pd.DataFrame(x)
    return df.rename(columns=dict(zip(df.columns, new_column_names)))


_Curve = namedtuple("_Curve", ["plot_fn", "plot_fn_args", "auc"])


def _gen_classifier_curve(
    is_binomial,
    y,
    y_probs,
    labels,
    pos_label,
    curve_type,
    sample_weights,
):
    """
    Generate precision-recall curve or ROC curve for classifier.

    Args:
        is_binomial: True if it is binary classifier otherwise False
        y: True label values
        y_probs: if binary classifier, the predicted probability for positive class.
                  if multiclass classifier, the predicted probabilities for all classes.
        labels: The set of labels.
        pos_label: The label of the positive class.
        curve_type: "pr" or "roc"
        sample_weights: Optional sample weights.

    Returns:
        An instance of "_Curve" which includes attributes "plot_fn", "plot_fn_args", "auc".
    """
    if curve_type == "roc":

        def gen_line_x_y_label_auc(_y, _y_prob, _pos_label):
            fpr, tpr, _ = sk_metrics.roc_curve(
                _y,
                _y_prob,
                sample_weight=sample_weights,
                # For multiclass classification where a one-vs-rest ROC curve is produced for each
                # class, the positive label is binarized and should not be included in the plot
                # legend
                pos_label=_pos_label if _pos_label == pos_label else None,
            )
            auc = sk_metrics.roc_auc_score(y_true=_y, y_score=_y_prob, sample_weight=sample_weights)
            return fpr, tpr, f"AUC={auc:.3f}", auc

        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        title = "ROC curve"
        if pos_label:
            xlabel = f"False Positive Rate (Positive label: {pos_label})"
            ylabel = f"True Positive Rate (Positive label: {pos_label})"
    elif curve_type == "pr":

        def gen_line_x_y_label_auc(_y, _y_prob, _pos_label):
            precision, recall, _ = sk_metrics.precision_recall_curve(
                _y,
                _y_prob,
                sample_weight=sample_weights,
                # For multiclass classification where a one-vs-rest precision-recall curve is
                # produced for each class, the positive label is binarized and should not be
                # included in the plot legend
                pos_label=_pos_label if _pos_label == pos_label else None,
            )
            # NB: We return average precision score (AP) instead of AUC because AP is more
            # appropriate for summarizing a precision-recall curve
            ap = sk_metrics.average_precision_score(
                y_true=_y, y_score=_y_prob, pos_label=_pos_label, sample_weight=sample_weights
            )
            return recall, precision, f"AP={ap:.3f}", ap

        xlabel = "Recall"
        ylabel = "Precision"
        title = "Precision recall curve"
        if pos_label:
            xlabel = f"Recall (Positive label: {pos_label})"
            ylabel = f"Precision (Positive label: {pos_label})"
    else:
        assert False, "illegal curve type"

    if is_binomial:
        x_data, y_data, line_label, auc = gen_line_x_y_label_auc(y, y_probs, pos_label)
        data_series = [(line_label, x_data, y_data)]
    else:
        curve_list = []
        for positive_class_index, positive_class in enumerate(labels):
            y_bin, _, y_prob_bin = _get_binary_sum_up_label_pred_prob(
                positive_class_index, positive_class, y, labels, y_probs
            )

            x_data, y_data, line_label, auc = gen_line_x_y_label_auc(
                y_bin, y_prob_bin, _pos_label=1
            )
            curve_list.append((positive_class, x_data, y_data, line_label, auc))

        data_series = [
            (f"label={positive_class},{line_label}", x_data, y_data)
            for positive_class, x_data, y_data, line_label, _ in curve_list
        ]
        auc = [auc for _, _, _, _, auc in curve_list]

    def _do_plot(**kwargs):
        from matplotlib import pyplot

        _, ax = plot_lines(**kwargs)
        dash_line_args = {
            "color": "gray",
            "alpha": 0.3,
            "drawstyle": "default",
            "linestyle": "dashed",
        }
        if curve_type == "pr":
            ax.plot([0, 1], [1, 0], **dash_line_args)
        elif curve_type == "roc":
            ax.plot([0, 1], [0, 1], **dash_line_args)

        if is_binomial:
            ax.legend(loc="best")
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            pyplot.subplots_adjust(right=0.6, bottom=0.25)

    return _Curve(
        plot_fn=_do_plot,
        plot_fn_args={
            "data_series": data_series,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "line_kwargs": {"drawstyle": "steps-post", "linewidth": 1},
            "title": title,
        },
        auc=auc,
    )


def _get_aggregate_metrics_values(metrics):
    return {name: MetricValue(aggregate_results={name: value}) for name, value in metrics.items()}


_matplotlib_config = {
    "figure.dpi": 175,
    "figure.figsize": [6.0, 4.0],
    "figure.autolayout": True,
    "font.size": 8,
}


def _extract_output_and_other_columns(model_predictions, output_column_name):
    y_pred = None
    other_output_columns = None
    ERROR_MISSING_OUTPUT_COLUMN_NAME = (
        "Output column name is not specified for the multi-output model. "
        "Please set the correct output column name using the `predictions` parameter."
    )

    if isinstance(model_predictions, list) and all(isinstance(p, dict) for p in model_predictions):
        # Extract 'y_pred' and 'other_output_columns' from list of dictionaries
        if output_column_name in model_predictions[0]:
            y_pred = pd.Series(
                [p.get(output_column_name) for p in model_predictions], name=output_column_name
            )
            other_output_columns = pd.DataFrame(
                [{k: v for k, v in p.items() if k != output_column_name} for p in model_predictions]
            )
        elif len(model_predictions[0]) == 1:
            # Set the only key as self.predictions and its value as self.y_pred
            key, value = list(model_predictions[0].items())[0]
            y_pred = pd.Series(value, name=key)
            output_column_name = key
        elif output_column_name is None:
            raise MlflowException(
                ERROR_MISSING_OUTPUT_COLUMN_NAME,
                error_code=INVALID_PARAMETER_VALUE,
            )
        else:
            raise MlflowException(
                f"Output column name '{output_column_name}' is not found in the model "
                f"predictions list: {model_predictions}. Please set the correct output column "
                "name using the `predictions` parameter.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    elif isinstance(model_predictions, pd.DataFrame):
        if output_column_name in model_predictions.columns:
            y_pred = model_predictions[output_column_name]
            other_output_columns = model_predictions.drop(columns=output_column_name)
        elif len(model_predictions.columns) == 1:
            output_column_name = model_predictions.columns[0]
            y_pred = model_predictions[output_column_name]
        elif output_column_name is None:
            raise MlflowException(
                ERROR_MISSING_OUTPUT_COLUMN_NAME,
                error_code=INVALID_PARAMETER_VALUE,
            )
        else:
            raise MlflowException(
                f"Output column name '{output_column_name}' is not found in the model "
                f"predictions dataframe {model_predictions.columns}. Please set the correct "
                "output column name using the `predictions` parameter.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    elif isinstance(model_predictions, dict):
        if output_column_name in model_predictions:
            y_pred = pd.Series(model_predictions[output_column_name], name=output_column_name)
            other_output_columns = pd.DataFrame(
                {k: v for k, v in model_predictions.items() if k != output_column_name}
            )
        elif len(model_predictions) == 1:
            key, value = list(model_predictions.items())[0]
            y_pred = pd.Series(value, name=key)
            output_column_name = key
        elif output_column_name is None:
            raise MlflowException(
                ERROR_MISSING_OUTPUT_COLUMN_NAME,
                error_code=INVALID_PARAMETER_VALUE,
            )
        else:
            raise MlflowException(
                f"Output column name '{output_column_name}' is not found in the "
                f"model predictions dict {model_predictions}. Please set the correct "
                "output column name using the `predictions` parameter.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    return (
        y_pred if y_pred is not None else model_predictions,
        other_output_columns,
        output_column_name,
    )


class _Metric(NamedTuple):
    """
    A namedtuple representing a metric function and its properties.

    function : the metric function
    name : the name of the metric function
    index : the index of the function in the ``extra_metrics`` argument of mlflow.evaluate
    """

    function: Callable
    name: str
    index: int
    version: Optional[str] = None


class _CustomArtifact(NamedTuple):
    """
    A namedtuple representing a custom artifact function and its properties.

    function : the custom artifact function
    name : the name of the custom artifact function
    index : the index of the function in the ``custom_artifacts`` argument of mlflow.evaluate
    artifacts_dir : the path to a temporary directory to store produced artifacts of the function
    """

    function: Callable
    name: str
    index: int
    artifacts_dir: str


def _is_numeric(value):
    return isinstance(value, (int, float, np.number))


def _is_string(value):
    return isinstance(value, str)


def _evaluate_metric(metric_tuple, eval_fn_args):
    """
    This function calls the metric function and performs validations on the returned
    result to ensure that they are in the expected format. It will warn and will not log metrics
    that are in the wrong format.

    Args:
        extra_metric_tuple: Containing a user provided function and its index in the
            ``extra_metrics`` parameter of ``mlflow.evaluate``
        eval_fn_args: A dictionary of args needed to compute the eval metrics.

    Returns:
        MetricValue
    """
    if metric_tuple.index < 0:
        exception_header = f"Did not log builtin metric '{metric_tuple.name}' because it"
    else:
        exception_header = (
            f"Did not log metric '{metric_tuple.name}' at index "
            f"{metric_tuple.index} in the `extra_metrics` parameter because it"
        )

    metric = metric_tuple.function(*eval_fn_args)

    if metric is None:
        _logger.warning(f"{exception_header} returned None.")
        return

    if _is_numeric(metric):
        return MetricValue(aggregate_results={metric_tuple.name: metric})

    if not isinstance(metric, MetricValue):
        _logger.warning(f"{exception_header} did not return a MetricValue.")
        return

    scores = metric.scores
    justifications = metric.justifications
    aggregates = metric.aggregate_results

    if scores is not None:
        if not isinstance(scores, list):
            _logger.warning(f"{exception_header} must return MetricValue with scores as a list.")
            return
        if any(not (_is_numeric(score) or _is_string(score) or score is None) for score in scores):
            _logger.warning(
                f"{exception_header} must return MetricValue with numeric or string scores."
            )
            return

    if justifications is not None:
        if not isinstance(justifications, list):
            _logger.warning(
                f"{exception_header} must return MetricValue with justifications as a list."
            )
            return
        if any(not (_is_string(jus) or jus is None) for jus in justifications):
            _logger.warning(
                f"{exception_header} must return MetricValue with string justifications."
            )
            return

    if aggregates is not None:
        if not isinstance(aggregates, dict):
            _logger.warning(
                f"{exception_header} must return MetricValue with aggregate_results as a dict."
            )
            return

        if any(
            not (isinstance(k, str) and (_is_numeric(v) or v is None))
            for k, v in aggregates.items()
        ):
            _logger.warning(
                f"{exception_header} must return MetricValue with aggregate_results with "
                "str keys and numeric values."
            )
            return

    return metric


def _is_valid_artifacts(artifacts):
    return isinstance(artifacts, dict) and all(isinstance(k, str) for k in artifacts.keys())


def _evaluate_custom_artifacts(custom_artifact_tuple, eval_df, builtin_metrics):
    """
    This function calls the `custom_artifact` function and performs validations on the returned
    result to ensure that they are in the expected format. It will raise a MlflowException if
    the result is not in the expected format.

    Args:
        custom_artifact_tuple: Containing a user provided function and its index in the
            ``custom_artifacts`` parameter of ``mlflow.evaluate``
        eval_df: A Pandas dataframe object containing a prediction and a target column.
        builtin_metrics: A dictionary of metrics produced by the default evaluator.

    Returns:
        A dictionary of artifacts.
    """
    exception_header = (
        f"Custom artifact function '{custom_artifact_tuple.name}' "
        " at index {custom_artifact_tuple.index}"
        " in the `custom_artifacts` parameter"
    )
    artifacts = custom_artifact_tuple.function(
        eval_df, builtin_metrics, custom_artifact_tuple.artifacts_dir
    )

    if artifacts is None:
        _logger.warning(f"{exception_header} returned None.")
        return

    if not _is_valid_artifacts(artifacts):
        _logger.warning(
            f"{exception_header} did not return artifacts as a dictionary of string artifact "
            "names with their corresponding objects."
        )
        return

    return artifacts


def _compute_df_mode_or_mean(df):
    """
    Compute mean (for continuous columns) and compute mode (for other columns) for the
    input dataframe, return a dict, key is column name, value is the corresponding mode or
    mean value, this function calls `_is_continuous` to determine whether the
    column is continuous column.
    """
    continuous_cols = [c for c in df.columns if _is_continuous(df[c])]
    df_cont = df[continuous_cols]
    df_non_cont = df.drop(continuous_cols, axis=1)

    means = {} if df_cont.empty else df_cont.mean().to_dict()
    modes = {} if df_non_cont.empty else df_non_cont.mode().loc[0].to_dict()
    return {**means, **modes}


_SUPPORTED_SHAP_ALGORITHMS = ("exact", "permutation", "partition", "kernel")


def _shap_predict_fn(x, predict_fn, feature_names):
    return predict_fn(_get_dataframe_with_renamed_columns(x, feature_names))


class DefaultEvaluator(ModelEvaluator):
    def __init__(self):
        self.client = MlflowClient()

    def can_evaluate(self, *, model_type, evaluator_config, **kwargs):
        return model_type in _ModelType.values() or model_type is None

    def _log_metrics(self):
        """
        Helper method to log metrics into specified run.
        """
        timestamp = get_current_time_millis()
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(
                    key=key,
                    value=value,
                    timestamp=timestamp,
                    step=0,
                )
                for key, value in self.aggregate_metrics.items()
            ],
        )

    def _log_image_artifact(
        self,
        do_plot,
        artifact_name,
    ):
        from matplotlib import pyplot

        prefix = self.evaluator_config.get("metric_prefix", "")
        artifact_file_name = f"{prefix}{artifact_name}.png"
        artifact_file_local_path = self.temp_dir.path(artifact_file_name)

        try:
            pyplot.clf()
            do_plot()
            pyplot.savefig(artifact_file_local_path, bbox_inches="tight")
        finally:
            pyplot.close(pyplot.gcf())

        mlflow.log_artifact(artifact_file_local_path)
        artifact = ImageEvaluationArtifact(uri=mlflow.get_artifact_uri(artifact_file_name))
        artifact._load(artifact_file_local_path)
        self.artifacts[artifact_name] = artifact

    def _log_pandas_df_artifact(self, pandas_df, artifact_name):
        artifact_file_name = f"{artifact_name}.csv"
        artifact_file_local_path = self.temp_dir.path(artifact_file_name)
        pandas_df.to_csv(artifact_file_local_path, index=False)
        mlflow.log_artifact(artifact_file_local_path)
        artifact = CsvEvaluationArtifact(
            uri=mlflow.get_artifact_uri(artifact_file_name),
            content=pandas_df,
        )
        artifact._load(artifact_file_local_path)
        self.artifacts[artifact_name] = artifact

    def _log_model_explainability(self):
        if not self.evaluator_config.get("log_model_explainability", True):
            return

        if self.is_model_server and not self.evaluator_config.get(
            "log_model_explainability", False
        ):
            _logger.warning(
                "Skipping model explainability because a model server is used for environment "
                "restoration."
            )
            return

        if self.model_loader_module == "mlflow.spark":
            # TODO: Shap explainer need to manipulate on each feature values,
            #  but spark model input dataframe contains Vector type feature column
            #  which shap explainer does not support.
            #  To support this, we need expand the Vector type feature column into
            #  multiple scalar feature columns and pass it to shap explainer.
            _logger.warning(
                "Logging model explainability insights is not currently supported for PySpark "
                "models."
            )
            return

        if not (np.issubdtype(self.y.dtype, np.number) or self.y.dtype == np.bool_):
            # Note: python bool type inherits number type but np.bool_ does not inherit np.number.
            _logger.warning(
                "Skip logging model explainability insights because it requires all label "
                "values to be numeric or boolean."
            )
            return

        algorithm = self.evaluator_config.get("explainability_algorithm", None)
        if algorithm is not None and algorithm not in _SUPPORTED_SHAP_ALGORITHMS:
            raise MlflowException(
                message=f"Specified explainer algorithm {algorithm} is unsupported. Currently only "
                f"support {','.join(_SUPPORTED_SHAP_ALGORITHMS)} algorithms.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if algorithm != "kernel":
            feature_dtypes = list(self.X.get_original().dtypes)
            for feature_dtype in feature_dtypes:
                if not np.issubdtype(feature_dtype, np.number):
                    _logger.warning(
                        "Skip logging model explainability insights because the shap explainer "
                        f"{algorithm} requires all feature values to be numeric, and each feature "
                        "column must only contain scalar values."
                    )
                    return

        try:
            import shap
            from matplotlib import pyplot
        except ImportError:
            _logger.warning(
                "SHAP or matplotlib package is not installed, so model explainability insights "
                "will not be logged."
            )
            return

        if Version(shap.__version__) < Version("0.40"):
            _logger.warning(
                "Shap package version is lower than 0.40, Skip log model explainability."
            )
            return

        is_multinomial_classifier = (
            self.model_type == _ModelType.CLASSIFIER and self.num_classes > 2
        )

        sample_rows = self.evaluator_config.get(
            "explainability_nsamples", _DEFAULT_SAMPLE_ROWS_FOR_SHAP
        )

        X_df = self.X.copy_to_avoid_mutation()

        sampled_X = shap.sample(X_df, sample_rows, random_state=0)

        mode_or_mean_dict = _compute_df_mode_or_mean(X_df)
        sampled_X = sampled_X.fillna(mode_or_mean_dict)

        # shap explainer might call provided `predict_fn` with a `numpy.ndarray` type
        # argument, this might break some model inference, so convert the argument into
        # a pandas dataframe.
        # The `shap_predict_fn` calls model's predict function, we need to restore the input
        # dataframe with original column names, because some model prediction routine uses
        # the column name.

        shap_predict_fn = functools.partial(
            _shap_predict_fn, predict_fn=self.predict_fn, feature_names=self.feature_names
        )

        try:
            if algorithm:
                if algorithm == "kernel":
                    # We need to lazily import shap, so lazily import `_PatchedKernelExplainer`
                    from mlflow.models.evaluation._shap_patch import _PatchedKernelExplainer

                    kernel_link = self.evaluator_config.get(
                        "explainability_kernel_link", "identity"
                    )
                    if kernel_link not in ["identity", "logit"]:
                        raise ValueError(
                            "explainability_kernel_link config can only be set to 'identity' or "
                            f"'logit', but got '{kernel_link}'."
                        )
                    background_X = shap.sample(X_df, sample_rows, random_state=3)
                    background_X = background_X.fillna(mode_or_mean_dict)

                    explainer = _PatchedKernelExplainer(
                        shap_predict_fn, background_X, link=kernel_link
                    )
                else:
                    explainer = shap.Explainer(
                        shap_predict_fn,
                        sampled_X,
                        feature_names=self.feature_names,
                        algorithm=algorithm,
                    )
            else:
                if (
                    self.raw_model
                    and not is_multinomial_classifier
                    and not isinstance(self.raw_model, sk_Pipeline)
                ):
                    # For mulitnomial classifier, shap.Explainer may choose Tree/Linear explainer
                    # for raw model, this case shap plot doesn't support it well, so exclude the
                    # multinomial_classifier case here.
                    explainer = shap.Explainer(
                        self.raw_model, sampled_X, feature_names=self.feature_names
                    )
                else:
                    # fallback to default explainer
                    explainer = shap.Explainer(
                        shap_predict_fn, sampled_X, feature_names=self.feature_names
                    )

            _logger.info(f"Shap explainer {explainer.__class__.__name__} is used.")

            if algorithm == "kernel":
                shap_values = shap.Explanation(
                    explainer.shap_values(sampled_X), feature_names=self.feature_names
                )
            else:
                shap_values = explainer(sampled_X)
        except Exception as e:
            # Shap evaluation might fail on some edge cases, e.g., unsupported input data values
            # or unsupported model on specific shap explainer. Catch exception to prevent it
            # breaking the whole `evaluate` function.

            if not self.evaluator_config.get("ignore_exceptions", True):
                raise e

            _logger.warning(
                f"Shap evaluation failed. Reason: {e!r}. "
                "Set logging level to DEBUG to see the full traceback."
            )
            _logger.debug("", exc_info=True)
            return
        try:
            mlflow.shap.log_explainer(explainer, artifact_path="explainer")
        except Exception as e:
            # TODO: The explainer saver is buggy, if `get_underlying_model_flavor` return "unknown",
            #   then fallback to shap explainer saver, and shap explainer will call `model.save`
            #   for sklearn model, there is no `.save` method, so error will happen.
            _logger.warning(
                f"Logging explainer failed. Reason: {e!r}. "
                "Set logging level to DEBUG to see the full traceback."
            )
            _logger.debug("", exc_info=True)

        def _adjust_color_bar():
            pyplot.gcf().axes[-1].set_aspect("auto")
            pyplot.gcf().axes[-1].set_box_aspect(50)

        def _adjust_axis_tick():
            pyplot.xticks(fontsize=10)
            pyplot.yticks(fontsize=10)

        def plot_beeswarm():
            shap.plots.beeswarm(shap_values, show=False, color_bar=True)
            _adjust_color_bar()
            _adjust_axis_tick()

        self._log_image_artifact(
            plot_beeswarm,
            "shap_beeswarm_plot",
        )

        def plot_summary():
            shap.summary_plot(shap_values, show=False, color_bar=True)
            _adjust_color_bar()
            _adjust_axis_tick()

        self._log_image_artifact(
            plot_summary,
            "shap_summary_plot",
        )

        def plot_feature_importance():
            shap.plots.bar(shap_values, show=False)
            _adjust_axis_tick()

        self._log_image_artifact(
            plot_feature_importance,
            "shap_feature_importance_plot",
        )

    def _evaluate_sklearn_model_score_if_scorable(self):
        if self.model_loader_module == "mlflow.sklearn" and self.raw_model is not None:
            try:
                score = self.raw_model.score(
                    self.X.copy_to_avoid_mutation(), self.y, sample_weight=self.sample_weights
                )
                self.metrics_values.update(_get_aggregate_metrics_values({"score": score}))
            except Exception as e:
                _logger.warning(
                    f"Computing sklearn model score failed: {e!r}. Set logging level to "
                    "DEBUG to see the full traceback."
                )
                _logger.debug("", exc_info=True)

    def _compute_roc_and_pr_curve(self):
        if self.y_probs is not None:
            self.roc_curve = _gen_classifier_curve(
                is_binomial=True,
                y=self.y,
                y_probs=self.y_probs[:, 1],
                labels=self.label_list,
                pos_label=self.pos_label,
                curve_type="roc",
                sample_weights=self.sample_weights,
            )

            self.metrics_values.update(
                _get_aggregate_metrics_values({"roc_auc": self.roc_curve.auc})
            )
            self.pr_curve = _gen_classifier_curve(
                is_binomial=True,
                y=self.y,
                y_probs=self.y_probs[:, 1],
                labels=self.label_list,
                pos_label=self.pos_label,
                curve_type="pr",
                sample_weights=self.sample_weights,
            )

            self.metrics_values.update(
                _get_aggregate_metrics_values({"precision_recall_auc": self.pr_curve.auc})
            )

    def _log_multiclass_classifier_artifacts(self):
        per_class_metrics_collection_df = _get_classifier_per_class_metrics_collection_df(
            self.y,
            self.y_pred,
            labels=self.label_list,
            sample_weights=self.sample_weights,
        )

        log_roc_pr_curve = False
        if self.y_probs is not None:
            max_classes_for_multiclass_roc_pr = self.evaluator_config.get(
                "max_classes_for_multiclass_roc_pr", 10
            )
            if self.num_classes <= max_classes_for_multiclass_roc_pr:
                log_roc_pr_curve = True
            else:
                _logger.warning(
                    f"The classifier num_classes > {max_classes_for_multiclass_roc_pr}, skip "
                    f"logging ROC curve and Precision-Recall curve. You can add evaluator config "
                    f"'max_classes_for_multiclass_roc_pr' to increase the threshold."
                )

        if log_roc_pr_curve:
            roc_curve = _gen_classifier_curve(
                is_binomial=False,
                y=self.y,
                y_probs=self.y_probs,
                labels=self.label_list,
                pos_label=self.pos_label,
                curve_type="roc",
                sample_weights=self.sample_weights,
            )

            def plot_roc_curve():
                roc_curve.plot_fn(**roc_curve.plot_fn_args)

            self._log_image_artifact(plot_roc_curve, "roc_curve_plot")
            per_class_metrics_collection_df["roc_auc"] = roc_curve.auc

            pr_curve = _gen_classifier_curve(
                is_binomial=False,
                y=self.y,
                y_probs=self.y_probs,
                labels=self.label_list,
                pos_label=self.pos_label,
                curve_type="pr",
                sample_weights=self.sample_weights,
            )

            def plot_pr_curve():
                pr_curve.plot_fn(**pr_curve.plot_fn_args)

            self._log_image_artifact(plot_pr_curve, "precision_recall_curve_plot")
            per_class_metrics_collection_df["precision_recall_auc"] = pr_curve.auc

        self._log_pandas_df_artifact(per_class_metrics_collection_df, "per_class_metrics")

    def _log_binary_classifier_artifacts(self):
        from mlflow.models.evaluation.lift_curve import plot_lift_curve

        if self.y_probs is not None:

            def plot_roc_curve():
                self.roc_curve.plot_fn(**self.roc_curve.plot_fn_args)

            self._log_image_artifact(plot_roc_curve, "roc_curve_plot")

            def plot_pr_curve():
                self.pr_curve.plot_fn(**self.pr_curve.plot_fn_args)

            self._log_image_artifact(plot_pr_curve, "precision_recall_curve_plot")

            self._log_image_artifact(
                lambda: plot_lift_curve(self.y, self.y_probs, pos_label=self.pos_label),
                "lift_curve_plot",
            )

    def _log_custom_metric_artifact(self, artifact_name, raw_artifact, custom_metric_tuple):
        """
        This function logs and returns a custom metric artifact. Two cases:
            - The provided artifact is a path to a file, the function will make a copy of it with
              a formatted name in a temporary directory and call mlflow.log_artifact.
            - Otherwise: will attempt to save the artifact to an temporary path with an inferred
              type. Then call mlflow.log_artifact.

        Args:
            artifact_name: the name of the artifact
            raw_artifact: the object representing the artifact
            custom_metric_tuple: an instance of the _CustomMetric namedtuple

        Returns:
            EvaluationArtifact
        """

        exception_and_warning_header = (
            f"Custom artifact function '{custom_metric_tuple.name}' at index "
            f"{custom_metric_tuple.index} in the `custom_artifacts` parameter"
        )

        inferred_from_path, inferred_type, inferred_ext = _infer_artifact_type_and_ext(
            artifact_name, raw_artifact, custom_metric_tuple
        )
        artifact_file_local_path = self.temp_dir.path(artifact_name + inferred_ext)

        if pathlib.Path(artifact_file_local_path).exists():
            raise MlflowException(
                f"{exception_and_warning_header} produced an artifact '{artifact_name}' that "
                "cannot be logged because there already exists an artifact with the same name."
            )

        # ParquetEvaluationArtifact isn't explicitly stated here because such artifacts can only
        # be supplied through file. Which is handled by the first if clause. This is because
        # DataFrame objects default to be stored as CsvEvaluationArtifact.
        if inferred_from_path:
            shutil.copy2(raw_artifact, artifact_file_local_path)
        elif inferred_type is JsonEvaluationArtifact:
            with open(artifact_file_local_path, "w") as f:
                if isinstance(raw_artifact, str):
                    f.write(raw_artifact)
                else:
                    json.dump(raw_artifact, f, cls=NumpyEncoder)
        elif inferred_type is CsvEvaluationArtifact:
            raw_artifact.to_csv(artifact_file_local_path, index=False)
        elif inferred_type is NumpyEvaluationArtifact:
            np.save(artifact_file_local_path, raw_artifact, allow_pickle=False)
        elif inferred_type is ImageEvaluationArtifact:
            raw_artifact.savefig(artifact_file_local_path)
        else:
            # storing as pickle
            try:
                with open(artifact_file_local_path, "wb") as f:
                    pickle.dump(raw_artifact, f)
                _logger.warning(
                    f"{exception_and_warning_header} produced an artifact '{artifact_name}'"
                    f" with type '{type(raw_artifact)}' that is logged as a pickle artifact."
                )
            except pickle.PickleError:
                raise MlflowException(
                    f"{exception_and_warning_header} produced an unsupported artifact "
                    f"'{artifact_name}' with type '{type(raw_artifact)}' that cannot be pickled. "
                    "Supported object types for artifacts are:\n"
                    "- A string uri representing the file path to the artifact. MLflow"
                    "  will infer the type of the artifact based on the file extension.\n"
                    "- A string representation of a JSON object. This will be saved as a "
                    ".json artifact.\n"
                    "- Pandas DataFrame. This will be saved as a .csv artifact."
                    "- Numpy array. This will be saved as a .npy artifact."
                    "- Matplotlib Figure. This will be saved as an .png image artifact."
                    "- Other objects will be attempted to be pickled with default protocol."
                )

        mlflow.log_artifact(artifact_file_local_path)
        artifact = inferred_type(uri=mlflow.get_artifact_uri(artifact_name + inferred_ext))
        artifact._load(artifact_file_local_path)
        return artifact

    def _get_column_in_metrics_values(self, column):
        for metric_name, metric_value in self.metrics_values.items():
            if metric_name.split("/")[0] == column:
                return metric_value

    def _get_args_for_metrics(
        self, metric_tuple, eval_df, input_df
    ) -> Tuple[bool, List[Union[str, pd.DataFrame]]]:
        """
        Given a metric_tuple, read the signature of the metric function and get the appropriate
        arguments from the input/output columns, other calculated metrics, and evaluator_config.

        Args:
            metric_tuple: The metric tuple containing a user provided function and its index
                in the ``extra_metrics`` parameter of ``mlflow.evaluate``.
            eval_df: The evaluation dataframe containing the prediction and target columns.
            input_df: The input dataframe containing the features used to make predictions.

        Returns:
            tuple: A tuple of (bool, list) where the bool indicates if the given metric can
            be calculated with the given eval_df, metrics, and input_df.
                - If the user is missing "targets" or "predictions" parameters when needed, or we
                cannot find a column or metric for a parameter to the metric, return
                    (False, list of missing parameters)
                - If all arguments to the metric function were found, return
                    (True, list of arguments).
        """
        # deepcopying eval_df and builtin_metrics for each custom metric function call,
        # in case the user modifies them inside their function(s).
        eval_df_copy = eval_df.copy()
        parameters = inspect.signature(metric_tuple.function).parameters
        eval_fn_args = []
        params_not_found = []
        if len(parameters) == 2:
            param_0_name, param_1_name = parameters.keys()

        # eval_fn has parameters (eval_df, builtin_metrics) for backwards compatibility
        if len(parameters) == 2 and param_0_name != "predictions" and param_1_name != "targets":
            eval_fn_args.append(eval_df_copy)
            self._update_aggregate_metrics()
            eval_fn_args.append(copy.deepcopy(self.aggregate_metrics))
        # eval_fn can have parameters like (predictions, targets, metrics, random_col)
        else:
            for param_name, param in parameters.items():
                column = self.col_mapping.get(param_name, param_name)

                if (
                    column == "predictions"
                    or column == self.predictions
                    or column == self.dataset.predictions_name
                ):
                    eval_fn_args.append(eval_df_copy["prediction"])
                elif column == "targets" or column == self.dataset.targets_name:
                    if "target" in eval_df_copy:
                        eval_fn_args.append(eval_df_copy["target"])
                    else:
                        if param.default == inspect.Parameter.empty:
                            params_not_found.append(param_name)
                        else:
                            eval_fn_args.append(param.default)
                elif column == "metrics":
                    eval_fn_args.append(copy.deepcopy(self.metrics_values))
                else:
                    # case when column passed in col_mapping contains the entire column
                    if not isinstance(column, str):
                        eval_fn_args.append(column)

                    # case column in col_mapping is string and the column value
                    # is part of the input_df
                    elif column in input_df.columns:
                        eval_fn_args.append(input_df[column])

                    # case column in col_mapping is string and the column value
                    # is part of the output_df(other than predictions)
                    elif (
                        self.other_output_columns is not None
                        and column in self.other_output_columns.columns
                    ):
                        self.other_output_columns_for_eval.add(column)
                        eval_fn_args.append(self.other_output_columns[column])

                    # case where the param is defined as part of the evaluator_config
                    elif column in self.evaluator_config:
                        eval_fn_args.append(self.evaluator_config.get(column))

                    # case where this is the name of another metric
                    elif metric_value := self._get_column_in_metrics_values(column):
                        eval_fn_args.append(metric_value)

                    # in the case that:
                    # the metric has not been calculated yet, but is scheduled to be calculated
                    # "before" this metric in self.ordered_metrics, we append None to indicate
                    # that there is not an error in the dependencies
                    elif column in [metric_tuple.name for metric_tuple in self.ordered_metrics]:
                        eval_fn_args.append(None)

                    elif param.default == inspect.Parameter.empty:
                        params_not_found.append(param_name)
                    else:
                        eval_fn_args.append(param.default)

        if len(params_not_found) > 0:
            return False, params_not_found
        return True, eval_fn_args

    def _log_custom_artifacts(self, eval_df):
        if not self.custom_artifacts:
            return
        for index, custom_artifact in enumerate(self.custom_artifacts):
            with tempfile.TemporaryDirectory() as artifacts_dir:
                # deepcopying eval_df and builtin_metrics for each custom artifact function call,
                # in case the user modifies them inside their function(s).
                custom_artifact_tuple = _CustomArtifact(
                    function=custom_artifact,
                    index=index,
                    name=getattr(custom_artifact, "__name__", repr(custom_artifact)),
                    artifacts_dir=artifacts_dir,
                )
                artifact_results = _evaluate_custom_artifacts(
                    custom_artifact_tuple,
                    eval_df.copy(),
                    copy.deepcopy(self.metrics_values),
                )
                if artifact_results:
                    for artifact_name, raw_artifact in artifact_results.items():
                        self.artifacts[artifact_name] = self._log_custom_metric_artifact(
                            artifact_name,
                            raw_artifact,
                            custom_artifact_tuple,
                        )

    def _log_confusion_matrix(self):
        """
        Helper method for logging confusion matrix
        """
        # normalize the confusion matrix, keep consistent with sklearn autologging.
        confusion_matrix = sk_metrics.confusion_matrix(
            self.y,
            self.y_pred,
            labels=self.label_list,
            normalize="true",
            sample_weight=self.sample_weights,
        )

        def plot_confusion_matrix():
            import matplotlib
            import matplotlib.pyplot as plt

            with matplotlib.rc_context(
                {
                    "font.size": min(8, math.ceil(50.0 / self.num_classes)),
                    "axes.labelsize": 8,
                }
            ):
                _, ax = plt.subplots(1, 1, figsize=(6.0, 4.0), dpi=175)
                disp = sk_metrics.ConfusionMatrixDisplay(
                    confusion_matrix=confusion_matrix,
                    display_labels=self.label_list,
                ).plot(cmap="Blues", ax=ax)
                disp.ax_.set_title("Normalized confusion matrix")

        if hasattr(sk_metrics, "ConfusionMatrixDisplay"):
            self._log_image_artifact(
                plot_confusion_matrix,
                "confusion_matrix",
            )
        return

    def _generate_model_predictions(self, compute_latency=False):
        """
        Helper method for generating model predictions
        """

        def predict_with_latency(X_copy):
            y_pred_list = []
            pred_latencies = []
            if len(X_copy) == 0:
                raise ValueError("Empty input data")

            is_dataframe = isinstance(X_copy, pd.DataFrame)

            for row in X_copy.iterrows() if is_dataframe else enumerate(X_copy):
                i, row_data = row
                single_input = row_data.to_frame().T if is_dataframe else row_data
                start_time = time.time()
                y_pred = self.model.predict(single_input)
                end_time = time.time()
                pred_latencies.append(end_time - start_time)
                y_pred_list.append(y_pred)

            # Update latency metric
            self.metrics_values.update({_LATENCY_METRIC_NAME: MetricValue(scores=pred_latencies)})

            # Aggregate all predictions into model_predictions
            sample_pred = y_pred_list[0]
            if isinstance(sample_pred, pd.DataFrame):
                return pd.concat(y_pred_list)
            elif isinstance(sample_pred, np.ndarray):
                return np.concatenate(y_pred_list, axis=0)
            elif isinstance(sample_pred, list):
                return sum(y_pred_list, [])
            elif isinstance(sample_pred, pd.Series):
                return pd.concat(y_pred_list, ignore_index=True)
            else:
                raise MlflowException(
                    message=f"Unsupported prediction type {type(sample_pred)} for model type "
                    f"{self.model_type}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        X_copy = self.X.copy_to_avoid_mutation()
        if self.model is not None:
            _logger.info("Computing model predictions.")

            if compute_latency:
                model_predictions = predict_with_latency(X_copy)
            else:
                model_predictions = self.model.predict(X_copy)
        else:
            if self.dataset.predictions_data is None:
                raise MlflowException(
                    message="Predictions data is missing when model is not provided. "
                    "Please provide predictions data in a dataset or provide a model. "
                    "See the documentation for mlflow.evaluate() for how to specify "
                    "the predictions data in a dataset.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if compute_latency:
                _logger.warning(
                    "Setting the latency to 0 for all entries because the model " "is not provided."
                )
                self.metrics_values.update(
                    {_LATENCY_METRIC_NAME: MetricValue(scores=[0.0] * len(X_copy))}
                )
            model_predictions = self.dataset.predictions_data

        if self.model_type == _ModelType.CLASSIFIER:
            self.label_list = np.unique(self.y)
            self.num_classes = len(self.label_list)

            if self.predict_fn is not None:
                self.y_pred = self.predict_fn(self.X.copy_to_avoid_mutation())
            else:
                self.y_pred = self.dataset.predictions_data
            self.is_binomial = self.num_classes <= 2

            if self.is_binomial:
                if self.pos_label in self.label_list:
                    self.label_list = np.delete(
                        self.label_list, np.where(self.label_list == self.pos_label)
                    )
                    self.label_list = np.append(self.label_list, self.pos_label)
                elif self.pos_label is None:
                    self.pos_label = self.label_list[-1]
                _logger.info(
                    "The evaluation dataset is inferred as binary dataset, positive label is "
                    f"{self.label_list[1]}, negative label is {self.label_list[0]}."
                )
            else:
                _logger.info(
                    "The evaluation dataset is inferred as multiclass dataset, number of classes "
                    f"is inferred as {self.num_classes}"
                )

            if self.predict_proba_fn is not None:
                self.y_probs = self.predict_proba_fn(self.X.copy_to_avoid_mutation())
            else:
                self.y_probs = None

        output_column_name = self.predictions
        (
            self.y_pred,
            self.other_output_columns,
            self.predictions,
        ) = _extract_output_and_other_columns(model_predictions, output_column_name)
        self.other_output_columns_for_eval = set()

    def _compute_builtin_metrics(self):
        """
        Helper method for computing builtin metrics
        """
        self._evaluate_sklearn_model_score_if_scorable()
        if self.model_type == _ModelType.CLASSIFIER:
            if self.is_binomial:
                self.metrics_values.update(
                    _get_aggregate_metrics_values(
                        _get_binary_classifier_metrics(
                            y_true=self.y,
                            y_pred=self.y_pred,
                            y_proba=self.y_probs,
                            labels=self.label_list,
                            pos_label=self.pos_label,
                            sample_weights=self.sample_weights,
                        )
                    )
                )
                self._compute_roc_and_pr_curve()
            else:
                average = self.evaluator_config.get("average", "weighted")
                self.metrics_values.update(
                    _get_aggregate_metrics_values(
                        _get_multiclass_classifier_metrics(
                            y_true=self.y,
                            y_pred=self.y_pred,
                            y_proba=self.y_probs,
                            labels=self.label_list,
                            average=average,
                            sample_weights=self.sample_weights,
                        )
                    )
                )
        elif self.model_type == _ModelType.REGRESSOR:
            self.metrics_values.update(
                _get_aggregate_metrics_values(
                    _get_regressor_metrics(self.y, self.y_pred, self.sample_weights)
                )
            )

    def _get_error_message_missing_columns(self, metric_name, param_names):
        error_message_parts = [f"Metric '{metric_name}' requires the following:"]

        special_params = ["targets", "predictions"]
        for param in special_params:
            if param in param_names:
                error_message_parts.append(f"  - the '{param}' parameter needs to be specified")

        remaining_params = [param for param in param_names if param not in special_params]

        if remaining_params:
            error_message_parts.append(
                f"  - missing columns {remaining_params} need to be defined or mapped"
            )

        return "\n".join(error_message_parts)

    def _construct_error_message_for_malformed_metrics(
        self, malformed_results, input_columns, output_columns
    ):
        error_messages = [
            self._get_error_message_missing_columns(metric_name, param_names)
            for metric_name, param_names in malformed_results
        ]
        joined_error_message = "\n".join(error_messages)

        full_message = f"""Error: Metric calculation failed for the following metrics:
        {joined_error_message}

        Below are the existing column names for the input/output data:
        Input Columns: {input_columns}
        Output Columns: {output_columns}

        To resolve this issue, you may need to:
         - specify any required parameters
         - if you are missing columns, check that there are no circular dependencies among your
         metrics, and you may want to map them to an existing column using the following
         configuration:
        evaluator_config={{'col_mapping': {{<missing column name>: <existing column name>}}}}"""

        return "\n".join(l.lstrip() for l in full_message.splitlines())

    def _raise_exception_for_malformed_metrics(self, malformed_results, eval_df):
        output_columns = (
            [] if self.other_output_columns is None else list(self.other_output_columns.columns)
        )
        if self.predictions:
            output_columns.append(self.predictions)
        elif self.dataset.predictions_name:
            output_columns.append(self.dataset.predictions_name)
        else:
            output_columns.append("predictions")

        input_columns = list(self.X.copy_to_avoid_mutation().columns)
        if "target" in eval_df:
            if self.dataset.targets_name:
                input_columns.append(self.dataset.targets_name)
            else:
                input_columns.append("targets")

        error_message = self._construct_error_message_for_malformed_metrics(
            malformed_results, input_columns, output_columns
        )

        raise MlflowException(error_message, error_code=INVALID_PARAMETER_VALUE)

    # to order the metrics, we append metrics to self.ordered_metrics if they can be calculated
    # given the metrics that will be calculated before it
    # we stop when all metrics are in self.ordered_metrics or we cannot "calculate" any more metrics
    # and raise an exception in the latter case
    def _order_extra_metrics(self, eval_df):
        remaining_metrics = self.extra_metrics
        input_df = self.X.copy_to_avoid_mutation()

        while len(remaining_metrics) > 0:
            pending_metrics = []
            failed_results = []
            did_append_metric = False
            for metric_tuple in remaining_metrics:
                can_calculate, eval_fn_args = self._get_args_for_metrics(
                    metric_tuple, eval_df, input_df
                )
                if can_calculate:
                    self.ordered_metrics.append(metric_tuple)
                    did_append_metric = True
                else:  # cannot calculate the metric yet
                    pending_metrics.append(metric_tuple)
                    failed_results.append((metric_tuple.name, eval_fn_args))

            # cant calculate any more metrics
            if not did_append_metric:
                self._raise_exception_for_malformed_metrics(failed_results, eval_df)

            remaining_metrics = pending_metrics

    def _test_first_row(self, eval_df):
        # test calculations on first row of eval_df
        _logger.info("Testing metrics on first row...")
        exceptions = []
        first_row_df = eval_df.iloc[[0]]
        first_row_input_df = self.X.copy_to_avoid_mutation().iloc[[0]]
        for metric_tuple in self.ordered_metrics:
            try:
                _, eval_fn_args = self._get_args_for_metrics(
                    metric_tuple, first_row_df, first_row_input_df
                )
                metric_value = _evaluate_metric(metric_tuple, eval_fn_args)
                if metric_value:
                    name = (
                        f"{metric_tuple.name}/{metric_tuple.version}"
                        if metric_tuple.version
                        else metric_tuple.name
                    )
                    self.metrics_values.update({name: metric_value})
            except Exception as e:
                stacktrace_str = traceback.format_exc()
                if isinstance(e, MlflowException):
                    exceptions.append(
                        f"Metric '{metric_tuple.name}': Error:\n{e.message}\n{stacktrace_str}"
                    )
                else:
                    exceptions.append(
                        f"Metric '{metric_tuple.name}': Error:\n{e!r}\n{stacktrace_str}"
                    )

        if len(exceptions) > 0:
            raise MlflowException("\n".join(exceptions))

    def _metric_to_metric_tuple(self, index, metric):
        return _Metric(
            function=metric.eval_fn, index=index, name=metric.name, version=metric.version
        )

    def _evaluate_metrics(self, eval_df):
        self._order_extra_metrics(eval_df)
        self._test_first_row(eval_df)

        # calculate metrics for the full eval_df
        input_df = self.X.copy_to_avoid_mutation()
        for metric_tuple in self.ordered_metrics:
            _, eval_fn_args = self._get_args_for_metrics(metric_tuple, eval_df, input_df)
            metric_value = _evaluate_metric(metric_tuple, eval_fn_args)

            if metric_value:
                name = (
                    f"{metric_tuple.name}/{metric_tuple.version}"
                    if metric_tuple.version
                    else metric_tuple.name
                )
                self.metrics_values.update({name: metric_value})

    def _log_artifacts(self):
        """
        Helper method for generating artifacts, logging metrics and artifacts.
        """
        if self.model_type in (_ModelType.CLASSIFIER, _ModelType.REGRESSOR):
            if self.model_type == _ModelType.CLASSIFIER:
                if self.is_binomial:
                    self._log_binary_classifier_artifacts()
                else:
                    self._log_multiclass_classifier_artifacts()
                self._log_confusion_matrix()
            self._log_model_explainability()

    def _log_eval_table(self):
        # only log eval table if there are per row metrics recorded
        if not any(
            metric_value.scores is not None or metric_value.justifications is not None
            for _, metric_value in self.metrics_values.items()
        ):
            return

        metric_prefix = self.evaluator_config.get("metric_prefix", "")
        if not isinstance(metric_prefix, str):
            metric_prefix = ""
        if isinstance(self.dataset.features_data, pd.DataFrame):
            # Handle DataFrame case
            if self.dataset.has_targets:
                data = self.dataset.features_data.assign(
                    **{
                        self.dataset.targets_name or "target": self.y,
                        self.dataset.predictions_name or self.predictions or "outputs": self.y_pred,
                    }
                )
            else:
                data = self.dataset.features_data.assign(outputs=self.y_pred)
        else:
            # Handle NumPy array case, converting it to a DataFrame
            data = pd.DataFrame(self.dataset.features_data, columns=self.dataset.feature_names)
            if self.dataset.has_targets:
                data = data.assign(
                    **{
                        self.dataset.targets_name or "target": self.y,
                        self.dataset.predictions_name or self.predictions or "outputs": self.y_pred,
                    }
                )
            else:
                data = data.assign(outputs=self.y_pred)

        # Include other_output_columns used in evaluation to the eval table
        if self.other_output_columns is not None and len(self.other_output_columns_for_eval) > 0:
            for column in self.other_output_columns_for_eval:
                data[column] = self.other_output_columns[column]

        columns = {}
        for metric_name, metric_value in self.metrics_values.items():
            scores = metric_value.scores
            justifications = metric_value.justifications

            if scores:
                if metric_name.startswith(metric_prefix) and metric_name[len(metric_prefix) :] in [
                    _TOKEN_COUNT_METRIC_NAME,
                    _LATENCY_METRIC_NAME,
                ]:
                    columns[metric_name] = scores
                else:
                    columns[f"{metric_name}/score"] = scores
            if justifications:
                columns[f"{metric_name}/justification"] = justifications
        data = data.assign(**columns)
        artifact_file_name = f"{metric_prefix}{_EVAL_TABLE_FILE_NAME}"
        mlflow.log_table(data, artifact_file=artifact_file_name)
        if self.eval_results_path:
            eval_table_spark = self.spark_session.createDataFrame(data)
            try:
                eval_table_spark.write.mode(self.eval_results_mode).option(
                    "mergeSchema", "true"
                ).format("delta").saveAsTable(self.eval_results_path)
            except Exception as e:
                _logger.info(f"Saving eval table to delta table failed. Reason: {e}")

        name = _EVAL_TABLE_FILE_NAME.split(".", 1)[0]
        self.artifacts[name] = JsonEvaluationArtifact(
            uri=mlflow.get_artifact_uri(artifact_file_name)
        )

    def _update_aggregate_metrics(self):
        self.aggregate_metrics = {}
        for metric_name, metric_value in self.metrics_values.items():
            if metric_value.aggregate_results:
                for agg_name, agg_value in metric_value.aggregate_results.items():
                    if agg_value is not None:
                        if agg_name == metric_name.split("/")[0]:
                            self.aggregate_metrics[metric_name] = agg_value
                        else:
                            self.aggregate_metrics[f"{metric_name}/{agg_name}"] = agg_value

    def _handle_builtin_metrics_by_model_type(self):
        text_metrics = [
            token_count(),
            toxicity(),
            flesch_kincaid_grade_level(),
            ari_grade_level(),
        ]
        builtin_metrics = []

        if self.model_type in (_ModelType.CLASSIFIER, _ModelType.REGRESSOR):
            self._compute_builtin_metrics()
        elif self.model_type == _ModelType.QUESTION_ANSWERING:
            builtin_metrics = [*text_metrics, exact_match()]
        elif self.model_type == _ModelType.TEXT_SUMMARIZATION:
            builtin_metrics = [
                *text_metrics,
                rouge1(),
                rouge2(),
                rougeL(),
                rougeLsum(),
            ]
        elif self.model_type == _ModelType.TEXT:
            builtin_metrics = text_metrics
        elif self.model_type == _ModelType.RETRIEVER:
            # default k to 3 if not specified
            retriever_k = self.evaluator_config.pop("retriever_k", 3)
            builtin_metrics = [
                precision_at_k(retriever_k),
                recall_at_k(retriever_k),
                ndcg_at_k(retriever_k),
            ]

        self.ordered_metrics = [
            self._metric_to_metric_tuple(-1, metric) for metric in builtin_metrics
        ]

    def _add_prefix_to_metrics(self):
        def _prefix_value(value):
            aggregate = (
                {f"{prefix}{k}": v for k, v in value.aggregate_results.items()}
                if value.aggregate_results
                else None
            )
            return MetricValue(value.scores, value.justifications, aggregate)

        if prefix := self.evaluator_config.get("metric_prefix"):
            self.metrics_values = {
                f"{prefix}{k}": _prefix_value(v) for k, v in self.metrics_values.items()
            }

        self._update_aggregate_metrics()

    def _evaluate(
        self,
        model: "mlflow.pyfunc.PyFuncModel" = None,
        is_baseline_model=False,
        **kwargs,
    ):
        import matplotlib

        with TempDir() as temp_dir, matplotlib.rc_context(_matplotlib_config):
            self.temp_dir = temp_dir
            self.model = model

            self.is_model_server = isinstance(model, _ServedPyFuncModel)

            if getattr(model, "metadata", None):
                self.model_loader_module, self.raw_model = _extract_raw_model(model)
            else:
                # model is constructed from a user specified function or not provided
                self.model_loader_module, self.raw_model = None, None
            self.predict_fn, self.predict_proba_fn = _extract_predict_fn(model, self.raw_model)

            self.artifacts = {}
            self.aggregate_metrics = {}
            self.metrics_values = {}
            self.ordered_metrics = []

            with mlflow.utils.autologging_utils.disable_autologging():
                compute_latency = False
                for extra_metric in self.extra_metrics:
                    # If latency metric is specified, we will compute latency for the model
                    # during prediction, and we will remove the metric from the list of extra
                    # metrics to be computed after prediction.
                    if extra_metric.name == _LATENCY_METRIC_NAME:
                        compute_latency = True
                        self.extra_metrics.remove(extra_metric)
                        break
                self._generate_model_predictions(compute_latency=compute_latency)
                self._handle_builtin_metrics_by_model_type()

                eval_df = pd.DataFrame({"prediction": copy.deepcopy(self.y_pred)})
                if self.dataset.has_targets:
                    eval_df["target"] = self.y

                self._evaluate_metrics(eval_df)
                if not is_baseline_model:
                    self._log_custom_artifacts(eval_df)

                self._add_prefix_to_metrics()

                if not is_baseline_model:
                    self._log_artifacts()
                    self._log_metrics()
                    self._log_eval_table()
                return EvaluationResult(
                    metrics=self.aggregate_metrics, artifacts=self.artifacts, run_id=self.run_id
                )

    def evaluate(
        self,
        *,
        model_type,
        dataset,
        run_id,
        evaluator_config,
        model: "mlflow.pyfunc.PyFuncModel" = None,
        custom_metrics=None,
        extra_metrics=None,
        custom_artifacts=None,
        baseline_model=None,
        predictions=None,
        **kwargs,
    ):
        self.dataset = dataset
        self.run_id = run_id
        self.model_type = model_type
        self.evaluator_config = evaluator_config
        self.feature_names = dataset.feature_names

        self.custom_artifacts = custom_artifacts
        self.y = dataset.labels_data
        self.predictions = predictions
        self.col_mapping = self.evaluator_config.get("col_mapping", {})
        self.pos_label = self.evaluator_config.get("pos_label")
        self.sample_weights = self.evaluator_config.get("sample_weights")
        self.eval_results_path = self.evaluator_config.get("eval_results_path")
        self.eval_results_mode = self.evaluator_config.get("eval_results_mode", "overwrite")

        if self.eval_results_path:
            from mlflow.utils._spark_utils import _get_active_spark_session

            self.spark_session = _get_active_spark_session()
            if not self.spark_session:
                raise MlflowException(
                    message="eval_results_path is only supported in Spark environment. ",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if self.eval_results_mode not in ["overwrite", "append"]:
                raise MlflowException(
                    message="eval_results_mode can only be 'overwrite' or 'append'. ",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        if extra_metrics and custom_metrics:
            raise MlflowException(
                "The 'custom_metrics' parameter in mlflow.evaluate is deprecated. Please update "
                "your code to only use the 'extra_metrics' parameter instead."
            )
        if custom_metrics:
            warnings.warn(
                "The 'custom_metrics' parameter in mlflow.evaluate is deprecated. "
                "Please update your code to use the 'extra_metrics' parameter instead.",
                FutureWarning,
                stacklevel=2,
            )
            extra_metrics = custom_metrics

        if extra_metrics is None:
            extra_metrics = []

        bad_metrics = []
        for metric in extra_metrics:
            if not isinstance(metric, EvaluationMetric):
                bad_metrics.append(metric)
        if len(bad_metrics) > 0:
            message = "\n".join(
                [f"- Metric '{m}' has type '{type(m).__name__}'" for m in bad_metrics]
            )
            raise MlflowException(
                f"In the 'extra_metrics' parameter, the following metrics have the wrong type:\n"
                f"{message}\n"
                f"Please ensure that all extra metrics are instances of "
                f"mlflow.metrics.EvaluationMetric."
            )

        self.extra_metrics = [
            self._metric_to_metric_tuple(index, metric)
            for index, metric in enumerate(extra_metrics)
        ]

        if self.model_type in (_ModelType.CLASSIFIER, _ModelType.REGRESSOR):
            inferred_model_type = _infer_model_type_by_labels(self.y)
            if inferred_model_type is not None and model_type != inferred_model_type:
                _logger.warning(
                    f"According to the evaluation dataset label values, the model type looks like "
                    f"{inferred_model_type}, but you specified model type {model_type}. Please "
                    f"verify that you set the `model_type` and `dataset` arguments correctly."
                )

        if evaluator_config.get("_disable_candidate_model", False):
            evaluation_result = EvaluationResult(metrics={}, artifacts={})
        else:
            if baseline_model:
                _logger.info("Evaluating candidate model:")
            evaluation_result = self._evaluate(model, is_baseline_model=False)

        if not baseline_model:
            return evaluation_result

        _logger.info("Evaluating baseline model:")
        baseline_evaluation_result = self._evaluate(baseline_model, is_baseline_model=True)

        return EvaluationResult(
            metrics=evaluation_result.metrics,
            artifacts=evaluation_result.artifacts,
            baseline_model_metrics=baseline_evaluation_result.metrics,
            run_id=self.run_id,
        )

    @property
    def X(self) -> pd.DataFrame:
        """
        The features (`X`) portion of the dataset, guarded against accidental mutations.
        """
        return DefaultEvaluator._MutationGuardedData(
            _get_dataframe_with_renamed_columns(self.dataset.features_data, self.feature_names)
        )

    class _MutationGuardedData:
        """
        Wrapper around a data object that requires explicit API calls to obtain either a copy
        of the data object, or, in cases where the caller can guaranteed that the object will not
        be mutated, the original data object.
        """

        def __init__(self, data):
            """
            Args:
                data: A data object, such as a Pandas DataFrame, numPy array, or list.
            """
            self._data = data

        def copy_to_avoid_mutation(self):
            """
            Obtain a copy of the data. This method should be called every time the data needs
            to be used in a context where it may be subsequently mutated, guarding against
            accidental reuse after mutation.

            Returns:
                A copy of the data object.
            """
            if isinstance(self._data, pd.DataFrame):
                return self._data.copy(deep=True)
            else:
                return copy.deepcopy(self._data)

        def get_original(self):
            """
            Obtain the original data object. This method should only be called if the caller
            can guarantee that it will not mutate the data during subsequent operations.

            Returns:
                The original data object.
            """
            return self._data
