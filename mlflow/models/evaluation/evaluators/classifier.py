import logging
import math
from collections import namedtuple
from contextlib import contextmanager
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

import mlflow
from mlflow import MlflowException
from mlflow.environment_variables import _MLFLOW_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS
from mlflow.models.evaluation.artifacts import CsvEvaluationArtifact
from mlflow.models.evaluation.base import EvaluationMetric, EvaluationResult, _ModelType
from mlflow.models.evaluation.evaluators.base import (
    BuiltInEvaluator,
    _extract_raw_model,
    _get_aggregate_metrics_values,
)
from mlflow.models.utils import plot_lines

_logger = logging.getLogger(__name__)


_Curve = namedtuple("_Curve", ["plot_fn", "plot_fn_args", "auc"])


class ClassifierEvaluator(BuiltInEvaluator):
    """
    A built-in evaluator for classifier models.
    """

    name = "classifier"

    @classmethod
    def can_evaluate(cls, *, model_type, evaluator_config, **kwargs):
        # TODO: Also the model needs to be pyfunc model, not function or endpoint URI
        return model_type == _ModelType.CLASSIFIER

    def _evaluate(
        self,
        model: Optional["mlflow.pyfunc.PyFuncModel"],
        extra_metrics: List[EvaluationMetric],
        custom_artifacts=None,
        **kwargs,
    ) -> Optional[EvaluationResult]:
        # Get classification config
        self.y_true = self.dataset.labels_data
        self.label_list = self.evaluator_config.get("label_list")
        self.pos_label = self.evaluator_config.get("pos_label")
        self.sample_weights = self.evaluator_config.get("sample_weights")
        if self.pos_label and self.label_list and self.pos_label not in self.label_list:
            raise MlflowException.invalid_parameter_value(
                f"'pos_label' {self.pos_label} must exist in 'label_list' {self.label_list}."
            )

        # Check if the model_type is consistent with ground truth labels
        inferred_model_type = _infer_model_type_by_labels(self.y_true)
        if _ModelType.CLASSIFIER != inferred_model_type:
            _logger.warning(
                f"According to the evaluation dataset label values, the model type looks like "
                f"{inferred_model_type}, but you specified model type 'classifier'. Please "
                f"verify that you set the `model_type` and `dataset` arguments correctly."
            )

        # Run model prediction
        input_df = self.X.copy_to_avoid_mutation()
        self.y_pred, self.y_probs = self._generate_model_predictions(model, input_df)

        self._validate_label_list()

        self._compute_builtin_metrics(model)
        self.evaluate_metrics(extra_metrics, prediction=self.y_pred, target=self.y_true)
        self.evaluate_and_log_custom_artifacts(
            custom_artifacts, prediction=self.y_pred, target=self.y_true
        )

        # Log metrics and artifacts
        self.log_metrics()
        self.log_eval_table(self.y_pred)

        if len(self.label_list) == 2:
            self._log_binary_classifier_artifacts()
        else:
            self._log_multiclass_classifier_artifacts()
        self._log_confusion_matrix()

        return EvaluationResult(
            metrics=self.aggregate_metrics, artifacts=self.artifacts, run_id=self.run_id
        )

    def _generate_model_predictions(self, model, input_df):
        predict_fn, predict_proba_fn = _extract_predict_fn_and_prodict_proba_fn(model)
        # Classifier model is guaranteed to output single column of predictions
        y_pred = self.dataset.predictions_data if model is None else predict_fn(input_df)

        # Predict class probabilities if the model supports it
        y_probs = predict_proba_fn(input_df) if predict_proba_fn is not None else None
        return y_pred, y_probs

    def _validate_label_list(self):
        if self.label_list is None:
            # If label list is not specified, infer label list from model output
            self.label_list = np.unique(np.concatenate([self.y_true, self.y_pred]))
        else:
            # np.where only works for numpy array, not list
            self.label_list = np.array(self.label_list)

        # sort label_list ASC, for binary classification it makes sure the last one is pos label
        self.label_list.sort()

        is_binomial = len(self.label_list) <= 2
        if is_binomial:
            if self.pos_label is None:
                self.pos_label = self.label_list[-1]
            else:
                if self.pos_label in self.label_list:
                    self.label_list = np.delete(
                        self.label_list, np.where(self.label_list == self.pos_label)
                    )
                self.label_list = np.append(self.label_list, self.pos_label)
            if len(self.label_list) < 2:
                raise MlflowException(
                    "Evaluation dataset for classification must contain at least two unique "
                    f"labels, but only {len(self.label_list)} unique labels were found.",
                )
            with _suppress_class_imbalance_errors(IndexError, log_warning=False):
                _logger.info(
                    "The evaluation dataset is inferred as binary dataset, positive label is "
                    f"{self.label_list[1]}, negative label is {self.label_list[0]}."
                )
        else:
            _logger.info(
                "The evaluation dataset is inferred as multiclass dataset, number of classes "
                f"is inferred as {len(self.label_list)}. If this is incorrect, please specify the "
                "`label_list` parameter in `evaluator_config`."
            )

    def _compute_builtin_metrics(self, model):
        self._evaluate_sklearn_model_score_if_scorable(model, self.y_true, self.sample_weights)

        if len(self.label_list) <= 2:
            metrics = _get_binary_classifier_metrics(
                y_true=self.y_true,
                y_pred=self.y_pred,
                y_proba=self.y_probs,
                labels=self.label_list,
                pos_label=self.pos_label,
                sample_weights=self.sample_weights,
            )
            if metrics:
                self.metrics_values.update(_get_aggregate_metrics_values(metrics))
                self._compute_roc_and_pr_curve()
        else:
            average = self.evaluator_config.get("average", "weighted")
            metrics = _get_multiclass_classifier_metrics(
                y_true=self.y_true,
                y_pred=self.y_pred,
                y_proba=self.y_probs,
                labels=self.label_list,
                average=average,
                sample_weights=self.sample_weights,
            )
            if metrics:
                self.metrics_values.update(_get_aggregate_metrics_values(metrics))

    def _compute_roc_and_pr_curve(self):
        if self.y_probs is not None:
            with _suppress_class_imbalance_errors(ValueError, log_warning=False):
                self.roc_curve = _gen_classifier_curve(
                    is_binomial=True,
                    y=self.y_true,
                    y_probs=self.y_probs[:, 1],
                    labels=self.label_list,
                    pos_label=self.pos_label,
                    curve_type="roc",
                    sample_weights=self.sample_weights,
                )

                self.metrics_values.update(
                    _get_aggregate_metrics_values({"roc_auc": self.roc_curve.auc})
                )
            with _suppress_class_imbalance_errors(ValueError, log_warning=False):
                self.pr_curve = _gen_classifier_curve(
                    is_binomial=True,
                    y=self.y_true,
                    y_probs=self.y_probs[:, 1],
                    labels=self.label_list,
                    pos_label=self.pos_label,
                    curve_type="pr",
                    sample_weights=self.sample_weights,
                )

                self.metrics_values.update(
                    _get_aggregate_metrics_values({"precision_recall_auc": self.pr_curve.auc})
                )

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

    def _log_multiclass_classifier_artifacts(self):
        per_class_metrics_collection_df = _get_classifier_per_class_metrics_collection_df(
            y=self.y_true,
            y_pred=self.y_pred,
            labels=self.label_list,
            sample_weights=self.sample_weights,
        )

        log_roc_pr_curve = False
        if self.y_probs is not None:
            max_classes_for_multiclass_roc_pr = self.evaluator_config.get(
                "max_classes_for_multiclass_roc_pr", 10
            )
            if len(self.label_list) <= max_classes_for_multiclass_roc_pr:
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
                y=self.y_true,
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
                y=self.y_true,
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

    def _log_roc_curve(self):
        def _plot_roc_curve():
            self.roc_curve.plot_fn(**self.roc_curve.plot_fn_args)

        self._log_image_artifact(_plot_roc_curve, "roc_curve_plot")

    def _log_precision_recall_curve(self):
        def _plot_pr_curve():
            self.pr_curve.plot_fn(**self.pr_curve.plot_fn_args)

        self._log_image_artifact(_plot_pr_curve, "precision_recall_curve_plot")

    def _log_lift_curve(self):
        def _plot_lift_curve():
            return plot_lift_curve(self.y_true, self.y_probs, pos_label=self.pos_label)

        self._log_image_artifact(_plot_lift_curve, "lift_curve_plot")

    def _log_binary_classifier_artifacts(self):
        if self.y_probs is not None:
            with _suppress_class_imbalance_errors(log_warning=False):
                self._log_roc_curve()
            with _suppress_class_imbalance_errors(log_warning=False):
                self._log_precision_recall_curve()
            with _suppress_class_imbalance_errors(ValueError, log_warning=False):
                self._log_lift_curve()

    def _log_confusion_matrix(self):
        """
        Helper method for logging confusion matrix
        """
        # normalize the confusion matrix, keep consistent with sklearn autologging.
        confusion_matrix = sk_metrics.confusion_matrix(
            self.y_true,
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
                    "font.size": min(8, math.ceil(50.0 / len(self.label_list))),
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


def _extract_predict_fn_and_prodict_proba_fn(model):
    predict_fn = None
    predict_proba_fn = None

    _, raw_model = _extract_raw_model(model)

    if raw_model is not None:
        predict_fn = raw_model.predict
        predict_proba_fn = getattr(raw_model, "predict_proba", None)
        try:
            from mlflow.xgboost import (
                _wrapped_xgboost_model_predict_fn,
                _wrapped_xgboost_model_predict_proba_fn,
            )

            # Because shap evaluation will pass evaluation data in ndarray format
            # (without feature names), if set validate_features=True it will raise error.
            predict_fn = _wrapped_xgboost_model_predict_fn(raw_model, validate_features=False)
            predict_proba_fn = _wrapped_xgboost_model_predict_proba_fn(
                raw_model, validate_features=False
            )
        except ImportError:
            pass
    elif model is not None:
        predict_fn = model.predict

    return predict_fn, predict_proba_fn


@contextmanager
def _suppress_class_imbalance_errors(exception_type=Exception, log_warning=True):
    """
    Exception handler context manager to suppress Exceptions if the private environment
    variable `_MLFLOW_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS` is set to `True`.
    The purpose of this handler is to prevent an evaluation call for a binary or multiclass
    classification automl run from aborting due to an extreme minority class imbalance
    encountered during iterative training cycles due to the non deterministic sampling
    behavior of Spark's DataFrame.sample() API.
    The Exceptions caught in the usage of this are broad and are designed purely to not
    interrupt the iterative hyperparameter tuning process. Final evaluations are done
    in a more deterministic (but expensive) fashion.
    """
    try:
        yield
    except exception_type as e:
        if _MLFLOW_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS.get():
            if log_warning:
                _logger.warning(
                    "Failed to calculate metrics due to class imbalance. "
                    "This is expected when the dataset is imbalanced."
                )
        else:
            raise e


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
        with _suppress_class_imbalance_errors(ValueError):
            metrics["log_loss"] = sk_metrics.log_loss(
                y_true, y_proba, labels=labels, sample_weight=sample_weights
            )
    return metrics


def _get_binary_classifier_metrics(
    *, y_true, y_pred, y_proba=None, labels=None, pos_label=1, sample_weights=None
):
    with _suppress_class_imbalance_errors(ValueError):
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
        binary_classifier_metrics = _get_binary_classifier_metrics(
            y_true=y_bin,
            y_pred=y_pred_bin,
            pos_label=1,
            sample_weights=sample_weights,
        )
        if binary_classifier_metrics:
            per_class_metrics.update(binary_classifier_metrics)
        per_class_metrics_list.append(per_class_metrics)

    return pd.DataFrame(per_class_metrics_list)


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


def _cumulative_gain_curve(y_true, y_score, pos_label=None):
    """
    This method is copied from scikit-plot package.
    See https://github.com/reiinakano/scikit-plot/blob/2dd3e6a76df77edcbd724c4db25575f70abb57cb/scikitplot/helpers.py#L157

    This function generates the points necessary to plot the Cumulative Gain

    Note: This implementation is restricted to the binary classification task.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).

        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative

    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.

        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains


def plot_lift_curve(
    y_true,
    y_probas,
    title="Lift Curve",
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    pos_label=None,
):
    """
    This method is copied from scikit-plot package.
    See https://github.com/reiinakano/scikit-plot/blob/2dd3e6a76df77edcbd724c4db25575f70abb57cb/scikitplot/metrics.py#L1133

    Generates the Lift Curve from labels and scores/probabilities

    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html.
    The implementation here works only for binary classification.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Lift Curve".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

        pos_label (optional): Label for the positive class.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> plot_lift_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_lift_curve.png
           :align: center
           :alt: Lift Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(f"Cannot calculate Lift Curve for data with {len(classes)} category/ies")

    # Compute Cumulative Gain Curves
    percentages, gains1 = _cumulative_gain_curve(y_true, y_probas[:, 0], classes[0])
    percentages, gains2 = _cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])

    percentages = percentages[1:]
    gains1 = gains1[1:]
    gains2 = gains2[1:]

    gains1 = gains1 / percentages
    gains2 = gains2 / percentages

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    label0 = f"Class {classes[0]}"
    label1 = f"Class {classes[1]}"
    # show (positive) next to the positive class in the legend
    if pos_label:
        if pos_label == classes[0]:
            label0 = f"Class {classes[0]} (positive)"
        elif pos_label == classes[1]:
            label1 = f"Class {classes[1]} (positive)"
        # do not mark positive class if pos_label is not in classes

    ax.plot(percentages, gains1, lw=3, label=label0)
    ax.plot(percentages, gains2, lw=3, label=label1)

    ax.plot([0, 1], [1, 1], "k--", lw=2, label="Baseline")

    ax.set_xlabel("Percentage of sample", fontsize=text_fontsize)
    ax.set_ylabel("Lift", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid("on")
    ax.legend(loc="best", fontsize=text_fontsize)

    return ax
