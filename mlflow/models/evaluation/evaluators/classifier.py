import logging
import math
from collections import namedtuple
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

import mlflow
from mlflow import MlflowException
from mlflow.environment_variables import _MLFLOW_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS
from mlflow.models.evaluation.artifacts import CsvEvaluationArtifact
from mlflow.models.evaluation.base import EvaluationMetric, EvaluationResult, _ModelType
from mlflow.models.evaluation.default_evaluator import (
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
        extra_metrics: list[EvaluationMetric],
        custom_artifacts=None,
        **kwargs,
    ) -> Optional[EvaluationResult]:
        # Get classification config
        self.y_true = self.dataset.labels_data
        self.label_list = self.evaluator_config.get("label_list")
        self.pos_label = self.evaluator_config.get("pos_label")
        self.sample_weights = self.evaluator_config.get("sample_weights")

        # Validate pos_label only for binary classification
        if self.pos_label is not None and self.label_list is not None:
            if len(self.label_list) == 2 and self.pos_label not in self.label_list:
                raise MlflowException.invalid_parameter_value(
                    f"'pos_label' {self.pos_label} must exist in 'label_list' {self.label_list}."
                )
            # For multiclass, pos_label is ignored, so no validation needed

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

        # Normalize data types to ensure consistency before validation
        self._normalize_data_types()
        self._validate_label_list()
        if self.label_list is None:
            # If label_list is not set due to validation error, stop evaluation
            return None

        # Tag the chosen positive label on the run for auditability (only for binary)
        _logger.debug(f"At tag set: label_list={self.label_list}, pos_label={self.pos_label}")
        if self.label_list is not None and len(self.label_list) == 2:
            pos_label_to_log = self.pos_label if self.pos_label is not None else self.label_list[-1]
            mlflow.set_tag("evaluation.positive_label", str(pos_label_to_log))
        # For multiclass, do not set this tag

        self._compute_builtin_metrics(model)
        self.evaluate_metrics(extra_metrics, prediction=self.y_pred, target=self.y_true)
        self.evaluate_and_log_custom_artifacts(
            custom_artifacts, prediction=self.y_pred, target=self.y_true
        )

        # Log metrics and artifacts
        self.log_metrics()
        self.log_eval_table(self.y_pred)

        if self.label_list is not None and len(self.label_list) == 2:
            self._log_binary_classifier_artifacts()
        elif self.label_list is not None:
            self._log_multiclass_classifier_artifacts()
        self._log_confusion_matrix()

        # Expose the chosen pos_label as a "metric" so it shows in the metric table
        # Only log positive_label for binary classification (for transparency)
        if self.label_list is not None and len(self.label_list) == 2 and self.pos_label is not None:
            self.aggregate_metrics["positive_label"] = self.pos_label
        # For multiclass, do not log positive_label (no single positive class)

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
        # Check for mixed/incompatible types in y_true and y_pred
        all_labels = list(self.y_true) + list(self.y_pred)

        # Normalize types to handle Python native vs NumPy types
        def normalize_type(x):
            if isinstance(x, (np.integer, int, np.bool_, bool)):
                return int  # treat bool and int as compatible
            elif isinstance(x, (np.floating, float)):
                return float
            elif isinstance(x, (np.str_, str)):
                return str
            else:
                return type(x)

        # Check if we have mixed types that can be coerced
        types = {normalize_type(x) for x in all_labels}

        if len(types) > 1:
            is_valid_mix = types.issubset({int, float})

            if not is_valid_mix and (str in types and (int in types or float in types)):
                try:
                    numeric_labels = []
                    for label in all_labels:
                        if isinstance(label, str):
                            try:
                                numeric_labels.append(int(label))
                            except ValueError:
                                numeric_labels.append(float(label))
                        else:
                            numeric_labels.append(label)

                    numeric_types = {normalize_type(x) for x in numeric_labels}
                    if numeric_types.issubset({int, float}):
                        is_valid_mix = True
                except (ValueError, TypeError):
                    is_valid_mix = False

            if not is_valid_mix:
                raise MlflowException.invalid_parameter_value(
                    f"Inconsistent label types detected in y_true/y_pred: {types}. "
                    "All labels must be of the same type or compatible numeric types (int/float)."
                )

        if self.label_list is None:
            # If label list is not specified, infer label list from model output
            self.label_list = np.unique(np.concatenate([self.y_true, self.y_pred]))
            # Sort inferred labels for consistent behavior
            try:
                self.label_list.sort()
            except TypeError:
                # Handle mixed-type array by converting to string before sorting
                self.label_list = np.sort(self.label_list.astype(str))

            # If pos_label is provided and we have binary classification,
            # check it is in the inferred label_list
            if (
                self.pos_label is not None
                and len(self.label_list) == 2
                and self.pos_label not in self.label_list
            ):
                raise MlflowException.invalid_parameter_value(
                    f"'pos_label' {self.pos_label} must exist in 'label_list' {self.label_list}."
                )
            # For multiclass, pos_label is ignored, so no validation needed
        else:
            # If user provided label_list, preserve their order
            self.label_list = np.array(self.label_list)

            # Validate pos_label is in label_list when explicitly provided
            if self.pos_label is not None and self.pos_label not in self.label_list:
                raise MlflowException.invalid_parameter_value(
                    f"'pos_label' {self.pos_label} must exist in 'label_list' {self.label_list}."
                )

        # Validate that we have at least 2 classes for classification
        if len(self.label_list) < 2:
            raise MlflowException(
                "Evaluation dataset for classification must contain at least two unique "
                f"labels, but only {len(self.label_list)} unique label(s) were found.",
            )

        is_binomial = len(self.label_list) == 2
        if is_binomial:
            if self.pos_label is None:
                # Dynamically choose last label as positive by default
                self.pos_label = self.label_list[-1]
                _logger.warning(
                    f"No `pos_label` provided—defaulting to positive label = "
                    f"{self.pos_label!r}. If this is not what you intended, please "
                    f"specify `evaluator_config['pos_label']` explicitly."
                )
            # pos_label validation is already done in _evaluate method
            with _suppress_class_imbalance_errors(IndexError, log_warning=False):
                negative_label = (
                    self.label_list[0]
                    if self.pos_label == self.label_list[1]
                    else self.label_list[1]
                )
                _logger.info(
                    "The evaluation dataset is inferred as binary dataset, positive label is "
                    f"{self.pos_label}, negative label is {negative_label}"
                )
        else:
            if self.pos_label is not None:
                # pos_label is ignored for multiclass classification - no warning needed
                pass

    def _normalize_data_types(self):
        """Normalize data types to ensure consistency for sklearn functions."""

        def to_numeric(series):
            try:
                return pd.to_numeric(series)
            except (ValueError, TypeError):
                return series

        self.y_true = to_numeric(pd.Series(self.y_true)).to_numpy()
        self.y_pred = to_numeric(pd.Series(self.y_pred)).to_numpy()

        if self.label_list is not None:
            self.label_list = to_numeric(pd.Series(self.label_list)).to_numpy()

        if self.pos_label is not None:
            # pos_label might be a string "0" that needs to be converted
            try:
                self.pos_label = to_numeric(pd.Series([self.pos_label]))[0]
            except (ValueError, TypeError):
                pass

    def _compute_builtin_metrics(self, model):
        self._evaluate_sklearn_model_score_if_scorable(model, self.y_true, self.sample_weights)

        if self.label_list is not None and len(self.label_list) <= 2:
            # Check if model supports predict_proba for binary classification
            if self.y_probs is None and model is not None:
                _, predict_proba_fn = _extract_predict_fn_and_prodict_proba_fn(model)
                if predict_proba_fn is None:
                    _logger.info(
                        "No predict_proba available—skipping ROC and PR curve computation."
                    )
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
                if self.y_probs is not None:
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
        if self.y_probs is None:
            return  # nothing to do
        # Determine which column holds the positive-class probability
        pos_index = list(self.label_list).index(self.pos_label)
        # Binarize ground truth
        y_true_bin = np.array([1 if y == self.pos_label else 0 for y in self.y_true])
        with _suppress_class_imbalance_errors(ValueError, log_warning=False):
            self.roc_curve = _gen_classifier_curve(
                is_binomial=True,
                y=y_true_bin,
                y_probs=self.y_probs[:, pos_index],
                labels=[0, 1],  # work in binary space
                pos_label=1,  # positive = 1 in this space
                curve_type="roc",
                sample_weights=self.sample_weights,
            )

            self.metrics_values.update(
                _get_aggregate_metrics_values({"roc_auc": self.roc_curve.auc})
            )
        with _suppress_class_imbalance_errors(ValueError, log_warning=False):
            self.pr_curve = _gen_classifier_curve(
                is_binomial=True,
                y=y_true_bin,
                y_probs=self.y_probs[:, pos_index],
                labels=[0, 1],  # work in binary space
                pos_label=1,  # positive = 1 in this space
                curve_type="pr",
                sample_weights=self.sample_weights,
            )

            self.metrics_values.update(
                _get_aggregate_metrics_values({"precision_recall_auc": self.pr_curve.auc})
            )

    def _log_pandas_df_artifact(self, pandas_df, artifact_name):
        artifact_file_name = f"{artifact_name}"
        if not artifact_name.endswith(".csv"):
            artifact_file_name += ".csv"

        artifact_file_local_path = self.temp_dir.path(artifact_file_name)
        pandas_df.to_csv(artifact_file_local_path, index=True)
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
            pos_label=None,  # No single positive label for multiclass
        )

        log_roc_pr_curve = False
        if self.y_probs is not None:
            with _suppress_class_imbalance_errors(TypeError, log_warning=False):
                self._log_calibration_curve()

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
            with _suppress_class_imbalance_errors(ValueError, log_warning=False):
                roc_curve = _gen_classifier_curve(
                    is_binomial=False,
                    y=self.y_true,
                    y_probs=self.y_probs,
                    labels=self.label_list,
                    pos_label=None,  # No single positive label for multiclass
                    curve_type="roc",
                    sample_weights=self.sample_weights,
                )

                def plot_roc_curve():
                    roc_curve.plot_fn(**roc_curve.plot_fn_args)

                self._log_image_artifact(plot_roc_curve, "roc_curve_plot")
                per_class_metrics_collection_df["roc_auc"] = roc_curve.auc

            with _suppress_class_imbalance_errors(ValueError, log_warning=False):
                pr_curve = _gen_classifier_curve(
                    is_binomial=False,
                    y=self.y_true,
                    y_probs=self.y_probs,
                    labels=self.label_list,
                    pos_label=None,  # No single positive label for multiclass
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
        from mlflow.models.evaluation.lift_curve import plot_lift_curve

        def _plot_lift_curve():
            return plot_lift_curve(self.y_true, self.y_probs, pos_label=self.pos_label)

        self._log_image_artifact(_plot_lift_curve, "lift_curve_plot")

    def _log_calibration_curve(self):
        from mlflow.models.evaluation.calibration_curve import plot_calibration_curve

        # For binary classification, use pos_label; for multiclass, use None
        pos_label_for_calibration = self.pos_label if len(self.label_list) == 2 else None

        def _plot_calibration_curve():
            return plot_calibration_curve(
                y_true=self.y_true,
                y_probs=self.y_probs,
                pos_label=pos_label_for_calibration,
                calibration_config={
                    k: v for k, v in self.evaluator_config.items() if k.startswith("calibration_")
                },
                label_list=self.label_list,
            )

        self._log_image_artifact(_plot_calibration_curve, "calibration_curve_plot")

    def _log_binary_classifier_artifacts(self):
        if self.y_probs is not None:
            with _suppress_class_imbalance_errors(log_warning=False):
                self._log_roc_curve()
            with _suppress_class_imbalance_errors(log_warning=False):
                self._log_precision_recall_curve()
            with _suppress_class_imbalance_errors(ValueError, log_warning=False):
                self._log_lift_curve()
            with _suppress_class_imbalance_errors(TypeError, log_warning=False):
                self._log_calibration_curve()

    def _log_confusion_matrix(self):
        """
        Helper method for logging confusion matrix
        """
        # normalize the confusion matrix, keep consistent with sklearn autologging.
        normalized_cm_array = sk_metrics.confusion_matrix(
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
                    confusion_matrix=normalized_cm_array,
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
    Return True represent they are continuous, return False represent we cannot determine result.
    """
    try:
        dtype_name = pd.Series(values).convert_dtypes().dtype.name.lower()
        return dtype_name.startswith("float")
    except (AttributeError, TypeError, ValueError):
        # If we can't determine the dtype or if there are issues with the data,
        # assume it's not continuous to be safe
        return False


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
    # For binary classification, if pos_label is None, use the last label as default
    if average == "binary" and pos_label is None and labels is not None:
        pos_label = labels[-1]

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
    """Compute binary classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for positive class. Optional.
        labels: List of labels to use for confusion matrix. If provided, ensures
               proper 2x2 structure even with single-class data.
        pos_label: Label of the positive class. Defaults to 1.
        sample_weights: Sample weights. Optional.

    Returns:
        Dictionary containing binary classification metrics including true_negatives,
        false_positives, false_negatives, true_positives, and other common metrics.
    """
    with _suppress_class_imbalance_errors(ValueError):
        # Use labels parameter to ensure proper 2x2 confusion matrix structure
        cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels)
        flat = cm.ravel()
        if flat.size != 4:
            _logger.warning(
                f"Unexpected confusion_matrix shape {cm.shape}; setting TN/FP/FN/TP = 0"
            )
            tn = fp = fn = tp = 0
        else:
            tn, fp, fn, tp = flat

        # Get common metrics (these will be computed normally even with zero confusion matrix
        # values)
        common_metrics = _get_common_classifier_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            labels=labels,
            average="binary",
            pos_label=pos_label,
            sample_weights=sample_weights,
        )

        return {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            **common_metrics,
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
        try:
            metrics.update(
                roc_auc=sk_metrics.roc_auc_score(
                    y_true=y_true,
                    y_score=y_proba,
                    sample_weight=sample_weights,
                    average=average,
                    multi_class="ovr",
                    labels=labels,
                )
            )
        except ValueError:
            metrics.update(roc_auc=math.nan)
    return metrics


def _get_classifier_per_class_metrics_collection_df(
    y, y_pred, labels, sample_weights, pos_label=None
):
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
        # Since _get_binary_sum_up_label_pred_prob converts everything to 0/1,
        # we need to use [0, 1] as labels for the confusion matrix
        # The pos_label should be 1 since that's what represents the positive class
        try:
            binary_classifier_metrics = _get_binary_classifier_metrics(
                y_true=y_bin,
                y_pred=y_pred_bin,
                labels=[0, 1],
                pos_label=1,
                sample_weights=sample_weights,
            )
        except ValueError:
            binary_classifier_metrics = {"roc_auc": float("nan")}
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

            try:
                auc = sk_metrics.roc_auc_score(
                    y_true=_y, y_score=_y_prob, sample_weight=sample_weights
                )
            except ValueError:
                # Handle case where only one class is present in y_true
                auc = math.nan
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
            try:
                ap = sk_metrics.average_precision_score(
                    y_true=_y, y_score=_y_prob, pos_label=_pos_label, sample_weight=sample_weights
                )
            except ValueError:
                # Handle case where only one class is present in y_true
                ap = math.nan
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
