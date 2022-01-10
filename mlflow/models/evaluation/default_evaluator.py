import mlflow
from mlflow.models.evaluation.base import (
    ModelEvaluator,
    EvaluationMetrics,
    EvaluationResult,
)
from mlflow.entities.metric import Metric
from mlflow.utils.file_utils import TempDir
from mlflow.utils.string_utils import truncate_str_from_middle
from mlflow.models.utils import plot_lines
from mlflow.models.evaluation.artifacts import ImageEvaluationArtifact, CsvEvaluationArtifact

from sklearn import metrics as sk_metrics
import math
from collections import namedtuple
import numbers
import pandas as pd
import numpy as np
import time
from functools import partial
import logging
from packaging.version import Version

_logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_ROWS_FOR_SHAP = 2000


def _infer_model_type_by_labels(labels):
    distinct_labels = set(labels)
    for v in distinct_labels:
        if not isinstance(v, numbers.Number):
            return "classifier"
        if not float(v).is_integer():
            return "regressor"
    if len(distinct_labels) > 1000 and len(distinct_labels) / len(labels) > 0.7:
        return "regressor"
    return "classifier"


def _extract_raw_model_and_predict_fn(model):
    model_loader_module = model.metadata.flavors["python_function"]["loader_module"]
    predict_fn = model.predict
    predict_proba_fn = None

    try:
        if model_loader_module == "mlflow.sklearn":
            raw_model = model._model_impl
        else:
            raw_model = None
    except Exception as e:
        raw_model = None
        _logger.warning(
            f"Raw model resolution fails unexpectedly on PyFuncModel {model!r}, "
            f"error message is {e}"
        )

    if raw_model:
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

    return model_loader_module, raw_model, predict_fn, predict_proba_fn


def _gen_log_key(key, dataset_name):
    return f"{key}_on_data_{dataset_name}"


def _get_regressor_metrics(y, y_pred):
    return {
        "example_count": len(y),
        "mean_absolute_error": sk_metrics.mean_absolute_error(y, y_pred),
        "mean_squared_error": sk_metrics.mean_squared_error(y, y_pred),
        "root_mean_squared_error": math.sqrt(sk_metrics.mean_squared_error(y, y_pred)),
        "sum_on_label": sum(y),
        "mean_on_label": sum(y) / len(y),
        "r2_score": sk_metrics.r2_score(y, y_pred),
        "max_error": sk_metrics.max_error(y, y_pred),
        "mean_absolute_percentage_error": sk_metrics.mean_absolute_percentage_error(y, y_pred),
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


def _get_classifier_per_class_metrics(y, y_pred):
    """
    get classifier metrics which computing over a specific class.
    For binary classifier, y/y_pred is for the positive class.
    For multiclass classifier, y/y_pred sum up to a binary "is class" and "is not class".
    """
    metrics = {}
    confusion_matrix = sk_metrics.confusion_matrix(y, y_pred)
    tn, fp, fn, tp = confusion_matrix.ravel()
    metrics["true_negatives"] = tn
    metrics["false_positives"] = fp
    metrics["false_negatives"] = fn
    metrics["true_positives"] = tp
    metrics["recall"] = sk_metrics.recall_score(y, y_pred)
    metrics["precision"] = sk_metrics.precision_score(y, y_pred)
    metrics["f1_score"] = sk_metrics.f1_score(y, y_pred)
    return metrics


def _get_classifier_global_metrics(is_binomial, y, y_pred, y_probs, labels):
    """
    get classifier metrics which computing over all classes examples.
    """
    metrics = {}
    metrics["accuracy"] = sk_metrics.accuracy_score(y, y_pred)
    metrics["example_count"] = len(y)

    if not is_binomial:
        metrics["f1_score_micro"] = sk_metrics.f1_score(y, y_pred, average="micro", labels=labels)
        metrics["f1_score_macro"] = sk_metrics.f1_score(y, y_pred, average="macro", labels=labels)

    if y_probs is not None:
        metrics["log_loss"] = sk_metrics.log_loss(y, y_probs, labels=labels)

    return metrics


def _get_classifier_per_class_metrics_collection_df(y, y_pred, labels):
    per_class_metrics_list = []
    for positive_class_index, positive_class in enumerate(labels):
        (y_bin, y_pred_bin, _,) = _get_binary_sum_up_label_pred_prob(
            positive_class_index, positive_class, y, y_pred, None
        )

        per_class_metrics = {"positive_class": positive_class}
        per_class_metrics.update(_get_classifier_per_class_metrics(y_bin, y_pred_bin))
        per_class_metrics_list.append(per_class_metrics)

    return pd.DataFrame(per_class_metrics_list)


_Curve = namedtuple("_Curve", ["plot_fn", "plot_fn_args", "auc"])


def _gen_classifier_curve(
    is_binomial,
    y,
    y_probs,
    labels,
    curve_type,
):
    """
    Generate precision-recall curve or ROC curve for classifier.
    :param is_binomial: True if it is binary classifier otherwise False
    :param y: True label values
    :param y_probs: if binary classifer, the predicted probability for positive class.
                    if multiclass classiifer, the predicted probabilities for all classes.
    :param labels: The set of labels.
    :param curve_type: "pr" or "roc"
    :return: An instance of "_Curve" which includes attributes "plot_fn", "plot_fn_args", "auc".
    """
    if curve_type == "roc":

        def gen_line_x_y_label_fn(_y, _y_prob):
            fpr, tpr, _ = sk_metrics.roc_curve(_y, _y_prob)
            auc = sk_metrics.auc(fpr, tpr)
            return fpr, tpr, f"AUC={auc:.3f}"

        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
    elif curve_type == "pr":

        def gen_line_x_y_label_fn(_y, _y_prob):
            precision, recall, _thresholds = sk_metrics.precision_recall_curve(_y, _y_prob)
            ap = np.mean(precision)
            return recall, precision, f"AP={ap:.3f}"

        xlabel = "recall"
        ylabel = "precision"
    else:
        assert False, "illegal curve type"

    if is_binomial:
        x_data, y_data, line_label = gen_line_x_y_label_fn(y, y_probs)
        data_series = [(line_label, x_data, y_data)]
        auc = sk_metrics.auc(x_data, y_data)
    else:
        curve_list = []
        for positive_class_index, positive_class in enumerate(labels):
            y_bin, _, y_prob_bin = _get_binary_sum_up_label_pred_prob(
                positive_class_index, positive_class, y, None, y_probs
            )

            x_data, y_data, line_label = gen_line_x_y_label_fn(y_bin, y_prob_bin)
            curve_list.append((positive_class, x_data, y_data, line_label))

        data_series = [
            (f"label={positive_class},{line_label}", x_data, y_data)
            for positive_class, x_data, y_data, line_label in curve_list
        ]
        auc = [sk_metrics.auc(x_data, y_data) for _, x_data, y_data, _ in curve_list]

    def _do_plot(**kwargs):
        import matplotlib.pyplot as pyplot

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
        },
        auc=auc,
    )


_matplotlib_config = {
    "figure.dpi": 288,
    "figure.figsize": [6.0, 4.0],
}


# pylint: disable=attribute-defined-outside-init
class DefaultEvaluator(ModelEvaluator):
    # pylint: disable=unused-argument
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs):
        return model_type in ["classifier", "regressor"]

    def _log_metrics(self):
        """
        Helper method to log metrics into specified run.
        """
        timestamp = int(time.time() * 1000)
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(
                    key=_gen_log_key(key, self.dataset_name),
                    value=value,
                    timestamp=timestamp,
                    step=0,
                )
                for key, value in self.metrics.items()
            ],
        )

    def _log_image_artifact(
        self,
        do_plot,
        artifact_name,
    ):
        import matplotlib.pyplot as pyplot

        artifact_file_name = _gen_log_key(artifact_name, self.dataset_name) + ".png"
        artifact_file_local_path = self.temp_dir.path(artifact_file_name)

        try:
            pyplot.clf()
            do_plot()
            pyplot.savefig(artifact_file_local_path)
        finally:
            pyplot.close(pyplot.gcf())

        mlflow.log_artifact(artifact_file_local_path)
        artifact = ImageEvaluationArtifact(uri=mlflow.get_artifact_uri(artifact_file_name))
        artifact.load(artifact_file_local_path)
        self.artifacts[artifact_name] = artifact

    def _log_pandas_df_artifact(self, pandas_df, artifact_name):
        artifact_file_name = _gen_log_key(artifact_name, self.dataset_name) + ".csv"
        artifact_file_local_path = self.temp_dir.path(artifact_file_name)
        pandas_df.to_csv(artifact_file_local_path, index=False)
        mlflow.log_artifact(artifact_file_local_path)
        artifact = CsvEvaluationArtifact(
            uri=mlflow.get_artifact_uri(artifact_file_name),
            content=pandas_df,
        )
        artifact.load(artifact_file_local_path)
        self.artifacts[artifact_name] = artifact

    def _log_model_explainability(self):
        if not self.evaluator_config.get("log_model_explainability", True):
            return

        if self.model_loader_module == "mlflow.spark":
            # TODO: Shap explainer need to manipulate on each feature values,
            #  but spark model input dataframe contains Vector type feature column
            #  which shap explainer does not support.
            #  To support this, we need expand the Vector type feature column into
            #  multiple scaler feature columns and pass it to shap explainer.
            _logger.warning(
                "Logging model explainability insights is not currently supported for PySpark "
                "models."
            )
            return

        if self.model_type == "classifier" and not all(
            [isinstance(label, (numbers.Number, np.bool_)) for label in self.label_list]
        ):
            _logger.warning(
                "Skip logging model explainability insights because it requires all label "
                "values to be Number type."
            )
            return

        try:
            import shap
            import matplotlib.pyplot as pyplot
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

        is_multinomial_classifier = self.model_type == "classifier" and self.num_classes > 2

        sample_rows = self.evaluator_config.get(
            "explainability_nsamples", _DEFAULT_SAMPLE_ROWS_FOR_SHAP
        )
        algorithm = self.evaluator_config.get("explainability_algorithm", None)

        truncated_feature_names = [truncate_str_from_middle(f, 20) for f in self.feature_names]
        for i, truncated_name in enumerate(truncated_feature_names):
            if truncated_name != self.feature_names[i]:
                # For duplicated truncated name, attach "(f_{feature_index})" at the end
                truncated_feature_names[i] = f"{truncated_name}(f_{i + 1})"

        truncated_feature_name_map = {
            f: f2 for f, f2 in zip(self.feature_names, truncated_feature_names)
        }

        sampled_X = shap.sample(self.X, sample_rows)

        if isinstance(sampled_X, pd.DataFrame):
            # For some shap explainer, the plot will use the DataFrame column names instead of
            # using feature_names argument value. So rename the dataframe column names.
            sampled_X = sampled_X.rename(columns=truncated_feature_name_map, copy=False)

        if algorithm:
            supported_algos = ["exact", "permutation", "partition"]
            if algorithm not in supported_algos:
                raise ValueError(
                    f"Specified explainer algorithm {algorithm} is unsupported. Currently only "
                    f"support {','.join(supported_algos)} algorithms."
                )
            explainer = shap.Explainer(
                self.predict_fn,
                sampled_X,
                feature_names=truncated_feature_names,
                algorithm=algorithm,
            )
        else:
            if self.raw_model and not is_multinomial_classifier:
                # For mulitnomial classifier, shap.Explainer may choose Tree/Linear explainer for
                # raw model, this case shap plot doesn't support it well, so exclude the
                # multinomial_classifier case here.
                explainer = shap.Explainer(
                    self.raw_model, sampled_X, feature_names=truncated_feature_names
                )
            else:
                # fallback to default explainer
                explainer = shap.Explainer(
                    self.predict_fn, sampled_X, feature_names=truncated_feature_names
                )

        _logger.info(f"Shap explainer {explainer.__class__.__name__} is used.")

        shap_values = explainer(sampled_X)

        try:
            mlflow.shap.log_explainer(
                explainer, artifact_path=_gen_log_key("explainer", self.dataset_name)
            )
        except Exception as e:
            # TODO: The explainer saver is buggy, if `get_underlying_model_flavor` return "unknown",
            #   then fallback to shap explainer saver, and shap explainer will call `model.save`
            #   for sklearn model, there is no `.save` method, so error will happen.
            _logger.warning(f"Log explainer failed. Reason: {str(e)}")

        def plot_beeswarm():
            pyplot.subplots_adjust(bottom=0.2, left=0.4)
            shap.plots.beeswarm(shap_values, show=False)

        self._log_image_artifact(
            plot_beeswarm,
            "shap_beeswarm_plot",
        )

        def plot_summary():
            pyplot.subplots_adjust(bottom=0.2, left=0.4)
            shap.summary_plot(shap_values, show=False)

        self._log_image_artifact(
            plot_summary,
            "shap_summary_plot",
        )

        def plot_feature_importance():
            pyplot.subplots_adjust(bottom=0.2, left=0.4)
            shap.plots.bar(shap_values, show=False)

        self._log_image_artifact(
            plot_feature_importance,
            "shap_feature_importance_plot",
        )

    def _log_binary_classifier(self):
        self.metrics.update(_get_classifier_per_class_metrics(self.y, self.y_pred))

        if self.y_probs is not None:
            roc_curve = _gen_classifier_curve(
                is_binomial=True,
                y=self.y,
                y_probs=self.y_prob,
                labels=self.label_list,
                curve_type="roc",
            )

            def plot_roc_curve():
                roc_curve.plot_fn(**roc_curve.plot_fn_args)

            self._log_image_artifact(plot_roc_curve, "roc_curve_plot")
            self.metrics["roc_auc"] = roc_curve.auc

            pr_curve = _gen_classifier_curve(
                is_binomial=True,
                y=self.y,
                y_probs=self.y_prob,
                labels=self.label_list,
                curve_type="pr",
            )

            def plot_pr_curve():
                pr_curve.plot_fn(**pr_curve.plot_fn_args)

            self._log_image_artifact(plot_pr_curve, "precision_recall_curve_plot")
            self.metrics["precision_recall_auc"] = pr_curve.auc

    def _log_multiclass_classifier(self):
        per_class_metrics_collection_df = _get_classifier_per_class_metrics_collection_df(
            self.y, self.y_pred, self.label_list
        )

        log_roc_pr_curve = False
        if self.y_probs is not None:
            max_num_classes_for_logging_curve = self.evaluator_config.get(
                "max_num_classes_threshold_logging_roc_pr_curve_for_multiclass_classifier", 10
            )
            if self.num_classes <= max_num_classes_for_logging_curve:
                log_roc_pr_curve = True
            else:
                _logger.warning(
                    f"The classifier num_classes > {max_num_classes_for_logging_curve}, skip "
                    f"logging ROC curve and Precision-Recall curve. You can add evaluator config "
                    f"'max_num_classes_threshold_logging_roc_pr_curve_for_multiclass_classifier' "
                    f"to increase the threshold."
                )

        if log_roc_pr_curve:
            roc_curve = _gen_classifier_curve(
                is_binomial=False,
                y=self.y,
                y_probs=self.y_probs,
                labels=self.label_list,
                curve_type="roc",
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
                curve_type="pr",
            )

            def plot_pr_curve():
                pr_curve.plot_fn(**pr_curve.plot_fn_args)

            self._log_image_artifact(plot_pr_curve, "precision_recall_curve_plot")
            per_class_metrics_collection_df["precision_recall_auc"] = pr_curve.auc

        self._log_pandas_df_artifact(per_class_metrics_collection_df, "per_class_metrics")

    def _evaluate_classifier(self):
        from mlflow.models.evaluation.lift_curve import plot_lift_curve

        self.label_list = np.unique(self.y)
        self.num_classes = len(self.label_list)

        self.y_pred = self.predict_fn(self.X)
        self.is_binomial = self.num_classes <= 2

        if self.is_binomial:
            if list(self.label_list) not in [[0, 1], [-1, 1]]:
                raise ValueError(
                    "Binary classifier evaluation dataset positive class label must be 1 or True, "
                    "negative class label must be 0 or -1 or False, and dataset must contains "
                    "both positive and negative examples."
                )
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
            self.y_probs = self.predict_proba_fn(self.X)
            if self.is_binomial:
                self.y_prob = self.y_probs[:, 1]
            else:
                self.y_prob = None
        else:
            self.y_probs = None
            self.y_prob = None

        self.metrics.update(
            _get_classifier_global_metrics(
                self.is_binomial, self.y, self.y_pred, self.y_probs, self.label_list
            )
        )

        if self.is_binomial:
            self._log_binary_classifier()
        else:
            self._log_multiclass_classifier()

        if self.is_binomial and self.y_probs is not None:
            self._log_image_artifact(
                lambda: plot_lift_curve(self.y, self.y_probs),
                "lift_curve_plot",
            )

        # normalize the confusion matrix, keep consistent with sklearn autologging.
        confusion_matrix = sk_metrics.confusion_matrix(
            self.y, self.y_pred, labels=self.label_list, normalize="true"
        )

        def plot_confusion_matrix():
            sk_metrics.ConfusionMatrixDisplay(
                confusion_matrix=confusion_matrix,
                display_labels=self.label_list,
            ).plot(cmap="Blues")

        if hasattr(sk_metrics, "ConfusionMatrixDisplay"):
            self._log_image_artifact(
                plot_confusion_matrix,
                "confusion_matrix",
            )

        self._log_metrics()
        self._log_model_explainability()
        return EvaluationResult(self.metrics, self.artifacts)

    def _evaluate_regressor(self):
        self.y_pred = self.model.predict(self.X)
        self.metrics.update(_get_regressor_metrics(self.y, self.y_pred))

        self._log_metrics()
        self._log_model_explainability()
        return EvaluationResult(self.metrics, self.artifacts)

    def evaluate(
        self,
        *,
        model: "mlflow.pyfunc.PyFuncModel",
        model_type,
        dataset,
        run_id,
        evaluator_config,
        **kwargs,
    ):
        import matplotlib

        with TempDir() as temp_dir, matplotlib.rc_context(_matplotlib_config):
            self.client = mlflow.tracking.MlflowClient()

            self.temp_dir = temp_dir
            self.model = model
            self.model_type = model_type
            self.dataset = dataset
            self.run_id = run_id
            self.evaluator_config = evaluator_config
            self.dataset_name = dataset.name
            self.feature_names = dataset.feature_names

            (
                model_loader_module,
                raw_model,
                predict_fn,
                predict_proba_fn,
            ) = _extract_raw_model_and_predict_fn(model)
            self.model_loader_module = model_loader_module
            self.raw_model = raw_model
            self.predict_fn = predict_fn
            self.predict_proba_fn = predict_proba_fn

            self.X = dataset.features_data
            self.y = dataset.labels_data
            self.metrics = EvaluationMetrics()
            self.artifacts = {}

            infered_model_type = _infer_model_type_by_labels(self.y)

            if model_type != infered_model_type:
                _logger.warning(
                    f"According to the evaluation dataset label values, the model type looks like "
                    f"{infered_model_type}, but you specified model type {model_type}. Please "
                    f"verify that you set the `model_type` and `dataset` arguments correctly."
                )

            if model_type == "classifier":
                return self._evaluate_classifier()
            elif model_type == "regressor":
                return self._evaluate_regressor()
            else:
                raise ValueError(f"Unsupported model type {model_type}")
