import mlflow
from mlflow.models.evaluation.base import (
    ModelEvaluator,
    EvaluationMetrics,
    EvaluationArtifact,
    EvaluationResult,
)
from mlflow.entities.metric import Metric
from mlflow.utils.file_utils import TempDir
from mlflow.utils.string_utils import truncate_str_from_middle
from mlflow.models.utils import plot_lines

from sklearn import metrics as sk_metrics
import math
from collections import namedtuple
import pandas as pd
import numpy as np
import json
import time
from functools import partial
import logging
from packaging.version import Version


from PIL.Image import open as open_image


_logger = logging.getLogger(__name__)


class ImageEvaluationArtifact(EvaluationArtifact):
    def save(self, output_artifact_path):
        self._content.save(output_artifact_path)

    def _load_content_from_file(self, local_artifact_path):
        self._content = open_image(local_artifact_path)
        return self._content


class CsvEvaluationArtifact(EvaluationArtifact):
    def save(self, output_artifact_path):
        self._content.to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = pd.read_csv(local_artifact_path)
        return self._content


_DEFAULT_SAMPLE_ROWS_FOR_SHAP = 2000


def _infer_model_type_by_labels(labels):
    distinct_labels = set(labels)
    for v in distinct_labels:
        if not float(v).is_integer():
            return "regressor"
    if len(distinct_labels) > 1000 and len(distinct_labels) / len(labels) > 0.7:
        return "regressor"
    return "classifier"


def _extract_raw_model_and_predict_fn(model):
    model_loader_module = model.metadata.flavors['python_function']["loader_module"]
    predict_fn = model.predict
    predict_proba_fn = None

    try:
        if model_loader_module == 'mlflow.sklearn':
            raw_model = model._model_impl
        elif model_loader_module == 'mlflow.lightgbm':
            raw_model = model._model_impl.lgb_model
        elif model_loader_module == 'mlflow.xgboost':
            raw_model = model._model_impl.xgb_model
        else:
            raw_model = None
    except Exception as e:
        raw_model = None
        _logger.warning(f'Raw model resolution fails unexpectedly on PyFuncModel {model!r}, '
                        f'error message is {e}')

    if raw_model:
        predict_fn = raw_model.predict
        predict_proba_fn = getattr(raw_model, 'predict_proba', None)

        try:
            import xgboost
            if isinstance(raw_model, xgboost.XGBModel):
                # Because shap evaluation will pass evaluation data in ndarray format
                # (without feature names), if set validate_features=True it will raise error.
                predict_fn = partial(predict_fn, validate_features=False)
                predict_proba_fn = partial(predict_proba_fn, validate_features=False)
        except ImportError:
            pass

    return model_loader_module, raw_model, predict_fn, predict_proba_fn


def _gen_log_key(key, dataset_name):
    return f'{key}_on_data_{dataset_name}'


def _get_regressor_metrics(y, y_pred):
    return {
        'example_count': len(y),
        'mean_absolute_error': sk_metrics.mean_absolute_error(y, y_pred),
        'mean_squared_error': sk_metrics.mean_squared_error(y, y_pred),
        'root_mean_squared_error': math.sqrt(sk_metrics.mean_squared_error(y, y_pred)),
        'sum_on_label': sum(y),
        'mean_on_label': sum(y) / len(y),
        'r2_score': sk_metrics.r2_score(y, y_pred),
        'max_error': sk_metrics.max_error(y, y_pred),
        'mean_absolute_percentage_error': sk_metrics.mean_absolute_percentage_error(y, y_pred)
    }


def _get_binary_sum_up_label_pred_prob(postive_class, y, y_pred, y_probs):
    y_is_positive = np.where(y == postive_class, 1, 0)
    y_pred_is_positive = np.where(y_pred == postive_class, 1, 0)

    if y_probs is not None:
        prob_of_positive = y_probs[:, postive_class]
    else:
        prob_of_positive = None

    return y_is_positive, y_pred_is_positive, prob_of_positive


def _get_classifier_per_class_metrics(y, y_pred):
    """
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


def _get_classifier_global_metrics(is_binomial, y, y_pred, y_probs):
    metrics = {}
    metrics["accuracy"] = sk_metrics.accuracy_score(y, y_pred)
    metrics["example_count"] = len(X)

    if not is_binomial:
        metrics['f1_score_micro'] = \
            sk_metrics.f1_score(y, y_pred, average='micro')
        metrics['f1_score_macro'] = \
            sk_metrics.f1_score(y, y_pred, average='macro')

    if y_probs is not None:
        metrics['log_loss'] = sk_metrics.log_loss(y, y_probs)

    return metrics


class DefaultEvaluator(ModelEvaluator):
    def can_evaluate(self, model_type, evaluator_config=None, **kwargs):
        return model_type in ["classifier", "regressor"]

    def _log_metrics(self):
        """
        Helper method to log metrics into specified run.
        """
        timestamp = int(time.time() * 1000)
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(key=_gen_log_key(key, self.dataset_name),
                       value=value, timestamp=timestamp, step=0)
                for key, value in self.metrics.items()
            ],
        )

    def _log_image_artifact(
        self, do_plot, artifact_name,
    ):
        import matplotlib.pyplot as pyplot
        artifact_file_name = _gen_log_key(artifact_name, self.dataset_name) + '.png'
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

    def _log_pandas_df_artifact(
        self, pandas_df, artifact_name
    ):
        artifact_file_name = _gen_log_key(artifact_name, self.dataset_name) + '.csv'
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
        if not self.evaluator_config.get('log_model_explainability', True):
            return

        if self.model_loader_module == 'mlflow.spark':
            # TODO: Shap explainer need to manipulate on each feature values,
            #  but spark model input dataframe contains Vector type feature column
            #  which shap explainer does not support.
            #  To support this, we need expand the Vector type feature column into
            #  multiple scaler feature columns and pass it to shap explainer.
            _logger.warning(
                'Logging model explainability insights is not currently supported for PySpark'
                ' models.'
            )
            return

        try:
            import shap
            import matplotlib.pyplot as pyplot
        except ImportError:
            _logger.warning(
                'SHAP or matplotlib package is not installed, so model explainability insights '
                'will not be logged.'
            )

        if Version(shap.__version__) < Version('0.40'):
            _logger.warning(
                'Shap package version is lower than 0.40, Skip log model explainability.'
            )
            return

        is_multinomial_classifier = self.model_type == 'classifier' and self.num_classes > 2

        sample_rows = self.evaluator_config.get('explainability_nsamples', _DEFAULT_SAMPLE_ROWS_FOR_SHAP)
        algorithm = self.evaluator_config.get('explainability_algorithm', None)

        truncated_feature_names = [truncate_str_from_middle(f, 20) for f in self.feature_names]
        for i, truncated_name in enumerate(truncated_feature_names):
            if truncated_name != self.feature_names[i]:
                # For duplicated truncated name, attach "(f_{feature_index})" at the end
                truncated_feature_names[i] = f'{truncated_name}(f_{i})'

        truncated_feature_name_map = {f: f2 for f, f2 in zip(self.feature_names, truncated_feature_names)}

        if isinstance(self.X, pd.DataFrame):
            # For some shap explainer, the plot will use the DataFrame column names instead of
            # using feature_names argument value. So rename the dataframe column names.
            renamed_X = self.X.rename(columns=truncated_feature_name_map, copy=False)
        else:
            renamed_X = self.X

        sampled_X = shap.sample(renamed_X, sample_rows)
        if algorithm:
            if algorithm == 'sampling':
                explainer = shap.explainers.Sampling(
                    self.predict_fn, renamed_X, feature_names=truncated_feature_names
                )
                shap_values = explainer(renamed_X, sample_rows)
            else:
                explainer = shap.Explainer(
                    self.predict_fn, sampled_X, feature_names=truncated_feature_names, algorithm=algorithm
                )
                shap_values = explainer(sampled_X)
        else:
            if self.raw_model and not is_multinomial_classifier:
                # For mulitnomial classifier, shap.Explainer may choose Tree/Linear explainer for
                # raw model, this case shap plot doesn't support it well, so exclude the
                # multinomial_classifier case here.
                explainer = shap.Explainer(self.raw_model, sampled_X, feature_names=truncated_feature_names)
                shap_values = explainer(sampled_X)
            else:
                # fallback to default explainer
                explainer = shap.Explainer(
                    self.predict_fn, sampled_X, feature_names=truncated_feature_names
                )
                shap_values = explainer(sampled_X)

        _logger.info(f'Shap explainer {explainer.__class__.__name__} is used.')

        try:
            mlflow.shap.log_explainer(
               explainer,
               artifact_path=_gen_log_key('explainer', self.dataset_name)
            )
        except Exception as e:
            # TODO: The explainer saver is buggy, if `get_underlying_model_flavor` return "unknown",
            #   then fallback to shap explainer saver, and shap explainer will call `model.save`
            #   for sklearn model, there is no `.save` method, so error will happen.
            _logger.warning(f'Log explainer failed. Reason: {str(e)}')

        def plot_beeswarm():
            pyplot.subplots_adjust(bottom=0.2, left=0.4)
            shap.plots.beeswarm(shap_values, show=False)

        self._log_image_artifact(
            plot_beeswarm, "shap_beeswarm_plot",
        )

        def plot_summary():
            pyplot.subplots_adjust(bottom=0.2, left=0.4)
            shap.summary_plot(shap_values, show=False)

        self._log_image_artifact(
            plot_summary, "shap_summary_plot",
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

        if self.y_prob is not None:
            fpr, tpr, thresholds = sk_metrics.roc_curve(self.y, self.y_prob)
            roc_curve_pandas_df = pd.DataFrame(
                {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
            )
            self._log_pandas_df_artifact(
                roc_curve_pandas_df, "roc_curve_data",
            )

            roc_auc = sk_metrics.auc(fpr, tpr)
            self.metrics["roc_auc"] = roc_auc

            def plot_roc_curve():
                # Do not use sklearn.metrics roc plot API because
                # older sklearn verison < 0.24 does not support.
                plot_lines(
                    {'roc': (fpr, tpr)},
                    xlabel='False Positive Rate', ylabel='True Positive Rate',
                    line_kwargs={"drawstyle": "steps-post"}
                )

            self._log_image_artifact(plot_roc_curve, "roc_curve_plot")

            precision, recall, thresholds = \
                sk_metrics.precision_recall_curve(self.y, self.y_prob)
            thresholds = np.append(thresholds, [1.0], axis=0)
            pr_curve_pandas_df = pd.DataFrame(
                {"precision": precision, "recall": recall, "thresholds": thresholds}
            )
            self._log_pandas_df_artifact(pr_curve_pandas_df, "precision_recall_curve_data")

            pr_auc = sk_metrics.auc(recall, precision)
            self.metrics["precision_recall_auc"] = pr_auc

            def plot_precision_recall_curve():
                # Do not use sklearn.metrics precision-recall plot API because
                # older sklearn verison < 0.24 does not support.
                plot_lines(
                    {'pr_curve': (recall, precision)}, xlabel='recall', ylabel='precision',
                    line_kwargs={"drawstyle": "steps-post"}
                )

            self._log_image_artifact(plot_precision_recall_curve, "precision_recall_curve_plot")

    def _log_multiclass_classifier(self):
        per_class_metrics_list = []
        per_class_roc_curve_data_list = []
        per_class_precision_recall_curve_data_list = []

        PerClassRocCurveData = namedtuple('PerClassRocCurveData', ['postive_class', 'fpr', 'tpr', 'thresholds'])
        PerClassPrecisionRecallCurveData = namedtuple(
            'PerClassPrecisionRecallCurveData', ['postive_class', 'precision', 'recall', 'thresholds']
        )
        log_roc_pr_curve = False
        if self.y_probs is not None:
            max_num_classes_for_logging_curve = \
                self.evaluator_config.get(
                    'max_num_classes_threshold_logging_roc_pr_curve_for_multiclass_classifier', 10
                )
            if self.num_classes <= max_num_classes_for_logging_curve:
                log_roc_pr_curve = True
            else:
                _logger.warning(f'The classifier num_classes > {max_num_classes_for_logging_curve}, skip logging '
                                f'ROC curve and Precision-Recall curve. You can add evaluator config '
                                f"'max_num_classes_threshold_logging_roc_pr_curve_for_multiclass_classifier' to "
                                f"increase the threshold.")

        for postive_class in self.label_list:
            y_is_positive, y_pred_is_positive, prob_of_positive = \
                _get_binary_sum_up_label_pred_prob(self.y, self.y_pred, self.y_probs)

            per_class_metrics = {'positive_class': postive_class}
            per_class_metrics_list.append(per_class_metrics)

            per_class_metrics.update(
                _get_classifier_per_class_metrics(y_is_positive, y_pred_is_positive)
            )

            if self.y_probs is not None:
                fpr, tpr, thresholds = sk_metrics.roc_curve(y_is_positive, prob_of_positive)
                if log_roc_pr_curve:
                    per_class_roc_curve_data_list.append(
                        PerClassRocCurveData(postive_class, fpr, tpr, thresholds)
                    )
                roc_auc = sk_metrics.auc(fpr, tpr)
                per_class_metrics["roc_auc"] = roc_auc

                precision, recall, thresholds = \
                    sk_metrics.precision_recall_curve(y_is_positive, prob_of_positive)
                thresholds = np.append(thresholds, [1.0], axis=0)
                if log_roc_pr_curve:
                    per_class_precision_recall_curve_data_list.append(
                        PerClassPrecisionRecallCurveData(postive_class, precision, recall, thresholds)
                    )
                pr_auc = sk_metrics.auc(recall, precision)
                per_class_metrics["precision_recall_auc"] = pr_auc

        per_class_metrics_pandas_df = pd.DataFrame(per_class_metrics_list)
        self._log_pandas_df_artifact(per_class_metrics_pandas_df, "per_class_metrics_data")

        if self.y_probs is not None and log_roc_pr_curve:
            per_class_roc_curve_pandas_df = pd.concat(
                [pd.DataFrame(item._asdict()) for item in per_class_roc_curve_data_list],
                ignore_index=True
            )
            self._log_pandas_df_artifact(per_class_roc_curve_pandas_df, "per_class_roc_curve_data")

            per_class_precision_recall_curve_pandas_df = pd.concat(
                [pd.DataFrame(item._asdict()) for item in per_class_precision_recall_curve_data_list],
                ignore_index=True
            )
            self._log_pandas_df_artifact(
                per_class_precision_recall_curve_pandas_df,
                "per_class_precision_recall_curve_data"
            )

            def plot_roc_curve():
                data_series = {
                    f'Positive Class = {postive_class}': (fpr, tpr)
                    for postive_class, fpr, tpr, _ in per_class_roc_curve_data_list
                }
                plot_lines(
                    data_series, xlabel='False Positive Rate', ylabel='True Positive Rate',
                    legend_loc='lower right',
                    line_kwargs={"drawstyle": "steps-post"}
                )

            def plot_precision_recall_curve():
                data_series = {
                    f'Positive Class = {postive_class}': (recall, precision)
                    for postive_class, precision, recall, _ in per_class_precision_recall_curve_data_list
                }
                plot_lines(
                    data_series, xlabel='recall', ylabel='precision',
                    legend_loc='lower left',
                    line_kwargs={"drawstyle": "steps-post"}
                )

            self._log_image_artifact(plot_roc_curve, "roc_curve_plot")
            self._log_image_artifact(plot_precision_recall_curve, "precision_recall_curve_plot")

    def _evaluate_classifier(self):
        from mlflow.models.evaluation.lift_curve import plot_lift_curve

        label_list = sorted(list(set(self.y)))
        self.label_list = label_list
        self.num_classes = len(label_list)

        self.y_pred = self.predict_fn(self.X)
        self.is_binomial = self.num_classes <= 2

        if self.is_binomial:
            for label in label_list:
                if int(label) not in [-1, 0, 1]:
                    raise ValueError(
                        'Binomial classification require evaluation dataset label values to be '
                        '-1, 0, or 1.'
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
            _get_classifier_global_metrics(self.is_binomial, self.y, self.y_pred, self.y_probs)
        )

        if self.is_binomial:
            self._log_binary_classifier()
        else:
            self._log_multiclass_classifier()

        if self.is_binomial and self.y_probs is not None:
            self._log_image_artifact(
                lambda: plot_lift_curve(self.y, self.y_probs), "lift_curve_plot",
            )

        # TODO: Shall we also log confusion_matrix data as a json artifact ?
        confusion_matrix = sk_metrics.confusion_matrix(self.y, self.y_pred)

        def plot_confusion_matrix():
            sk_metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix).plot()

        if hasattr(sk_metrics, 'ConfusionMatrixDisplay'):
            self._log_image_artifact(
                plot_confusion_matrix, "confusion_matrix",
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
        model: "mlflow.pyfunc.PyFuncModel",
        model_type,
        dataset,
        run_id,
        evaluator_config,
        **kwargs,
    ):
        with TempDir() as temp_dir:
            self.client = mlflow.tracking.MlflowClient()

            self.temp_dir = temp_dir
            self.model = model
            self.model_type = model_type
            self.dataset = dataset
            self.run_id = run_id
            self.evaluator_config = evaluator_config
            self.dataset_name = dataset.name
            self.feature_names = dataset.feature_names

            model_loader_module, raw_model, predict_fn, predict_proba_fn = \
                _extract_raw_model_and_predict_fn(model)
            self.model_loader_module = model_loader_module
            self.raw_model = raw_model
            self.predict_fn = predict_fn
            self.predict_proba_fn = predict_proba_fn

            X, y = dataset._extract_features_and_labels()
            self.X = X
            self.y = y
            self.metrics = EvaluationMetrics()
            self.artifacts = {}

            infered_model_type = _infer_model_type_by_labels(y)

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
