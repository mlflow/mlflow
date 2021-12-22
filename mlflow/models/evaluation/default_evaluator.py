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

from sklearn import metrics as sk_metrics
import math
import pandas as pd
import numpy as np
import json
import time
from functools import partial
import logging

"""
[P0] Accuracy: Calculates how often predictions equal labels.
[P0] BinaryCrossentropy: Computes the crossentropy metric between the labels and predictions.
[P0] Hinge: Computes the hinge metric between y_true and y_pred.
[P0] Sum: Computes the (weighted) sum of the given values.
[P0] Mean: Computes the (weighted) mean of the given values.
[P0] ExampleCount: Computes the total number of evaluation examples.
[P0] MeanAbsoluteError: Computes the mean absolute error between the labels and predictions.
[P0] MeanSquaredError: Computes the mean squared error between y_true and y_pred.
[P0] RootMeanSquaredError: Computes root mean squared error metric between y_true and y_pred.

[P0] TrueNegatives: Calculates the number of true negatives.
[P0] TruePositives: Calculates the number of true positives.
[P0] FalseNegatives: Calculates the number of false negatives.
[P0] FalsePositives: Calculates the number of false positives.
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix

[P0] Precision: Computes the precision of the predictions with respect to the labels.
[P0] Recall: Computes the recall of the predictions with respect to the labels.
[P0] AUC: Approximates the AUC (Area under the curve) of the ROC or PR curves.
[P0] F1 Score: 2*precision*recall / (precision+recall)

[P0] BinaryClassConfusionMatrix

Plots
[P0] Confusion matrix
[P0] Interactive ROC curve with metrics (TP/TN/FP/FN/Acc/F1/AUC), binary classification
[P0] Lift chart

Global explainability
[P0] Model built-in feature importance (supported models)
[P0] SHAP explainers
     [P0] Summary plot
"""

from PIL.Image import Image, open as open_image


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


_shap_initialized = False


def _infer_model_type_by_labels(labels):
    distinct_labels = set(labels)
    for v in distinct_labels:
        if v < 0 or not float(v).is_integer():
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
                'Log model explainability currently does not support spark model.'
            )
            return

        try:
            global _shap_initialized
            import shap
            import shap.maskers
            from IPython.core.display import display, HTML

            if not _shap_initialized:
                # Call `shap.getjs` instead of call `shap.initjs` to prevent
                # display a logo picture in IPython notebook.
                display(HTML(shap.getjs()))
                _shap_initialized = True
        except ImportError:
            _logger.warning('Shap package is not installed. Skip log model explainability.')
            return

        import matplotlib.pyplot as pyplot

        is_multinomial_classifier = self.model_type == 'classifier' and self.num_classes > 2

        sample_rows = self.evaluator_config.get('explainability_nsamples', _DEFAULT_SAMPLE_ROWS_FOR_SHAP)
        algorithm = self.evaluator_config.get('explainability_algorithm', None)

        truncated_feature_names = [truncate_str_from_middle(f, 20) for f in self.feature_names]
        for i, truncated_name in enumerate(truncated_feature_names):
            if truncated_name != self.feature_names[i]:
                # For truncated name, attach "(f_{feature_index})" at the end
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

    def _evaluate_per_class(self, positive_class, y, y_pred, y_proba):
        """
        if positive_class is an interger, generate metrics and artifacts on this class vs. rest,
         and the y/y_pred/y_proba must be sum up to a binary "is class" and "is not class"
        if positive_class is None, generate metrics and artifacts on binary y/y_pred/y_proba
        """

        def _gen_metric_name(name):
            if positive_class is not None:
                return f"class_{positive_class}_{name}"
            else:
                return name

        confusion_matrix = sk_metrics.confusion_matrix(y, y_pred)
        tn, fp, fn, tp = confusion_matrix.ravel()
        self.metrics[_gen_metric_name("true_negatives")] = tn
        self.metrics[_gen_metric_name("false_positives")] = fp
        self.metrics[_gen_metric_name("false_negatives")] = fn
        self.metrics[_gen_metric_name("true_positives")] = tp
        self.metrics[_gen_metric_name("recall")] = sk_metrics.recall_score(y, y_pred)
        self.metrics[_gen_metric_name("precision")] = sk_metrics.precision_score(y, y_pred)
        self.metrics[_gen_metric_name("f1_score")] = sk_metrics.f1_score(y, y_pred)

        if y_proba is not None:
            fpr, tpr, thresholds = sk_metrics.roc_curve(y, y_proba)
            roc_curve_pandas_df = pd.DataFrame(
                {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
            )
            self._log_pandas_df_artifact(
                roc_curve_pandas_df, _gen_metric_name("roc_curve_data"),
            )

            roc_auc = sk_metrics.auc(fpr, tpr)
            self.metrics[_gen_metric_name("roc_auc")] = roc_auc

            def plot_roc_curve():
                sk_metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()

            if hasattr(sk_metrics, 'RocCurveDisplay'):
                self._log_image_artifact(plot_roc_curve, _gen_metric_name("roc_curve_plot"))

            precision, recall, thresholds = \
                sk_metrics.precision_recall_curve(y, y_proba)
            thresholds = np.append(thresholds, [1.0], axis=0)
            pr_curve_pandas_df = pd.DataFrame(
                {"precision": precision, "recall": recall, "thresholds": thresholds}
            )
            self._log_pandas_df_artifact(
                pr_curve_pandas_df,
                _gen_metric_name("precision_recall_curve_data"),
            )

            pr_auc = sk_metrics.auc(recall, precision)
            self.metrics[_gen_metric_name("precision_recall_auc")] = pr_auc

            def plot_pr_curve():
                sk_metrics.PrecisionRecallDisplay(precision, recall).plot()

            if hasattr(sk_metrics, 'PrecisionRecallDisplay'):
                self._log_image_artifact(
                    plot_pr_curve,
                    _gen_metric_name("precision_recall_curve_plot"),
                )

    def _evaluate_classifier(self):
        from mlflow.models.evaluation.lift_curve import plot_lift_curve

        # Note: require labels to be number of 0, 1, 2, .. num_classes - 1
        label_list = sorted(list(set(self.y)))
        assert label_list[0] == 0, "Label values must being at '0'."
        self.num_classes = len(label_list)

        self.y_pred = self.predict_fn(self.X)
        self.is_binomial = self.num_classes <= 2

        self.metrics["accuracy"] = sk_metrics.accuracy_score(self.y, self.y_pred)
        self.metrics["example_count"] = len(self.X)

        if self.predict_proba_fn is not None:
            self.y_probs = self.predict_proba_fn(self.X)
            if self.is_binomial:
                self.y_prob = self.y_probs[:, 1]
            else:
                self.y_prob = None
        else:
            self.y_probs = None
            self.y_prob = None

        if self.predict_proba_fn is not None:
            self.metrics['log_loss'] = sk_metrics.log_loss(self.y, self.y_probs)

            if self.is_binomial:
                self._evaluate_per_class(None, self.y, self.y_pred, self.y_prob)
                self._log_image_artifact(
                    lambda: plot_lift_curve(self.y, self.y_probs), "lift_curve_plot",
                )
            else:
                self.metrics['f1_score_micro'] = \
                    sk_metrics.f1_score(self.y, self.y_pred, average='micro')
                self.metrics['f1_score_macro'] = \
                    sk_metrics.f1_score(self.y, self.y_pred, average='macro')
                for postive_class in range(self.num_classes):
                    y_per_class = np.where(self.y == postive_class, 1, 0)
                    y_pred_per_class = np.where(self.y_pred == postive_class, 1, 0)
                    pos_class_prob = self.y_probs[:, postive_class]
                    self._evaluate_per_class(
                        postive_class, y_per_class, y_pred_per_class, pos_class_prob
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
        self.metrics["example_count"] = len(self.X)
        self.metrics["mean_absolute_error"] = sk_metrics.mean_absolute_error(self.y, self.y_pred)
        self.metrics["mean_squared_error"] = sk_metrics.mean_squared_error(self.y, self.y_pred)
        self.metrics["root_mean_squared_error"] = math.sqrt(self.metrics["mean_squared_error"])
        self.metrics['sum_on_label'] = sum(self.y)
        self.metrics['mean_on_label'] = self.metrics['sum_on_label'] / self.metrics["example_count"]
        self.metrics['r2_score'] = sk_metrics.r2_score(self.y, self.y_pred)
        self.metrics['max_error'] = sk_metrics.max_error(self.y, self.y_pred)
        self.metrics['mean_absolute_percentage_error'] = \
            sk_metrics.mean_absolute_percentage_error(self.y, self.y_pred)

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
                    f"check you set model type or evaluation dataset correctly."
                )

            if model_type == "classifier":
                return self._evaluate_classifier()
            elif model_type == "regressor":
                return self._evaluate_regressor()
            else:
                raise ValueError(f"Unsupported model type {model_type}")
