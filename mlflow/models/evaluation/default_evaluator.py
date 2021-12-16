import mlflow
from mlflow.models.evaluation.base import (
    ModelEvaluator,
    EvaluationMetrics,
    EvaluationArtifact,
    EvaluationResult,
)
from mlflow.entities.metric import Metric
from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import get_artifact_uri
from mlflow.utils.string_utils import truncate_str_from_middle

from sklearn import metrics as sk_metrics
import math
import pandas as pd
import numpy as np
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


class DefaultEvaluator(ModelEvaluator):
    def can_evaluate(self, model_type, evaluator_config=None, **kwargs):
        return model_type in ["classifier", "regressor"]

    @staticmethod
    def _gen_log_key(key, dataset_name):
        return f'{key}_on_data_{dataset_name}'

    def _log_metrics(self, run_id, metrics, dataset_name):
        """
        Helper method to log metrics into specified run.
        """
        client = mlflow.tracking.MlflowClient()
        timestamp = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[
                Metric(key=DefaultEvaluator._gen_log_key(key, dataset_name),
                       value=value, timestamp=timestamp, step=0)
                for key, value in metrics.items()
            ],
        )

    def _log_image_artifact(
        self, artifacts, temp_dir, do_plot, run_id, artifact_name, dataset_name,
    ):
        import matplotlib.pyplot as pyplot
        client = mlflow.tracking.MlflowClient()
        pyplot.clf()
        do_plot()
        artifact_file_name = DefaultEvaluator._gen_log_key(artifact_name, dataset_name) + '.png'
        artifact_file_local_path = temp_dir.path(artifact_file_name)
        pyplot.savefig(artifact_file_local_path)
        client.log_artifact(run_id, artifact_file_local_path)
        artifact = ImageEvaluationArtifact(uri=get_artifact_uri(run_id, artifact_file_name))
        artifact.load(artifact_file_local_path)
        artifacts[artifact_name] = artifact

    def _log_pandas_df_artifact(
        self, artifacts, temp_dir, pandas_df, run_id, artifact_name, dataset_name, model
    ):
        client = mlflow.tracking.MlflowClient()
        artifact_file_name = DefaultEvaluator._gen_log_key(artifact_name, dataset_name) + '.csv'
        artifact_file_local_path = temp_dir.path(artifact_file_name)
        pandas_df.to_csv(artifact_file_local_path, index=False)
        client.log_artifact(run_id, artifact_file_local_path)
        artifact = CsvEvaluationArtifact(
            uri=get_artifact_uri(run_id, artifact_file_name),
            content=pandas_df,
        )
        artifact.load(artifact_file_local_path)
        artifacts[artifact_name] = artifact

    def _log_model_explainability(
            self, artifacts, temp_dir, model, X, dataset_name, feature_names, run_id, evaluator_config,
            is_multinomial_classifier,
    ):
        if not evaluator_config.get('log_model_explainability', True):
            return

        try:
            global _shap_initialized
            import shap
            import shap.maskers

            if not _shap_initialized:
                shap.initjs()
                _shap_initialized = True
        except ImportError:
            _logger.warning('Shap package is not installed. Skip log model explainability.')
            return

        import matplotlib.pyplot as pyplot

        sample_rows = evaluator_config.get('explainability_nsamples', _DEFAULT_SAMPLE_ROWS_FOR_SHAP)
        algorithm = evaluator_config.get('explainability_algorithm', None)

        model_loader_module = model.metadata.flavors['python_function']["loader_module"]

        predict_fn = model.predict

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
            try:
                import xgboost
                if isinstance(raw_model, xgboost.XGBModel):
                    # Because shap evaluation will pass evaluation data in ndarray format
                    # (without feature names), if set validate_features=True it will raise error.
                    predict_fn = partial(predict_fn, validate_features=False)
            except ImportError:
                pass

        truncated_feature_names = [truncate_str_from_middle(f, 20) for f in feature_names]
        for i, truncated_name in enumerate(truncated_feature_names):
            if truncated_name != feature_names[i]:
                # For truncated name, attach "(f_{feature_index})" at the end
                truncated_feature_names[i] = f'{truncated_name}(f_{i})'

        truncated_feature_name_map = {f: f2 for f, f2 in zip(feature_names, truncated_feature_names)}
        if isinstance(X, pd.DataFrame):
            X = X.rename(columns=truncated_feature_name_map, copy=False)

        sampled_X = shap.sample(X, sample_rows)
        if algorithm:
            if algorithm == 'sampling':
                explainer = shap.explainers.Sampling(
                    predict_fn, X, feature_names=truncated_feature_names
                )
                shap_values = explainer(X, sample_rows)
            else:
                explainer = shap.Explainer(
                    predict_fn, sampled_X, feature_names=truncated_feature_names, algorithm=algorithm
                )
                shap_values = explainer(sampled_X)
        else:
            maskers = shap.maskers.Independent(sampled_X)
            if raw_model and not is_multinomial_classifier and \
                    shap.explainers.Linear.supports_model_with_masker(raw_model, maskers):
                explainer = shap.explainers.Linear(
                    raw_model, maskers, feature_names=truncated_feature_names
                )
                shap_values = explainer(sampled_X)
            elif raw_model and not is_multinomial_classifier and \
                    shap.explainers.Tree.supports_model_with_masker(raw_model, maskers):
                explainer = shap.explainers.Tree(
                    raw_model, maskers, feature_names=truncated_feature_names
                )
                shap_values = explainer(sampled_X)
            elif raw_model and shap.explainers.Additive.supports_model_with_masker(
                    raw_model, maskers
            ):
                explainer = shap.explainers.Additive(
                    raw_model, maskers, feature_names=truncated_feature_names
                )
                shap_values = explainer(sampled_X)
            else:
                # fallback to default explainer
                explainer = shap.Explainer(
                    predict_fn, sampled_X, feature_names=truncated_feature_names
                )
                shap_values = explainer(sampled_X)

        _logger.info(f'Shap explainer {explainer.__class__.__name__} is used.')

        # TODO: seems infer pip req fail when log_explainer.
        # TODO: The explainer saver is buggy, if `get_underlying_model_flavor` return "unknown",
        #   then fallback to shap explainer saver, and shap explainer will call `model.save`
        #   for sklearn model, there is no `.save` method, so error will happen.

        mlflow.shap.log_explainer(
            explainer,
            artifact_path=DefaultEvaluator._gen_log_key('explainer', dataset_name)
        )


        def plot_beeswarm():
            pyplot.subplots_adjust(bottom=0.2, left=0.4)
            shap.plots.beeswarm(shap_values, show=False)

        self._log_image_artifact(
            artifacts, temp_dir, plot_beeswarm, run_id, "shap_beeswarm", dataset_name,
        )

        def plot_summary():
            pyplot.subplots_adjust(bottom=0.2, left=0.4)
            shap.summary_plot(shap_values, show=False)

        self._log_image_artifact(
            artifacts, temp_dir, plot_summary, run_id, "shap_summary", dataset_name,
        )

        def plot_feature_importance():
            pyplot.subplots_adjust(bottom=0.2, left=0.4)
            shap.plots.bar(shap_values, show=False)

        self._log_image_artifact(
            artifacts,
            temp_dir,
            plot_feature_importance,
            run_id,
            "shap_feature_importance",
            dataset_name,
        )

    def _evaluate_classifier(self, temp_dir, model, X, y, dataset_name, feature_names, run_id, evaluator_config):
        from mlflow.models.evaluation.lift_curve import plot_lift_curve

        # Note: require labels to be number of 0, 1, 2, .. num_classes - 1
        label_list = sorted(list(set(y)))
        assert label_list[0] == 0, "Label values must being at '0'."
        num_classes = len(label_list)

        y_pred = model.predict(X)

        is_binomial = num_classes <= 2

        metrics = EvaluationMetrics()
        artifacts = {}
        metrics["accuracy"] = sk_metrics.accuracy_score(y, y_pred)
        metrics["example_count"] = len(X)

        # TODO: sum/mean on what data ?
        #  [P0] Sum: Computes the (weighted) sum of the given values.
        #  [P0] Mean: Computes the (weighted) mean of the given values.

        if is_binomial:
            if model.support_predict_proba():
                # TODO: for xgb disable feature names check
                y_probs = model.predict_proba(X)
                y_prob = y_probs[:, 1]
            else:
                y_probs = None
                y_prob = None

            confusion_matrix = sk_metrics.confusion_matrix(y, y_pred)
            tn, fp, fn, tp = confusion_matrix.ravel()
            metrics["true_negatives"] = tn
            metrics["false_positives"] = fp
            metrics["false_negatives"] = fn
            metrics["true_positives"] = tp
            metrics["recall"] = sk_metrics.recall_score(y, y_pred)
            metrics["precision"] = sk_metrics.precision_score(y, y_pred)
            metrics["f1_score"] = sk_metrics.f1_score(y, y_pred)

            # TODO:
            #  compute hinge loss, this requires calling decision_function of the model
            #  e.g., see https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function
            if y_probs is not None:
                fpr, tpr, thresholds = sk_metrics.roc_curve(y, y_prob)
                roc_curve_pandas_df = pd.DataFrame(
                    {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
                )
                self._log_pandas_df_artifact(
                    artifacts, temp_dir, roc_curve_pandas_df, run_id, "roc_curve", dataset_name, model,
                )

                roc_auc = sk_metrics.auc(fpr, tpr)
                metrics["roc_auc"] = roc_auc

                def plot_roc_curve():
                    sk_metrics.RocCurveDisplay(
                        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                    ).plot()

                if hasattr(sk_metrics, 'RocCurveDisplay'):
                    self._log_image_artifact(
                        artifacts, temp_dir, plot_roc_curve, run_id, "roc_curve", dataset_name,
                    )

                precision, recall, thresholds = sk_metrics.precision_recall_curve(y, y_prob)
                thresholds = np.append(thresholds, [1.0], axis=0)
                pr_curve_pandas_df = pd.DataFrame(
                    {"precision": precision, "recall": recall, "thresholds": thresholds}
                )
                self._log_pandas_df_artifact(
                    artifacts, temp_dir, pr_curve_pandas_df, run_id, "precision_recall_curve",
                    dataset_name, model,
                )

                pr_auc = sk_metrics.auc(recall, precision)
                metrics["precision_recall_auc"] = pr_auc

                def plot_pr_curve():
                    sk_metrics.PrecisionRecallDisplay(
                        precision, recall,
                    ).plot()

                if hasattr(sk_metrics, 'PrecisionRecallDisplay'):
                    self._log_image_artifact(
                        artifacts, temp_dir, plot_pr_curve, run_id, "precision_recall_curve", dataset_name,
                    )

                self._log_image_artifact(
                    artifacts, temp_dir,
                    lambda: plot_lift_curve(y, y_probs),
                    run_id, "lift_curve", dataset_name,
                )

            def plot_confusion_matrix():
                sk_metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix).plot()

            if hasattr(sk_metrics, 'ConfusionMatrixDisplay'):
                self._log_image_artifact(
                    artifacts, temp_dir, plot_confusion_matrix, run_id, "confusion_matrix", dataset_name,
                )

        self._log_metrics(run_id, metrics, dataset_name)

        self._log_model_explainability(
            artifacts, temp_dir, model, X, dataset_name, feature_names, run_id, evaluator_config,
            is_multinomial_classifier=(num_classes > 2)
        )

        return EvaluationResult(metrics, artifacts)

    def _evaluate_regressor(self, temp_dir, model, X, y, dataset_name, feature_names, run_id, evaluator_config):
        metrics = EvaluationMetrics()
        artifacts = {}
        y_pred = model.predict(X)
        metrics["example_count"] = len(X)
        metrics["mean_absolute_error"] = sk_metrics.mean_absolute_error(y, y_pred)
        metrics["mean_squared_error"] = sk_metrics.mean_squared_error(y, y_pred)
        metrics["root_mean_squared_error"] = math.sqrt(metrics["mean_squared_error"])
        metrics['sum_on_label'] = sum(y)
        metrics['mean_on_label'] = metrics['sum_on_label'] / metrics["example_count"]
        metrics['r2_score'] = sk_metrics.r2_score(y, y_pred)
        metrics['max_error'] = sk_metrics.max_error(y, y_pred)
        metrics['mean_absolute_percentage_error'] = \
            sk_metrics.mean_absolute_percentage_error(y, y_pred)

        self._log_model_explainability(
            artifacts, temp_dir, model, X, dataset_name, feature_names, run_id, evaluator_config,
            is_multinomial_classifier=False,
        )
        self._log_metrics(run_id, metrics, dataset_name)
        return EvaluationResult(metrics, artifacts)

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
            X, y = dataset._extract_features_and_labels()
            if model_type == "classifier":
                return self._evaluate_classifier(
                    temp_dir, model, X, y, dataset.name, dataset.feature_names, run_id, evaluator_config
                )
            elif model_type == "regressor":
                return self._evaluate_regressor(
                    temp_dir, model, X, y, dataset.name, dataset.feature_names, run_id, evaluator_config
                )
            else:
                raise ValueError(f"Unsupported model type {model_type}")
