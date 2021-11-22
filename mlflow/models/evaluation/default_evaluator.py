import mlflow
from mlflow.models.evaluation.base import (
    ModelEvaluator,
    EvaluationMetrics,
    EvaluationArtifact,
)
from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import get_artifact_uri
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as pyplot
import scikitplot
import shap
import math

shap.initjs()

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


class ImageEvaluationArtifact(EvaluationArtifact):

    @classmethod
    def save_content_to_file(cls, content, output_artifact_path):
        assert isinstance(content, Image)
        content.save(output_artifact_path)

    @classmethod
    def load_content_from_file(cls, local_artifact_path):
        return open_image(local_artifact_path)


class DefaultEvaluator(ModelEvaluator):

    def can_evaluate(self, model_type, evaluator_config=None, **kwargs):
        return model_type in ["classifier", "regressor"]

    def _log_image_artifact(self, artifacts, temp_dir, do_plot, run_id, artifact_name, dataset_name):
        pyplot.clf()
        do_plot()
        artifact_file_name = f"{artifact_name}_on_{dataset_name}.png"
        artifact_file_local_path = temp_dir.path(artifact_file_name)
        pyplot.savefig(artifact_file_local_path)
        mlflow.log_artifact(artifact_file_local_path, artifact_file_name)
        artifacts[artifact_file_name] = \
            ImageEvaluationArtifact(
                location=get_artifact_uri(run_id, artifact_file_name),
                content=ImageEvaluationArtifact.load_content_from_file(artifact_file_local_path)
            )

    def _log_model_explainality(
        self, artifacts, temp_dir, model, X, dataset_name, run_id
    ):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        def plot_summary():
            shap.plots.beeswarm(shap_values, show=False)

        self._log_image_artifact(artifacts, temp_dir, plot_summary, run_id, 'shap_summary', dataset_name)

        def plot_feature_importance():
            shap.plots.bar(shap_values, show=False)

        self._log_image_artifact(
            artifacts, temp_dir, plot_feature_importance, run_id, 'shap_feature_importance', dataset_name
        )

    def _classifier_compute_metrics_and_compute_and_log_artifacts(
        self, temp_dir, model, X, y, dataset_name, run_id, evaluator_config
    ):
        # TODO: how to get num_classes without extra config ?
        num_classes = evaluator_config['num_classes']
        y_pred = model.predict(X)

        is_binomial = (num_classes <= 2)

        metrics = EvaluationMetrics()
        artifacts = {}
        metrics['accuracy'] = sk_metrics.accuracy_score(y, y_pred)
        metrics['example_count'] = len(X)

        # TODO: sum/mean on what data ?
        #  [P0] Sum: Computes the (weighted) sum of the given values.
        #  [P0] Mean: Computes the (weighted) mean of the given values.

        if is_binomial:
            if hasattr(model, 'predict_proba'):
                y_probs = model.predict_proba(X)
                y_prob = y_probs[:, 1]
            else:
                y_probs = None
                y_prob = None

            confusion_matrix = sk_metrics.confusion_matrix(y, y_pred)
            tn, fp, fn, tp = confusion_matrix.ravel()
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['true_positives'] = tp
            metrics['recall'] = sk_metrics.recall_score(y, y_pred)
            metrics['precision'] = sk_metrics.precision_score(y, y_pred)
            metrics['f1_score'] = sk_metrics.f1_score(y, y_pred)

            # TODO:
            #  compute hinge loss, this requires calling decision_function of the model
            #  e.g., see https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function

            if y_probs is not None:
                metrics['roc_auc'] = sk_metrics.roc_auc_score(y, y_prob)
                fpr, tpr, thresholds = sk_metrics.roc_curve(y, y_prob)
                roc_auc = sk_metrics.auc(fpr, tpr)
                metrics['precision_recall_auc'] = roc_auc

                def plot_roc_curve():
                    sk_metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                            estimator_name='example estimator').plot()

                self._log_image_artifact(artifacts, temp_dir, plot_roc_curve, run_id, 'roc_curve', dataset_name)

                def plot_lift_curve():
                    scikitplot.metrics.plot_lift_curve(y, y_probs)

                self._log_image_artifact(artifacts, temp_dir, plot_lift_curve, run_id, 'lift_curve', dataset_name)

            def plot_confusion_matrix():
                sk_metrics.ConfusionMatrixDisplay.from_predictions(y, y_pred)

            self._log_image_artifact(artifacts, temp_dir, plot_confusion_matrix, run_id, 'confusion_matrix', dataset_name)

            self._log_model_explainality(artifacts, temp_dir, model, X, dataset_name, run_id)

        return metrics, artifacts

    def _regressor_compute_metrics_and_compute_and_log_artifacts(
        self, temp_dir, model, X, y, dataset_name, run_id, evaluator_config
    ):
        metrics = EvaluationMetrics()
        artifacts = {}
        y_pred = model.predict(X)
        metrics['example_count'] = len(X)
        metrics['mean_absolute_error'] = sk_metrics.mean_absolute_error(y, y_pred)
        metrics['mean_square_error'] = sk_metrics.mean_squared_error(y, y_pred)
        metrics['root_mean_square_error'] = math.sqrt(metrics['mean_square_error'])
        self._log_model_explainality(artifacts, temp_dir, model, X, dataset_name, run_id)

    def compute_metrics_and_compute_and_log_artifacts(
        self, model, model_type, dataset, evaluator_config, run_id
    ):
        with TempDir() as temp_dir:
            X, y = dataset._extract_features_and_labels()
            if model_type == "classifier":
                return self._classifier_compute_metrics_and_compute_and_log_artifacts(
                    temp_dir, model, X, y, dataset.name, run_id, evaluator_config
                )
            elif model_type == "regressor":
                return self._regressor_compute_metrics_and_compute_and_log_artifacts(
                    temp_dir, model, X, y, dataset.name, run_id, evaluator_config
                )
            else:
                raise ValueError(f"Unsupported model type {model_type}")
