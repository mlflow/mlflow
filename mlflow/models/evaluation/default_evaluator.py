import mlflow
from mlflow.models.evaluation.base import (
    ModelEvaluator,
    EvaluationMetrics,
    EvaluationArtifact,
    EvaluationResult,
)
from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import get_artifact_uri
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as pyplot
import scikitplot
import shap
import math
import pandas as pd

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


class DefaultEvaluator(ModelEvaluator):
    def can_evaluate(self, model_type, evaluator_config=None, **kwargs):
        return model_type in ["classifier", "regressor"]

    def _log_image_artifact(
        self, artifacts, temp_dir, do_plot, run_id, artifact_name, dataset_name
    ):
        client = mlflow.tracking.MlflowClient()
        pyplot.clf()
        do_plot()
        artifact_file_name = f"{artifact_name}_on_{dataset_name}.png"
        artifact_file_local_path = temp_dir.path(artifact_file_name)
        pyplot.savefig(artifact_file_local_path)
        client.log_artifact(run_id, artifact_file_local_path)
        artifact = ImageEvaluationArtifact(uri=get_artifact_uri(run_id, artifact_file_name))
        artifact.load(artifact_file_local_path)
        artifacts[artifact_file_name] = artifact

    def _log_pandas_df_artifact(
        self, artifacts, temp_dir, pandas_df, run_id, artifact_name, dataset_name
    ):
        client = mlflow.tracking.MlflowClient()
        artifact_file_name = f"{artifact_name}_on_{dataset_name}.csv"
        artifact_file_local_path = temp_dir.path(artifact_file_name)
        pandas_df.to_csv(artifact_file_local_path, index=False)
        client.log_artifact(run_id, artifact_file_local_path)
        artifact = CsvEvaluationArtifact(
            uri=get_artifact_uri(run_id, artifact_file_name),
            content=pandas_df,
        )
        artifact.load(artifact_file_local_path)
        artifacts[artifact_file_name] = artifact

    def _log_model_explainality(self, artifacts, temp_dir, model, X, dataset_name, run_id):
        explainer = shap.Explainer(model._model_impl, X)
        shap_values = explainer(X)

        def plot_summary():
            shap.plots.beeswarm(shap_values, show=False)

        self._log_image_artifact(
            artifacts, temp_dir, plot_summary, run_id, "shap_summary", dataset_name
        )

        def plot_feature_importance():
            shap.plots.bar(shap_values, show=False)

        self._log_image_artifact(
            artifacts,
            temp_dir,
            plot_feature_importance,
            run_id,
            "shap_feature_importance",
            dataset_name,
        )

    def _evaluate_classifier(self, temp_dir, model, X, y, dataset_name, run_id, evaluator_config):
        # Note: require labels to be number of 0, 1, 2, .. num_classes - 1
        label_list = sorted(list(set(y)))
        assert label_list[0] >= 0, "Evaluation dataset labels must be positive integers."
        max_label = label_list[-1]
        num_classes = max_label + 1

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
            if hasattr(model, "predict_proba"):
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
                metrics["roc_auc"] = sk_metrics.roc_auc_score(y, y_prob)
                fpr, tpr, thresholds = sk_metrics.roc_curve(y, y_prob)
                roc_curve_pandas_df = pd.DataFrame(
                    {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
                )
                self._log_pandas_df_artifact(
                    artifacts, temp_dir, roc_curve_pandas_df, run_id, "roc_curve", dataset_name
                )

                roc_auc = sk_metrics.auc(fpr, tpr)
                metrics["precision_recall_auc"] = roc_auc

                def plot_roc_curve():
                    sk_metrics.RocCurveDisplay(
                        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="example estimator"
                    ).plot()

                self._log_image_artifact(
                    artifacts, temp_dir, plot_roc_curve, run_id, "roc_curve", dataset_name
                )

                def plot_lift_curve():
                    scikitplot.metrics.plot_lift_curve(y, y_probs)

                self._log_image_artifact(
                    artifacts, temp_dir, plot_lift_curve, run_id, "lift_curve", dataset_name
                )

            def plot_confusion_matrix():
                sk_metrics.ConfusionMatrixDisplay.from_predictions(y, y_pred, normalize='all')

            self._log_image_artifact(
                artifacts, temp_dir, plot_confusion_matrix, run_id, "confusion_matrix", dataset_name
            )

        self._log_metrics(run_id, metrics, dataset_name)
        self._log_model_explainality(artifacts, temp_dir, model, X, dataset_name, run_id)

        return EvaluationResult(metrics, artifacts)

    def _evaluate_regressor(self, temp_dir, model, X, y, dataset_name, run_id, evaluator_config):
        metrics = EvaluationMetrics()
        artifacts = {}
        y_pred = model.predict(X)
        metrics["example_count"] = len(X)
        metrics["mean_absolute_error"] = sk_metrics.mean_absolute_error(y, y_pred)
        metrics["mean_squared_error"] = sk_metrics.mean_squared_error(y, y_pred)
        metrics["root_mean_squared_error"] = math.sqrt(metrics["mean_squared_error"])
        self._log_model_explainality(artifacts, temp_dir, model, X, dataset_name, run_id)
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
                    temp_dir, model, X, y, dataset.name, run_id, evaluator_config
                )
            elif model_type == "regressor":
                return self._evaluate_regressor(
                    temp_dir, model, X, y, dataset.name, run_id, evaluator_config
                )
            else:
                raise ValueError(f"Unsupported model type {model_type}")
