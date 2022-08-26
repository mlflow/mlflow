from mlflow import MlflowClient
from mlflow.models.evaluation import (
    ModelEvaluator,
    EvaluationArtifact,
    EvaluationResult,
)
from mlflow.models.evaluation.artifacts import ImageEvaluationArtifact
from mlflow.tracking.artifact_utils import get_artifact_uri
from mlflow.entities import Metric
from sklearn import metrics as sk_metrics
import time
import pandas as pd
import io
from PIL import Image


class Array2DEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        pd.DataFrame(self._content).to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        pdf = pd.read_csv(local_artifact_path)
        return pdf.to_numpy()


# pylint: disable=attribute-defined-outside-init
class DummyEvaluator(ModelEvaluator):
    # pylint: disable=unused-argument
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs):
        return model_type in ["classifier", "regressor"]

    def _log_metrics(self, run_id, metrics, dataset_name):
        """
        Helper method to log metrics into specified run.
        """
        client = MlflowClient()
        timestamp = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[
                Metric(key=f"{key}_on_{dataset_name}", value=value, timestamp=timestamp, step=0)
                for key, value in metrics.items()
            ],
        )

    def _evaluate(self, y_pred, is_baseline_model=False):
        if self.model_type == "classifier":
            accuracy_score = sk_metrics.accuracy_score(self.y, y_pred)

            metrics = {"accuracy_score": accuracy_score}
            artifacts = {}
            if not is_baseline_model:
                self._log_metrics(self.run_id, metrics, self.dataset.name)
                confusion_matrix = sk_metrics.confusion_matrix(self.y, y_pred)
                confusion_matrix_artifact_name = f"confusion_matrix_on_{self.dataset.name}"
                confusion_matrix_artifact = Array2DEvaluationArtifact(
                    uri=get_artifact_uri(self.run_id, confusion_matrix_artifact_name + ".csv"),
                    content=confusion_matrix,
                )
                confusion_matrix_csv_buff = io.StringIO()
                confusion_matrix_artifact._save(confusion_matrix_csv_buff)
                if not is_baseline_model:
                    self.client.log_text(
                        self.run_id,
                        confusion_matrix_csv_buff.getvalue(),
                        confusion_matrix_artifact_name + ".csv",
                    )

                confusion_matrix_figure = sk_metrics.ConfusionMatrixDisplay.from_predictions(
                    self.y, y_pred
                ).figure_
                img_buf = io.BytesIO()
                confusion_matrix_figure.savefig(img_buf)
                img_buf.seek(0)
                confusion_matrix_image = Image.open(img_buf)

                confusion_matrix_image_artifact_name = (
                    f"confusion_matrix_image_on_{self.dataset.name}"
                )
                confusion_matrix_image_artifact = ImageEvaluationArtifact(
                    uri=get_artifact_uri(
                        self.run_id, confusion_matrix_image_artifact_name + ".png"
                    ),
                    content=confusion_matrix_image,
                )
                confusion_matrix_image_artifact._save(confusion_matrix_image_artifact_name + ".png")
                self.client.log_image(
                    self.run_id,
                    confusion_matrix_image,
                    confusion_matrix_image_artifact_name + ".png",
                )

                artifacts = {
                    confusion_matrix_artifact_name: confusion_matrix_artifact,
                    confusion_matrix_image_artifact_name: confusion_matrix_image_artifact,
                }
        elif self.model_type == "regressor":
            mean_absolute_error = sk_metrics.mean_absolute_error(self.y, y_pred)
            mean_squared_error = sk_metrics.mean_squared_error(self.y, y_pred)
            metrics = {
                "mean_absolute_error": mean_absolute_error,
                "mean_squared_error": mean_squared_error,
            }
            if not is_baseline_model:
                self._log_metrics(self.run_id, metrics, self.dataset.name)
            artifacts = {}
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    # pylint: disable=unused-argument
    def evaluate(
        self, *, model, model_type, dataset, run_id, evaluator_config, baseline_model=None, **kwargs
    ):
        self.model_type = model_type
        self.client = MlflowClient()
        self.dataset = dataset
        self.run_id = run_id
        self.X = dataset.features_data
        self.y = dataset.labels_data
        y_pred = model.predict(self.X)
        eval_result = self._evaluate(y_pred, is_baseline_model=False)

        if not baseline_model:
            return eval_result

        y_pred_baseline = baseline_model.predict(self.X)
        baseline_model_eval_result = self._evaluate(y_pred_baseline, is_baseline_model=True)
        return EvaluationResult(
            metrics=eval_result.metrics,
            artifacts=eval_result.artifacts,
            baseline_model_metrics=baseline_model_eval_result.metrics,
        )
