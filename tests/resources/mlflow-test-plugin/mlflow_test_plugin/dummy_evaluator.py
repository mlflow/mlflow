import io
import os
import tempfile

import pandas as pd
from PIL import Image
from sklearn import metrics as sk_metrics

from mlflow import MlflowClient
from mlflow.entities import Metric
from mlflow.models.evaluation import (
    EvaluationArtifact,
    EvaluationResult,
    ModelEvaluator,
)
from mlflow.models.evaluation.artifacts import ImageEvaluationArtifact
from mlflow.tracking.artifact_utils import get_artifact_uri
from mlflow.utils.time import get_current_time_millis


class Array2DEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        pd.DataFrame(self._content).to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        pdf = pd.read_csv(local_artifact_path)
        return pdf.to_numpy()


class DummyEvaluator(ModelEvaluator):
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs):
        return model_type in ["classifier", "regressor"]

    def _log_metrics(self, run_id, metrics):
        """
        Helper method to log metrics into specified run.
        """
        client = MlflowClient()
        timestamp = get_current_time_millis()
        client.log_batch(
            run_id,
            metrics=[
                Metric(key=key, value=value, timestamp=timestamp, step=0)
                for key, value in metrics.items()
            ],
        )

    def _evaluate(self, y_pred):
        if self.model_type == "classifier":
            accuracy_score = sk_metrics.accuracy_score(self.y, y_pred)

            metrics = {"accuracy_score": accuracy_score}
            artifacts = {}
            self._log_metrics(self.run_id, metrics)
            confusion_matrix = sk_metrics.confusion_matrix(self.y, y_pred)
            confusion_matrix_artifact_name = "confusion_matrix"
            confusion_matrix_artifact = Array2DEvaluationArtifact(
                uri=get_artifact_uri(self.run_id, confusion_matrix_artifact_name + ".csv"),
                content=confusion_matrix,
            )
            confusion_matrix_csv_buff = io.StringIO()
            confusion_matrix_artifact._save(confusion_matrix_csv_buff)
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

            confusion_matrix_image_artifact_name = "confusion_matrix_image"
            confusion_matrix_image_artifact = ImageEvaluationArtifact(
                uri=get_artifact_uri(self.run_id, confusion_matrix_image_artifact_name + ".png"),
                content=confusion_matrix_image,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, confusion_matrix_image_artifact_name + ".png")
                confusion_matrix_image_artifact._save(path)
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
            self._log_metrics(self.run_id, metrics)
            artifacts = {}
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def evaluate(self, *, model, model_type, dataset, run_id, evaluator_config, **kwargs):
        self.model_type = model_type
        self.client = MlflowClient()
        self.dataset = dataset
        self.run_id = run_id
        self.X = dataset.features_data
        self.y = dataset.labels_data
        y_pred = model.predict(self.X) if model is not None else self.dataset.predictions_data
        return self._evaluate(y_pred)
