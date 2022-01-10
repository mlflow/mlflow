import mlflow
from mlflow.models.evaluation import (
    ModelEvaluator,
    EvaluationMetrics,
    EvaluationArtifact,
    EvaluationResult,
)
from mlflow.tracking.artifact_utils import get_artifact_uri
from mlflow.entities import Metric
from sklearn import metrics as sk_metrics
import time
import pandas as pd
import io


class Array2DEvaluationArtifact(EvaluationArtifact):
    def save(self, output_artifact_path):
        pd.DataFrame(self._content).to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        pdf = pd.read_csv(local_artifact_path)
        return pdf.to_numpy()


class DummyEvaluator(ModelEvaluator):
    # pylint: disable=unused-argument
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs):
        return model_type in ["classifier", "regressor"]

    def _log_metrics(self, run_id, metrics, dataset_name):
        """
        Helper method to log metrics into specified run.
        """
        client = mlflow.tracking.MlflowClient()
        timestamp = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[
                Metric(key=f"{key}_on_{dataset_name}", value=value, timestamp=timestamp, step=0)
                for key, value in metrics.items()
            ],
        )

    # pylint: disable=unused-argument
    def evaluate(
        self, *, model, model_type, dataset, run_id, evaluator_config, **kwargs
    ) -> EvaluationResult:
        client = mlflow.tracking.MlflowClient()
        X = dataset.features_data
        y = dataset.labels_data
        y_pred = model.predict(X)
        if model_type == "classifier":
            accuracy_score = sk_metrics.accuracy_score(y, y_pred)

            metrics = EvaluationMetrics(accuracy_score=accuracy_score)
            self._log_metrics(run_id, metrics, dataset.name)
            confusion_matrix = sk_metrics.confusion_matrix(y, y_pred)
            confusion_matrix_artifact_name = f"confusion_matrix_on_{dataset.name}.csv"
            confusion_matrix_artifact = Array2DEvaluationArtifact(
                uri=get_artifact_uri(run_id, confusion_matrix_artifact_name),
                content=confusion_matrix,
            )
            confusion_matrix_csv_buff = io.StringIO()
            confusion_matrix_artifact.save(confusion_matrix_csv_buff)
            client.log_text(
                run_id, confusion_matrix_csv_buff.getvalue(), confusion_matrix_artifact_name
            )
            artifacts = {confusion_matrix_artifact_name: confusion_matrix_artifact}
        elif model_type == "regressor":
            mean_absolute_error = sk_metrics.mean_absolute_error(y, y_pred)
            mean_squared_error = sk_metrics.mean_squared_error(y, y_pred)
            metrics = EvaluationMetrics(
                mean_absolute_error=mean_absolute_error, mean_squared_error=mean_squared_error
            )
            self._log_metrics(run_id, metrics, dataset.name)
            artifacts = {}
        else:
            raise ValueError(f"Unsupported model type {model_type}")

        return EvaluationResult(metrics=metrics, artifacts=artifacts)
