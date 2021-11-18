import mlflow
from mlflow.evaluation import ModelEvaluator, EvaluationMetrics, \
    EvaluationArtifact, EvaluationResult, EvaluationDataset
from mlflow.tracking.artifact_utils import get_artifact_uri
from sklearn import metrics as sk_metrics
import numpy as np
import json


class JsonEvaluationArtifact(EvaluationArtifact):

    def save_content_to_file(self, content, output_artifact_path):
        with open(output_artifact_path, 'w') as fp:
            json.dump(content, fp)

    @classmethod
    def load_content_from_file(cls, local_artifact_path):
        with open(local_artifact_path, 'r') as fp:
            return json.load(fp)


class DummyEvaluator(ModelEvaluator):

    def can_evaluate(
        self, model_type, evaluator_config=None, **kwargs
    ):
        return evaluator_config.get('can_evaluate') and \
               model_type in ['classifier', 'regressor']

    def compute_metrics_and_compute_and_log_artifacts(
            self, model_type, predict, dataset, evaluator_config, run_id
    ):
        X = dataset.data
        assert isinstance(X, np.ndarray), 'Only support array type feature input'
        assert dataset.name is not None, 'Dataset name required'
        y = dataset.labels
        y_pred = predict(X)
        if model_type == 'classifier':
            accuracy_score = sk_metrics.accuracy_score(y, y_pred)
            brier_score_loss = sk_metrics.brier_score_loss(y, y_pred)

            metrics = EvaluationMetrics(
                accuracy_score=accuracy_score,
                brier_score_loss=brier_score_loss
            )
            confusion_matrix = sk_metrics.confusion_matrix(y, y_pred)
            confusion_matrix_artifact_name = f'confusion_matrix_on_{dataset.name}.json'
            confusion_matrix_artifact = JsonEvaluationArtifact(
                location=get_artifact_uri(run_id, confusion_matrix_artifact_name),
                content=confusion_matrix
            )
            self.mlflow_client.log_dict(run_id, confusion_matrix)
            artifacts = {confusion_matrix_artifact_name: confusion_matrix_artifact}
            return metrics, artifacts
        elif model_type == 'regressor':
            mean_absolute_error = sk_metrics.mean_absolute_error(y, y_pred)
            mean_squared_error = sk_metrics.mean_squared_error(y, y_pred)
            return EvaluationMetrics(
                mean_absolute_error=mean_absolute_error,
                mean_squared_error=mean_squared_error
            ), {}
