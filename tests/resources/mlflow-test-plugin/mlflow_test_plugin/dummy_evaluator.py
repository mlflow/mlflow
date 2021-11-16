import mlflow
from mlflow.evaluation import ModelEvaluator, EvaluationMetrics, \
    EvaluationArtifact, EvaluationResult, EvaluationDataset
from mlflow.tracking import artifact_utils
from sklearn import metrics as sk_metrics
import numpy as np
import pickle


class DummyEvaluationArtifact(EvaluationArtifact):

    def __init__(self, content, location):
        self._content = content
        self._location = location

    @property
    def content(self):
        return self._content

    @property
    def location(self) -> str:
        return self._location


class DummyEvaluationResult(EvaluationResult):

    def __init__(self, metric_values, artifact_content, artifact_location):
        self.metric_values = metric_values
        self.artifact_content = artifact_content
        self.artifact_location = artifact_location

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            obj = pickle.load(f)
        return obj

    def save(self, path):
        with open(path, 'w') as f:
            # TODO: skip dump artifact, instead, download artifact content when loading
            pickle.dump(self, f)

    @property
    def metrics(self):
        return self.metric_values

    @property
    def artifacts(self):
        return DummyEvaluationArtifact(
            content=self.artifact_content,
            location=self.artifact_location
        )


class DummyEvaluator(ModelEvaluator):

    def can_evaluate(
        self, model_type, evaluator_config=None, **kwargs
    ):
        return evaluator_config.get('can_evaluate') and \
               model_type in ['classifier', 'regressor']

    def _evaluate(self, predict, dataset, run_id, evaluator_config):
        X = dataset.data
        assert isinstance(X, np.ndarray), 'Only support array type feature input'
        assert dataset.name is not None, 'Dataset name required'
        y = dataset.labels
        y_pred = predict(X)

        metrics_to_calc = evaluator_config.get('metrics_to_calc')

        client = mlflow.tracking.MlflowClient()
        metric_values = {}
        for metric_name in metrics_to_calc:
            metric_value = getattr(sk_metrics, metric_name)(y, y_pred)
            metric_values[metric_name] = metric_value
            metric_key = f'{metric_name}_on_{dataset.name}'
            client.log_metric(run_id=run_id, key=metric_key, value=metric_value)

        client.log_dict(run_id, metric_values, 'metrics_artifact.json')

        # TODO: log `mlflow.datasets` tag containing a list of metadata for all datasets

        return DummyEvaluationResult(
            metric_values=metric_values,
            artifact_content=metric_values,
            artifact_location=artifact_utils.get_artifact_uri(run_id, 'metrics_artifact.json')
        )

    def evaluate(
        self, predict, dataset, run_id, evaluator_config=None
    ):
        if run_id is not None:
            return self._evaluate(predict, dataset, run_id, evaluator_config)
        elif mlflow.active_run() is not None:
            return self._evaluate(predict, dataset, mlflow.active_run().info.run_id,
                                  evaluator_config)
        else:
            with mlflow.start_run() as run:
                return self._evaluate(predict, dataset, run.info.run_id, evaluator_config)
