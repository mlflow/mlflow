import mlflow
from mlflow.evaluation import ModelEvaluator, EvaluationMetrics, \
    EvaluationArtifact, EvaluationResult, EvaluationDataset
from sklearn import metrics as sk_metrics
import numpy as np
import pickle


class DummyEvaluationArtifact(EvaluationArtifact):

    def __init__(self, content, location):
        self.content = content
        self.location = location

    @property
    def content(self):
        return self.content

    @property
    def location(self) -> str:
        return self.location


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
        return self.metric_values()

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
        return evaluator_config.get('can_evaluate')

    def _evaluate(self, predict, dataset, run_id, evaluator_config):
        X = dataset.data
        assert isinstance(X, np.ndarray), 'Only support array type feature input'
        assert dataset.name is not None, 'Dataset name required'
        y = dataset.labels
        y_pred = predict(X)

        metrics_to_calc = evaluator_config.get('metrics_to_calc')
        metric_values = {}
        for metric_name in metrics_to_calc:
            metric_values[metric_name] = getattr(sk_metrics, metric_name)(y, y_pred)

        return DummyEvaluationResult(metric_values)

    def evaluate(
        self, predict, dataset, run_id, evaluator_config=None
    ):
        if run_id is not None:
            return self._evaluate(self, predict, dataset, run_id, evaluator_config)
        else:
            with mlflow.start_run() as run:
                return self._evaluate(self, predict, dataset, run.info.run_id, evaluator_config)


class DummyRegressorEvaluator(DummyEvaluator):

    def can_evaluate(
        self, model_type, evaluator_config=None, **kwargs
    ):
        return model_type == 'regressor' and super(DummyRegressorEvaluator, self).can_evaluate(
            model_type, evaluator_config=None, **kwargs
        )


class DummyClassifierEvaluator(DummyEvaluator):

    def can_evaluate(
        self, model_type, evaluator_config=None, **kwargs
    ):
        return model_type == 'classifier' and super(DummyClassifierEvaluator, self).can_evaluate(
            model_type, evaluator_config=None, **kwargs
        )
