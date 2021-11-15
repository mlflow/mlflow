import mlflow
from mlflow.evaluation import ModelEvaluator, EvaluationMetrics, \
    EvaluationArtifact, EvaluationResult, EvaluationDataset


class DummyEvaluationResult(EvaluationResult):
    def __init__(self, predict, dataset, run_id, evaluator_config):
        # Dummy implementation, only store arguments passed to `evaluate`, for testing
        self.predict = predict
        self.dataset = dataset
        self.run_id = run_id
        self.evaluator_config = evaluator_config


class DummyClassifierEvaluationResult(DummyEvaluationResult)


class DummyClassifierEvaluator(ModelEvaluator):

    def can_evaluate(
        self, model_type, evaluator_config=None, **kwargs
    ):
        return model_type == 'classifier' and evaluator_config.get('can_evaluate')

    def _evaluate(self, predict, dataset, run_id, evaluator_config):
        return DummyClassifierEvaluationResult()

    def evaluate(
        self, predict, dataset, run_id, evaluator_config=None
    ):
        if run_id is not None:
            return self._evaluate(self, predict, dataset, run_id, evaluator_config)
        else:
            with mlflow.start_run() as run:
                return self._evaluate(self, predict, dataset, run.info.run_id, evaluator_config)


class DummyRegressorEvaluationResult(DummyEvaluationResult)


class DummyRegressorEvaluator(ModelEvaluator):

    def can_evaluate(
        self, model_type, evaluator_config=None, **kwargs
    ):
        return model_type == 'regressor' and evaluator_config.get('can_evaluate')

    def _evaluate(self, predict, dataset, run_id, evaluator_config):
        return DummyRegressorEvaluationResult()

    def evaluate(
        self, predict, dataset, run_id, evaluator_config=None
    ):
        if run_id is not None:
            return self._evaluate(self, predict, dataset, run_id, evaluator_config)
        else:
            with mlflow.start_run() as run:
                return self._evaluate(self, predict, dataset, run.info.run_id, evaluator_config)
