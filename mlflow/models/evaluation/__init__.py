from .base import (
    _model_evaluation_registry,
    evaluate,
    EvaluationDataset,
    EvaluationArtifact,
    EvaluationMetrics,
    ModelEvaluator
)
from .default_evaluator import DefaultEvaluator

_model_evaluation_registry.register('default', DefaultEvaluator)
