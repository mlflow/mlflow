from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.ragas.utils import RAGAS_NOT_INSTALLED_ERROR_MESSAGE

_METRIC_REGISTRY = {
    # Retrieval Augmented Generation
    "ContextPrecision": "ragas.metrics.ContextPrecision",
    "NonLLMContextPrecisionWithReference": "ragas.metrics.NonLLMContextPrecisionWithReference",
    "ContextRecall": "ragas.metrics.ContextRecall",
    "NonLLMContextRecall": "ragas.metrics.NonLLMContextRecall",
    "ContextEntityRecall": "ragas.metrics.ContextEntityRecall",
    "NoiseSensitivity": "ragas.metrics.NoiseSensitivity",
    # TODO: ResponseRelevancy requires embeddings model instead of LLM
    # "ResponseRelevancy": "ragas.metrics.ResponseRelevancy",
    "Faithfulness": "ragas.metrics.Faithfulness",
    # TODO: Nvidia Metrics not yet supported
    # "AnswerAccuracy": "ragas.metrics.AnswerAccuracy",
    # "ContextRelevance": "ragas.metrics.ContextRelevance",
    # "ResponseGroundedness": "ragas.metrics.ResponseGroundedness",
    # TODO: Agents or Tool Use Cases metrics not yet supported
    # "TopicAdherence": "ragas.metrics.TopicAdherence",
    # "ToolCallAccuracy": "ragas.metrics.ToolCallAccuracy",
    # "ToolCallF1": "ragas.metrics.ToolCallF1",
    # "AgentGoalAccuracy": "ragas.metrics.AgentGoalAccuracy",
    # Natural Language Comparison
    "FactualCorrectness": "ragas.metrics.FactualCorrectness",
    # TODO: SemanticSimilarity requires embeddings model instead of LLM
    # "SemanticSimilarity": "ragas.metrics.SemanticSimilarity",
    "NonLLMStringSimilarity": "ragas.metrics.NonLLMStringSimilarity",
    "BleuScore": "ragas.metrics.BleuScore",
    "ChrfScore": "ragas.metrics.ChrfScore",
    "RougeScore": "ragas.metrics.RougeScore",
    "StringPresence": "ragas.metrics.StringPresence",
    "ExactMatch": "ragas.metrics.ExactMatch",
    # TODO: SQL metrics not yet supported
    # "DatacompyScore": "ragas.metrics.DatacompyScore",
    # "SQLSemanticEquivalence": "ragas.metrics.SQLSemanticEquivalence",
    # General Purpose
    "AspectCritic": "ragas.metrics.AspectCritic",
    # TODO: DiscreteMetric not yet supported
    # "DiscreteMetric": "ragas.metrics.DiscreteMetric",
    "RubricsScore": "ragas.metrics.RubricsScore",
    "InstanceRubrics": "ragas.metrics.InstanceRubrics",
    # Other Tasks
    "SummarizationScore": "ragas.metrics.SummarizationScore",
}


def get_metric_class(metric_name: str):
    """
    Get RAGAS metric class by name.

    Args:
        metric_name: Name of the metric (e.g., "Faithfulness", "ContextPrecision")

    Returns:
        The RAGAS metric class

    Raises:
        MlflowException: If the metric name is not recognized or ragas is not installed
    """
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown metric: '{metric_name}'. Available metrics: {available_metrics}"
        )

    module_path, class_name = _METRIC_REGISTRY[metric_name].rsplit(".", 1)

    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise MlflowException.invalid_parameter_value(RAGAS_NOT_INSTALLED_ERROR_MESSAGE) from e


def is_deterministic_metric(metric_name: str) -> bool:
    deterministic_metrics = {
        "BleuScore",
        "RougeScore",
        "ExactMatch",
        "NonLLMStringSimilarity",
        "StringPresence",
        "ChrfScore",
        "NonLLMContextRecall",
        "NonLLMContextPrecisionWithReference",
    }
    return metric_name in deterministic_metrics
