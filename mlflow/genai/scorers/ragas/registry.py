from __future__ import annotations

from mlflow.exceptions import MlflowException

# (classpath, is_deterministic)
_METRIC_REGISTRY = {
    # Retrieval Augmented Generation
    "ContextPrecision": ("ragas.metrics.ContextPrecision", False),
    "NonLLMContextPrecisionWithReference": (
        "ragas.metrics.NonLLMContextPrecisionWithReference",
        True,
    ),
    "ContextRecall": ("ragas.metrics.ContextRecall", False),
    "NonLLMContextRecall": ("ragas.metrics.NonLLMContextRecall", True),
    "ContextEntityRecall": ("ragas.metrics.ContextEntityRecall", False),
    "NoiseSensitivity": ("ragas.metrics.NoiseSensitivity", False),
    "ResponseRelevancy": ("ragas.metrics.ResponseRelevancy", False),
    "Faithfulness": ("ragas.metrics.Faithfulness", False),
    # Nvidia Metrics
    "AnswerAccuracy": ("ragas.metrics.AnswerAccuracy", False),
    "ContextRelevance": ("ragas.metrics.ContextRelevance", False),
    "ResponseGroundedness": ("ragas.metrics.ResponseGroundedness", False),
    # Agents or Tool Use Cases
    "TopicAdherence": ("ragas.metrics.TopicAdherenceScore", False),
    "ToolCallAccuracy": ("ragas.metrics.ToolCallAccuracy", True),
    "ToolCallF1": ("ragas.metrics.ToolCallF1", True),
    "AgentGoalAccuracyWithReference": (
        "ragas.metrics.AgentGoalAccuracyWithReference",
        False,
    ),
    "AgentGoalAccuracyWithoutReference": (
        "ragas.metrics.AgentGoalAccuracyWithoutReference",
        False,
    ),
    # Natural Language Comparison
    "FactualCorrectness": ("ragas.metrics.FactualCorrectness", False),
    "SemanticSimilarity": ("ragas.metrics.SemanticSimilarity", False),
    "NonLLMStringSimilarity": ("ragas.metrics.NonLLMStringSimilarity", True),
    "BleuScore": ("ragas.metrics.BleuScore", True),
    "ChrfScore": ("ragas.metrics.ChrfScore", True),
    "RougeScore": ("ragas.metrics.RougeScore", True),
    "StringPresence": ("ragas.metrics.StringPresence", True),
    "ExactMatch": ("ragas.metrics.ExactMatch", True),
    # TODO: SQL metrics not yet supported
    # "DatacompyScore": ("ragas.metrics.DatacompyScore", False),
    # "SQLSemanticEquivalence": ("ragas.metrics.SQLSemanticEquivalence", False),
    # General Purpose
    "AspectCritic": ("ragas.metrics.AspectCritic", False),
    "DiscreteMetric": ("ragas.metrics.DiscreteMetric", False),
    "RubricsScore": ("ragas.metrics.RubricsScore", False),
    "InstanceRubrics": ("ragas.metrics.InstanceRubrics", False),
    # Other Tasks
    "SummarizationScore": ("ragas.metrics.SummarizationScore", False),
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

    classpath, _ = _METRIC_REGISTRY[metric_name]
    module_path, class_name = classpath.rsplit(".", 1)

    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise MlflowException.invalid_parameter_value(
            "RAGAS metrics require the 'ragas' package. Please install it with: pip install ragas"
        ) from e


def is_deterministic_metric(metric_name: str) -> bool:
    _, is_deterministic = _METRIC_REGISTRY[metric_name]

    return is_deterministic


def no_llm_required_in_metric_constructor(metric_name: str) -> bool:
    metrics = {"DiscreteMetric"}

    return metric_name in metrics


def metric_requires_only_embeddings(metric_name: str) -> bool:
    metrics = {"SemanticSimilarity"}

    return metric_name in metrics


def is_agentic_metric(metric_name: str) -> bool:
    agentic_metrics = {
        "TopicAdherence",
        "ToolCallAccuracy",
        "ToolCallF1",
        "AgentGoalAccuracyWithReference",
        "AgentGoalAccuracyWithoutReference",
    }
    return metric_name in agentic_metrics
