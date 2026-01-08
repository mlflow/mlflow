from __future__ import annotations

from mlflow.exceptions import MlflowException

# (classpath, is_deterministic)
_METRIC_REGISTRY = {
    # Retrieval Augmented Generation
    "ContextPrecision": ("ragas.metrics.collections.ContextPrecision", False),
    "ContextUtilization": ("ragas.metrics.collections.ContextUtilization", False),
    "NonLLMContextPrecisionWithReference": (
        "ragas.metrics.NonLLMContextPrecisionWithReference",
        True,
    ),
    "ContextRecall": ("ragas.metrics.collections.ContextRecall", False),
    "NonLLMContextRecall": ("ragas.metrics.NonLLMContextRecall", True),
    "ContextEntityRecall": ("ragas.metrics.collections.ContextEntityRecall", False),
    "NoiseSensitivity": ("ragas.metrics.collections.NoiseSensitivity", False),
    "AnswerRelevancy": ("ragas.metrics.collections.AnswerRelevancy", False),
    "Faithfulness": ("ragas.metrics.collections.Faithfulness", False),
    # Nvidia Metrics
    "AnswerAccuracy": ("ragas.metrics.collections.AnswerAccuracy", False),
    "ContextRelevance": ("ragas.metrics.collections.ContextRelevance", False),
    "ResponseGroundedness": ("ragas.metrics.collections.ResponseGroundedness", False),
    # Agents or Tool Use Cases
    "TopicAdherence": ("ragas.metrics.collections.TopicAdherence", False),
    "ToolCallAccuracy": ("ragas.metrics.collections.ToolCallAccuracy", True),
    "ToolCallF1": ("ragas.metrics.collections.ToolCallF1", True),
    "AgentGoalAccuracyWithReference": (
        "ragas.metrics.collections.AgentGoalAccuracyWithReference",
        False,
    ),
    "AgentGoalAccuracyWithoutReference": (
        "ragas.metrics.collections.AgentGoalAccuracyWithoutReference",
        False,
    ),
    # Natural Language Comparison
    "FactualCorrectness": ("ragas.metrics.collections.FactualCorrectness", False),
    "SemanticSimilarity": ("ragas.metrics.collections.SemanticSimilarity", False),
    "NonLLMStringSimilarity": (
        "ragas.metrics.collections.NonLLMStringSimilarity",
        True,
    ),
    "BleuScore": ("ragas.metrics.collections.BleuScore", True),
    "ChrfScore": ("ragas.metrics.collections.ChrfScore", True),
    "RougeScore": ("ragas.metrics.collections.RougeScore", True),
    "StringPresence": ("ragas.metrics.collections.StringPresence", True),
    "ExactMatch": ("ragas.metrics.collections.ExactMatch", True),
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


def is_agentic_metric(metric_name: str) -> bool:
    agentic_metrics = {
        "TopicAdherence",
        "ToolCallAccuracy",
        "ToolCallF1",
        "AgentGoalAccuracyWithReference",
        "AgentGoalAccuracyWithoutReference",
    }
    return metric_name in agentic_metrics


_NO_LLM_IN_CONSTRUCTOR = {
    "SemanticSimilarity",  # Only needs embeddings
    "DiscreteMetric",  # llm passed to score() instead
}
_REQUIRES_EMBEDDINGS = {"SemanticSimilarity", "AnswerRelevancy"}


def llm_in_constructor(metric_name: str) -> bool:
    return metric_name not in _NO_LLM_IN_CONSTRUCTOR


def requires_embeddings(metric_name: str) -> bool:
    return metric_name in _REQUIRES_EMBEDDINGS


def requires_llm_at_score_time(metric_name: str) -> bool:
    return metric_name == "DiscreteMetric"
