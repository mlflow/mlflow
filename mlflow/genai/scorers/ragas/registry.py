from __future__ import annotations

from dataclasses import dataclass

from mlflow.exceptions import MlflowException


@dataclass(frozen=True)
class MetricConfig:
    classpath: str
    is_deterministic: bool = False
    is_agentic: bool = False
    requires_embeddings: bool = False
    llm_in_constructor: bool = True
    llm_at_score_time: bool = False


_METRIC_REGISTRY: dict[str, MetricConfig] = {
    # Retrieval Augmented Generation
    "ContextPrecision": MetricConfig("ragas.metrics.collections.ContextPrecision"),
    "ContextUtilization": MetricConfig("ragas.metrics.collections.ContextUtilization"),
    "NonLLMContextPrecisionWithReference": MetricConfig(
        "ragas.metrics.NonLLMContextPrecisionWithReference", is_deterministic=True
    ),
    "ContextRecall": MetricConfig("ragas.metrics.collections.ContextRecall"),
    "NonLLMContextRecall": MetricConfig("ragas.metrics.NonLLMContextRecall", is_deterministic=True),
    "ContextEntityRecall": MetricConfig("ragas.metrics.collections.ContextEntityRecall"),
    "NoiseSensitivity": MetricConfig("ragas.metrics.collections.NoiseSensitivity"),
    "AnswerRelevancy": MetricConfig(
        "ragas.metrics.collections.AnswerRelevancy", requires_embeddings=True
    ),
    "Faithfulness": MetricConfig("ragas.metrics.collections.Faithfulness"),
    # Nvidia Metrics
    "AnswerAccuracy": MetricConfig("ragas.metrics.collections.AnswerAccuracy"),
    "ContextRelevance": MetricConfig("ragas.metrics.collections.ContextRelevance"),
    "ResponseGroundedness": MetricConfig("ragas.metrics.collections.ResponseGroundedness"),
    # Agents or Tool Use Cases
    "TopicAdherence": MetricConfig("ragas.metrics.collections.TopicAdherence", is_agentic=True),
    "ToolCallAccuracy": MetricConfig(
        "ragas.metrics.collections.ToolCallAccuracy",
        is_deterministic=True,
        is_agentic=True,
    ),
    "ToolCallF1": MetricConfig(
        "ragas.metrics.collections.ToolCallF1", is_deterministic=True, is_agentic=True
    ),
    "AgentGoalAccuracyWithReference": MetricConfig(
        "ragas.metrics.collections.AgentGoalAccuracyWithReference", is_agentic=True
    ),
    "AgentGoalAccuracyWithoutReference": MetricConfig(
        "ragas.metrics.collections.AgentGoalAccuracyWithoutReference", is_agentic=True
    ),
    # Natural Language Comparison
    "FactualCorrectness": MetricConfig("ragas.metrics.collections.FactualCorrectness"),
    "SemanticSimilarity": MetricConfig(
        "ragas.metrics.collections.SemanticSimilarity",
        is_deterministic=True,
        requires_embeddings=True,
        llm_in_constructor=False,
    ),
    "NonLLMStringSimilarity": MetricConfig(
        "ragas.metrics.collections.NonLLMStringSimilarity", is_deterministic=True
    ),
    "BleuScore": MetricConfig("ragas.metrics.collections.BleuScore", is_deterministic=True),
    "CHRFScore": MetricConfig("ragas.metrics.collections.CHRFScore", is_deterministic=True),
    "RougeScore": MetricConfig("ragas.metrics.collections.RougeScore", is_deterministic=True),
    "StringPresence": MetricConfig(
        "ragas.metrics.collections.StringPresence", is_deterministic=True
    ),
    "ExactMatch": MetricConfig("ragas.metrics.collections.ExactMatch", is_deterministic=True),
    # General Purpose
    "AspectCritic": MetricConfig("ragas.metrics.AspectCritic"),
    "DiscreteMetric": MetricConfig(
        "ragas.metrics.DiscreteMetric",
        llm_in_constructor=False,
        llm_at_score_time=True,
    ),
    "RubricsScore": MetricConfig("ragas.metrics.RubricsScore"),
    "InstanceRubrics": MetricConfig("ragas.metrics.InstanceRubrics"),
    # Other Tasks
    "SummarizationScore": MetricConfig("ragas.metrics.collections.SummaryScore"),
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
    config = _get_config(metric_name)
    module_path, class_name = config.classpath.rsplit(".", 1)

    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise MlflowException.invalid_parameter_value(
            "RAGAS metrics require the 'ragas' package. Please install it with: pip install ragas"
        ) from e


def is_deterministic_metric(metric_name: str) -> bool:
    return _get_config(metric_name).is_deterministic


def is_agentic_metric(metric_name: str) -> bool:
    return _get_config(metric_name).is_agentic


def llm_in_constructor(metric_name: str) -> bool:
    return _get_config(metric_name).llm_in_constructor


def requires_embeddings(metric_name: str) -> bool:
    return _get_config(metric_name).requires_embeddings


def requires_llm_at_score_time(metric_name: str) -> bool:
    return _get_config(metric_name).llm_at_score_time


def _get_config(metric_name: str) -> MetricConfig:
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown metric: '{metric_name}'. Available metrics: {available_metrics}"
        )
    return _METRIC_REGISTRY[metric_name]
