from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.deepeval.utils import DEEPEVAL_NOT_INSTALLED_ERROR_MESSAGE

_MULTI_TURN_METRICS_REGISTRY = {
    "TurnRelevancy": "deepeval.metrics.TurnRelevancyMetric",
    "RoleAdherence": "deepeval.metrics.RoleAdherenceMetric",
    "KnowledgeRetention": "deepeval.metrics.KnowledgeRetentionMetric",
    "ConversationCompleteness": "deepeval.metrics.ConversationCompletenessMetric",
    "GoalAccuracy": "deepeval.metrics.GoalAccuracyMetric",
    "ToolUse": "deepeval.metrics.ToolUseMetric",
    "TopicAdherence": "deepeval.metrics.TopicAdherenceMetric",
}

_METRIC_REGISTRY = {
    # RAG Metrics
    "AnswerRelevancy": "deepeval.metrics.AnswerRelevancyMetric",
    "Faithfulness": "deepeval.metrics.FaithfulnessMetric",
    "ContextualRecall": "deepeval.metrics.ContextualRecallMetric",
    "ContextualPrecision": "deepeval.metrics.ContextualPrecisionMetric",
    "ContextualRelevancy": "deepeval.metrics.ContextualRelevancyMetric",
    # Agentic Metrics
    "TaskCompletion": "deepeval.metrics.TaskCompletionMetric",
    "ToolCorrectness": "deepeval.metrics.ToolCorrectnessMetric",
    "ArgumentCorrectness": "deepeval.metrics.ArgumentCorrectnessMetric",
    "StepEfficiency": "deepeval.metrics.StepEfficiencyMetric",
    "PlanAdherence": "deepeval.metrics.PlanAdherenceMetric",
    "PlanQuality": "deepeval.metrics.PlanQualityMetric",
    # Conversational Metrics (multi-turn session-level)
    **_MULTI_TURN_METRICS_REGISTRY,
    # Safety Metrics
    "Bias": "deepeval.metrics.BiasMetric",
    "Toxicity": "deepeval.metrics.ToxicityMetric",
    "NonAdvice": "deepeval.metrics.NonAdviceMetric",
    "Misuse": "deepeval.metrics.MisuseMetric",
    "PIILeakage": "deepeval.metrics.PIILeakageMetric",
    "RoleViolation": "deepeval.metrics.RoleViolationMetric",
    # General Metrics
    "Hallucination": "deepeval.metrics.HallucinationMetric",
    "Summarization": "deepeval.metrics.SummarizationMetric",
    "JsonCorrectness": "deepeval.metrics.JsonCorrectnessMetric",
    "PromptAlignment": "deepeval.metrics.PromptAlignmentMetric",
    # Deterministic Metrics
    "ExactMatch": "deepeval.metrics.ExactMatchMetric",
    "PatternMatch": "deepeval.metrics.PatternMatchMetric",
}


def get_metric_class(metric_name: str):
    """
    Get DeepEval metric class by name.

    Args:
        metric_name: Name of the metric (e.g., "AnswerRelevancy", "Faithfulness")

    Returns:
        The DeepEval metric class

    Raises:
        MlflowException: If the metric name is not recognized or deepeval is not installed
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
        raise MlflowException.invalid_parameter_value(DEEPEVAL_NOT_INSTALLED_ERROR_MESSAGE) from e


def is_deterministic_metric(metric_name: str):
    return metric_name in ("ExactMatch", "PatternMatch")


def is_multi_turn_metric(metric_name: str):
    return metric_name in _MULTI_TURN_METRICS_REGISTRY
