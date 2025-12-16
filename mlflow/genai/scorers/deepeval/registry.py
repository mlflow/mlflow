from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.deepeval.utils import DEEPEVAL_NOT_INSTALLED_ERROR_MESSAGE

# Registry format: metric_name -> (classpath, is_deterministic)
_METRIC_REGISTRY = {
    # RAG Metrics
    "AnswerRelevancy": ("deepeval.metrics.AnswerRelevancyMetric", False),
    "Faithfulness": ("deepeval.metrics.FaithfulnessMetric", False),
    "ContextualRecall": ("deepeval.metrics.ContextualRecallMetric", False),
    "ContextualPrecision": ("deepeval.metrics.ContextualPrecisionMetric", False),
    "ContextualRelevancy": ("deepeval.metrics.ContextualRelevancyMetric", False),
    # Agentic Metrics
    "TaskCompletion": ("deepeval.metrics.TaskCompletionMetric", False),
    "ToolCorrectness": ("deepeval.metrics.ToolCorrectnessMetric", False),
    "ArgumentCorrectness": ("deepeval.metrics.ArgumentCorrectnessMetric", False),
    "StepEfficiency": ("deepeval.metrics.StepEfficiencyMetric", False),
    "PlanAdherence": ("deepeval.metrics.PlanAdherenceMetric", False),
    "PlanQuality": ("deepeval.metrics.PlanQualityMetric", False),
    # Conversational Metrics (multi-turn session-level)
    "TurnRelevancy": ("deepeval.metrics.TurnRelevancyMetric", False),
    "RoleAdherence": ("deepeval.metrics.RoleAdherenceMetric", False),
    "KnowledgeRetention": ("deepeval.metrics.KnowledgeRetentionMetric", False),
    "ConversationCompleteness": ("deepeval.metrics.ConversationCompletenessMetric", False),
    "GoalAccuracy": ("deepeval.metrics.GoalAccuracyMetric", False),
    "ToolUse": ("deepeval.metrics.ToolUseMetric", False),
    "TopicAdherence": ("deepeval.metrics.TopicAdherenceMetric", False),
    # Safety Metrics
    "Bias": ("deepeval.metrics.BiasMetric", False),
    "Toxicity": ("deepeval.metrics.ToxicityMetric", False),
    "NonAdvice": ("deepeval.metrics.NonAdviceMetric", False),
    "Misuse": ("deepeval.metrics.MisuseMetric", False),
    "PIILeakage": ("deepeval.metrics.PIILeakageMetric", False),
    "RoleViolation": ("deepeval.metrics.RoleViolationMetric", False),
    # General Metrics
    "Hallucination": ("deepeval.metrics.HallucinationMetric", False),
    "Summarization": ("deepeval.metrics.SummarizationMetric", False),
    "JsonCorrectness": ("deepeval.metrics.JsonCorrectnessMetric", False),
    "PromptAlignment": ("deepeval.metrics.PromptAlignmentMetric", False),
    # Deterministic Metrics
    "ExactMatch": ("deepeval.metrics.ExactMatchMetric", True),
    "PatternMatch": ("deepeval.metrics.PatternMatchMetric", True),
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

    classpath, _ = _METRIC_REGISTRY[metric_name]
    module_path, class_name = classpath.rsplit(".", 1)

    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise MlflowException.invalid_parameter_value(DEEPEVAL_NOT_INSTALLED_ERROR_MESSAGE) from e


def is_deterministic_metric(metric_name: str) -> bool:
    if metric_name not in _METRIC_REGISTRY:
        return False
    _, is_deterministic = _METRIC_REGISTRY[metric_name]
    return is_deterministic
