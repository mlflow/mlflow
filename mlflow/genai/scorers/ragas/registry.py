from __future__ import annotations

from dataclasses import dataclass

from mlflow.exceptions import MlflowException


@dataclass(frozen=True)
class MetricConfig:
    classpath: str
    is_agentic_or_multiturn: bool = False
    requires_embeddings: bool = False
    requires_llm_in_constructor: bool = True
    requires_llm_at_score_time: bool = False
    requires_args_from_placeholders: bool = False


_METRIC_REGISTRY: dict[str, MetricConfig] = {
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
    # TODO: ResponseRelevancy requires embeddings model instead of LLM
    # "ResponseRelevancy": ("ragas.metrics.ResponseRelevancy", False),
    "Faithfulness": ("ragas.metrics.Faithfulness", False),
    "FactualCorrectness": ("ragas.metrics.FactualCorrectness", False),
    "NonLLMStringSimilarity": ("ragas.metrics.NonLLMStringSimilarity", True),
    "BleuScore": ("ragas.metrics.BleuScore", True),
    "ChrfScore": ("ragas.metrics.ChrfScore", True),
    "CHRFScore": ("ragas.metrics.ChrfScore", True),
    "RougeScore": ("ragas.metrics.RougeScore", True),
    "StringPresence": ("ragas.metrics.StringPresence", True),
    "ExactMatch": ("ragas.metrics.ExactMatch", True),
    "AspectCritic": ("ragas.metrics.AspectCritic", False),
    "RubricsScore": ("ragas.metrics.RubricsScore", False),
    "InstanceRubrics": ("ragas.metrics.InstanceRubrics", False),
    "SummarizationScore": ("ragas.metrics.SummarizationScore", False),
    # Healthcare
    "ClinicalAccuracy": ("mlflow.genai.scorers.ragas.healthcare_metrics.ClinicalAccuracy", False),
    "HIPAACompliance": ("mlflow.genai.scorers.ragas.healthcare_metrics.HIPAACompliance", False),
    "SourceAttribution": ("mlflow.genai.scorers.ragas.healthcare_metrics.SourceAttribution", False),
    "MedicalTerminologyConsistency": (
        "mlflow.genai.scorers.ragas.healthcare_metrics.MedicalTerminologyConsistency",
        False,
    ),
}


def get_metric_class(metric_name: str):
    """
    Resolve and return the metric class corresponding to the given metric name.

    This function looks up the provided ``metric_name`` in the internal
    ``_METRIC_REGISTRY`` mapping, dynamically imports the module that defines
    the metric, and returns the associated class object. It is part of the
    public API used by :class:`RagasScorer`.

    Parameters
    ----------
    metric_name : str
        The name of the metric to load. Must be one of the keys defined in
        ``_METRIC_REGISTRY``.

    Returns
    -------
    type
        The metric class corresponding to ``metric_name``.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If ``metric_name`` is not a known metric, or if the underlying metric
        class cannot be imported (for example, when RAGAS metrics are requested
        but the ``ragas`` package is not installed).
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


def is_agentic_or_multiturn_metric(metric_name: str) -> bool:
    return _get_config(metric_name).is_agentic_or_multiturn


def requires_llm_in_constructor(metric_name: str) -> bool:
    return _get_config(metric_name).requires_llm_in_constructor


def requires_embeddings(metric_name: str) -> bool:
    return _get_config(metric_name).requires_embeddings


def requires_llm_at_score_time(metric_name: str) -> bool:
    return _get_config(metric_name).requires_llm_at_score_time


def requires_args_from_placeholders(metric_name: str) -> bool:
    return _get_config(metric_name).requires_args_from_placeholders


def _get_config(metric_name: str) -> MetricConfig:
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown metric: '{metric_name}'. Available metrics: {available_metrics}"
        )
    return _METRIC_REGISTRY[metric_name]
