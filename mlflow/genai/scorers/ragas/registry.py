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
    # TODO: ResponseRelevancy requires embeddings model instead of LLM
    # "ResponseRelevancy": ("ragas.metrics.ResponseRelevancy", False),
    "Faithfulness": ("ragas.metrics.Faithfulness", False),
    "FactualCorrectness": ("ragas.metrics.FactualCorrectness", False),
    "NonLLMStringSimilarity": ("ragas.metrics.NonLLMStringSimilarity", True),
    "BleuScore": ("ragas.metrics.BleuScore", True),
    "ChrfScore": ("ragas.metrics.ChrfScore", True),
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
