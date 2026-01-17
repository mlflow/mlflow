import os

# 1. Define the content for the new files
HEALTHCARE_METRICS_CONTENT = """from __future__ import annotations

import typing as t
from ragas.metrics import AspectCritic, MetricOutputType

class ClinicalAccuracy(AspectCritic):
    def __init__(self, llm=None, **kwargs):
        super().__init__(
            name="ClinicalAccuracy",
            definition="Assess if the answer is clinically accurate based on the context and general medical knowledge, respecting medical ontologies. Answer 1 if accurate, 0 otherwise.",
            llm=llm,
            output_type=MetricOutputType.BINARY,
            **kwargs
        )

class HIPAACompliance(AspectCritic):
    def __init__(self, llm=None, **kwargs):
        super().__init__(
            name="HIPAACompliance",
            definition="Assess if the output is HIPAA compliant. It should not contain any Protected Health Information (PHI) or Personally Identifiable Information (PII). Answer 1 if compliant, 0 otherwise.",
            llm=llm,
            output_type=MetricOutputType.BINARY,
            **kwargs
        )

class SourceAttribution(AspectCritic):
    def __init__(self, llm=None, **kwargs):
        super().__init__(
            name="SourceAttribution",
            definition="Assess if all medical claims in the answer are attributed to the provided context. Answer 1 if fully attributed, 0 otherwise.",
            llm=llm,
            output_type=MetricOutputType.BINARY,
            **kwargs
        )

class MedicalTerminologyConsistency(AspectCritic):
    def __init__(self, llm=None, **kwargs):
        super().__init__(
            name="MedicalTerminologyConsistency",
            definition="Assess if the medical terminology used in the answer is consistent throughout and matches the context. Answer 1 if consistent, 0 otherwise.",
            llm=llm,
            output_type=MetricOutputType.BINARY,
            **kwargs
        )
"""

SCORERS_HEALTHCARE_CONTENT = """from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.ragas import RagasScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ClinicalAccuracy(RagasScorer):
    metric_name: ClassVar[str] = "ClinicalAccuracy"

@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class HIPAACompliance(RagasScorer):
    metric_name: ClassVar[str] = "HIPAACompliance"

@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class SourceAttribution(RagasScorer):
    metric_name: ClassVar[str] = "SourceAttribution"

@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class MedicalTerminologyConsistency(RagasScorer):
    metric_name: ClassVar[str] = "MedicalTerminologyConsistency"
"""

REGISTRY_CONTENT = """from __future__ import annotations

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
"""

# 2. Paths
base_dir = os.path.join("mlflow", "genai", "scorers", "ragas")
scorers_dir = os.path.join(base_dir, "scorers")

# 3. Write files
print("Writing healthcare_metrics.py...")
with open(os.path.join(base_dir, "healthcare_metrics.py"), "w") as f:
    f.write(HEALTHCARE_METRICS_CONTENT)

print("Writing scorers/healthcare.py...")
with open(os.path.join(scorers_dir, "healthcare.py"), "w") as f:
    f.write(SCORERS_HEALTHCARE_CONTENT)

print("Updating registry.py...")
with open(os.path.join(base_dir, "registry.py"), "w") as f:
    f.write(REGISTRY_CONTENT)

# 4. Append to __init__.py files
print("Updating scorers/__init__.py...")
scorers_init_path = os.path.join(scorers_dir, "__init__.py")
with open(scorers_init_path, "r") as f:
    content = f.read()

if "ClinicalAccuracy" not in content:
    with open(scorers_init_path, "a") as f:
        f.write("\n# Healthcare Scorers\n")
        f.write("from mlflow.genai.scorers.ragas.scorers.healthcare import (\n")
        f.write("    ClinicalAccuracy,\n")
        f.write("    HIPAACompliance,\n")
        f.write("    SourceAttribution,\n")
        f.write("    MedicalTerminologyConsistency,\n")
        f.write(")\n")
        f.write("\n__all__.extend([\n")
        f.write('    "ClinicalAccuracy",\n')
        f.write('    "HIPAACompliance",\n')
        f.write('    "SourceAttribution",\n')
        f.write('    "MedicalTerminologyConsistency",\n')
        f.write("])\n")

print("Updating ragas/__init__.py...")
ragas_init_path = os.path.join(base_dir, "__init__.py")
with open(ragas_init_path, "r") as f:
    content = f.read()

if "ClinicalAccuracy" not in content:
    with open(ragas_init_path, "a") as f:
        f.write("\n# Healthcare Scorers\n")
        f.write("from mlflow.genai.scorers.ragas.scorers.healthcare import (\n")
        f.write("    ClinicalAccuracy,\n")
        f.write("    HIPAACompliance,\n")
        f.write("    SourceAttribution,\n")
        f.write("    MedicalTerminologyConsistency,\n")
        f.write(")\n")
        f.write("\n__all__.extend([\n")
        f.write('    "ClinicalAccuracy",\n')
        f.write('    "HIPAACompliance",\n')
        f.write('    "SourceAttribution",\n')
        f.write('    "MedicalTerminologyConsistency",\n')
        f.write("])\n")

print("Success! Healthcare metrics have been added.")
