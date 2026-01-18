from __future__ import annotations

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
