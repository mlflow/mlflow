from __future__ import annotations

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
