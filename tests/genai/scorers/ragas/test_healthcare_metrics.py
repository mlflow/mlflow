import importlib.util
import sys
from types import ModuleType


def _load_healthcare_metrics_module():
    # Inject a lightweight mock for ragas.metrics so we can import the generated
    # healthcare module without depending on the full `ragas` package.
    metrics_mod = ModuleType("ragas.metrics")

    class AspectCritic:
        def __init__(self, name=None, definition=None, llm=None, output_type=None, **kwargs):
            self.name = name
            self.definition = definition
            self.llm = llm
            self.output_type = output_type

    class MetricOutputType:
        BINARY = "binary"

    metrics_mod.AspectCritic = AspectCritic
    metrics_mod.MetricOutputType = MetricOutputType
    sys.modules["ragas.metrics"] = metrics_mod

    spec = importlib.util.spec_from_file_location(
        "mlflow.genai.scorers.ragas.healthcare_metrics",
        "mlflow/genai/scorers/ragas/healthcare_metrics.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_healthcare_metric_classes_instantiate_and_have_expected_properties():
    mod = _load_healthcare_metrics_module()

    for cls_name in [
        "ClinicalAccuracy",
        "HIPAACompliance",
        "SourceAttribution",
        "MedicalTerminologyConsistency",
    ]:
        cls = getattr(mod, cls_name)
        inst = cls()
        assert inst.name == cls_name
        assert inst.output_type == "binary"


def test_registry_file_contains_healthcare_entries():
    registry_text = open("mlflow/genai/scorers/ragas/registry.py").read()

    # Sanity check: the registry file should contain the healthcare classpath strings
    assert "mlflow.genai.scorers.ragas.healthcare_metrics.ClinicalAccuracy" in registry_text
    assert "mlflow.genai.scorers.ragas.healthcare_metrics.HIPAACompliance" in registry_text
    assert "mlflow.genai.scorers.ragas.healthcare_metrics.SourceAttribution" in registry_text
    assert (
        "mlflow.genai.scorers.ragas.healthcare_metrics.MedicalTerminologyConsistency" in registry_text
    )
