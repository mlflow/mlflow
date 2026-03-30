from mlflow.demo.generators.evaluation import EvaluationDemoGenerator
from mlflow.demo.generators.issues import IssuesDemoGenerator
from mlflow.demo.generators.judges import JudgesDemoGenerator
from mlflow.demo.generators.prompts import PromptsDemoGenerator
from mlflow.demo.generators.traces import TracesDemoGenerator
from mlflow.demo.registry import demo_registry

# NB: Order matters here. Prompts must be created before traces (for linking),
# and traces must exist before evaluation (which references them).
# Judges are independent and can be registered last.
# Issues should be registered after traces exist (since they reference trace problems).
demo_registry.register(PromptsDemoGenerator)
demo_registry.register(TracesDemoGenerator)
demo_registry.register(EvaluationDemoGenerator)
demo_registry.register(JudgesDemoGenerator)
demo_registry.register(IssuesDemoGenerator)

__all__ = [
    "EvaluationDemoGenerator",
    "IssuesDemoGenerator",
    "JudgesDemoGenerator",
    "PromptsDemoGenerator",
    "TracesDemoGenerator",
]
