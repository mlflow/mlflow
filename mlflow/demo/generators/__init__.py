from mlflow.demo.generators.evaluation import EvaluationDemoGenerator
from mlflow.demo.generators.prompts import PromptsDemoGenerator
from mlflow.demo.generators.traces import TracesDemoGenerator
from mlflow.demo.registry import demo_registry

# NB: Order matters here. Prompts must be created before traces (for linking),
# and traces must exist before evaluation (which references them)
demo_registry.register(PromptsDemoGenerator)
demo_registry.register(TracesDemoGenerator)
demo_registry.register(EvaluationDemoGenerator)

__all__ = ["EvaluationDemoGenerator", "PromptsDemoGenerator", "TracesDemoGenerator"]
