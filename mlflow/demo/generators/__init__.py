from mlflow.demo.generators.evaluation import EvaluationDemoGenerator
from mlflow.demo.generators.prompts import PromptsDemoGenerator
from mlflow.demo.generators.traces import TracesDemoGenerator
from mlflow.demo.registry import demo_registry

demo_registry.register(TracesDemoGenerator)
demo_registry.register(EvaluationDemoGenerator)
demo_registry.register(PromptsDemoGenerator)

__all__ = ["EvaluationDemoGenerator", "PromptsDemoGenerator", "TracesDemoGenerator"]
