"""Demo data generators."""

from mlflow.demo.generators.evaluation import EvaluationDemoGenerator
from mlflow.demo.generators.traces import TracesDemoGenerator
from mlflow.demo.registry import demo_registry

# Register all generators
demo_registry.register(TracesDemoGenerator)
demo_registry.register(EvaluationDemoGenerator)

__all__ = ["EvaluationDemoGenerator", "TracesDemoGenerator"]
