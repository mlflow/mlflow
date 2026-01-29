"""Demo data generators."""

from mlflow.demo.generators.traces import TracesDemoGenerator
from mlflow.demo.registry import demo_registry

# Register all generators
demo_registry.register(TracesDemoGenerator)

__all__ = ["TracesDemoGenerator"]
