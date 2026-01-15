import logging

from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DEMO_PROMPT_PREFIX, BaseDemoGenerator, DemoResult
from mlflow.demo.registry import demo_registry

_logger = logging.getLogger(__name__)

__all__ = [
    "DEMO_EXPERIMENT_NAME",
    "DEMO_PROMPT_PREFIX",
    "BaseDemoGenerator",
    "DemoResult",
    "demo_registry",
    "generate_all_demos",
]


def generate_all_demos() -> list[DemoResult]:
    results = []
    for name in demo_registry.list_generators():
        generator_cls = demo_registry.get(name)
        generator = generator_cls()
        if generator.is_generated():
            _logger.debug(f"Demo '{name}' already exists, skipping")
            continue
        _logger.info(f"Generating demo data for '{name}'")
        result = generator.generate()
        generator.store_version()
        results.append(result)
    return results
