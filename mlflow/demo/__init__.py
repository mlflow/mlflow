import logging

import mlflow.demo.generators  # noqa: F401
from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    DEMO_PROMPT_PREFIX,
    BaseDemoGenerator,
    DemoResult,
    get_demo_experiment_name,
    set_uc_schema,
)
from mlflow.demo.registry import demo_registry
from mlflow.utils.workspace_context import WorkspaceContext, get_request_workspace

_logger = logging.getLogger(__name__)

__all__ = [
    "DEMO_EXPERIMENT_NAME",
    "DEMO_PROMPT_PREFIX",
    "BaseDemoGenerator",
    "DemoResult",
    "demo_registry",
    "generate_all_demos",
    "get_demo_experiment_name",
    "set_uc_schema",
]


def generate_all_demos(
    refresh: bool = False,
    features: list[str] | None = None,
) -> list[DemoResult]:
    results = []
    generator_names = demo_registry.list_generators()
    if features is not None:
        generator_names = [n for n in generator_names if n in features]

    # Propagate the workspace to the environment so that child threads spawned during
    # demo generation (e.g. by the evaluation harness's ThreadPoolExecutor) can resolve
    # the active workspace via the MLFLOW_WORKSPACE env-var fallback.  The ContextVar
    # set by the server middleware is thread-local and is invisible to new threads.
    with WorkspaceContext(get_request_workspace()):
        for name in generator_names:
            generator_cls = demo_registry.get(name)
            generator = generator_cls()
            if refresh:
                _logger.debug(f"Refresh requested, deleting existing demo data for '{name}'")
                generator.delete_demo()
            elif generator.is_generated():
                _logger.debug(f"Demo '{name}' already exists, skipping")
                continue
            _logger.info(f"Generating demo data for '{name}'")
            result = generator.generate()
            generator.store_version()
            results.append(result)

    return results
