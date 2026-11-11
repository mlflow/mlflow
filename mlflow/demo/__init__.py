import logging
from collections.abc import Callable
from contextvars import ContextVar

import mlflow.demo.generators  # noqa: F401
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DEMO_PROMPT_PREFIX, BaseDemoGenerator, DemoResult
from mlflow.demo.registry import demo_registry
from mlflow.utils.workspace_context import WorkspaceContext, get_request_workspace

_logger = logging.getLogger(__name__)
_demo_refresh_check = ContextVar[Callable[[BaseDemoGenerator], bool] | None](
    "demo_refresh_check", default=None
)

__all__ = [
    "DEMO_EXPERIMENT_NAME",
    "DEMO_PROMPT_PREFIX",
    "BaseDemoGenerator",
    "DemoResult",
    "demo_registry",
    "generate_all_demos",
]


def generate_all_demos(
    refresh: bool = False,
    features: list[str] | None = None,
    *,
    can_refresh: Callable[[BaseDemoGenerator], bool] | None = None,
) -> list[DemoResult]:
    """Generate missing demo data and refresh existing data when permitted.

    ``can_refresh`` gates deletion for both stale data and explicit refreshes. A denied
    generator is skipped, while other missing features can still be generated. Nested
    generation inherits the same check; without a check, refreshes are unrestricted.
    """
    results = []
    generator_names = demo_registry.list_generators()
    if features is not None:
        generator_names = [n for n in generator_names if n in features]

    if can_refresh is None:
        can_refresh = _demo_refresh_check.get()
    token = _demo_refresh_check.set(can_refresh)
    try:
        # Propagate the workspace to the environment so that child threads spawned during
        # demo generation (e.g. by the evaluation harness's ThreadPoolExecutor) can resolve
        # the active workspace via the MLFLOW_WORKSPACE env-var fallback. The ContextVar
        # set by the server middleware is thread-local and is invisible to new threads.
        with WorkspaceContext(get_request_workspace()):
            for name in generator_names:
                generator_cls = demo_registry.get(name)
                generator = generator_cls()
                data_exists = generator._data_exists()
                if (
                    not refresh
                    and data_exists
                    and generator._get_stored_version() == generator.version
                ):
                    _logger.debug(f"Demo '{name}' already exists, skipping")
                    continue
                if refresh or data_exists:
                    if can_refresh is not None and not can_refresh(generator):
                        _logger.debug(f"Refresh not permitted for demo '{name}', skipping")
                        continue
                    _logger.debug(f"Deleting existing demo data for '{name}' before regeneration")
                    generator.delete_demo()
                _logger.info(f"Generating demo data for '{name}'")
                result = generator.generate()
                generator.store_version()
                results.append(result)
    finally:
        _demo_refresh_check.reset(token)

    return results
