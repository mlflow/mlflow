import inspect
import logging

from mlflow.agno.autolog import patched_async_class_call, patched_class_call
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "agno"
_logger = logging.getLogger(__name__)


@experimental(version="3.3.0")
@autologging_integration(FLAVOR_NAME)
def autolog(*, log_traces: bool = True, disable: bool = False, silent: bool = False) -> None:
    """
    Enables (or disables) and configures autologging from Agno to MLflow.

    Args:
        log_traces: If ``True``, traces are logged for Agno Agents.
        disable: If ``True``, disables Agno autologging.
        silent: If ``True``, suppresses all MLflow event logs and warnings.
    """
    from mlflow.agno.utils import discover_storage_backends, find_model_subclasses

    class_map = {
        "agno.agent.Agent": ["run", "arun"],
        "agno.team.Team": ["run", "arun"],
        "agno.tools.function.FunctionCall": ["execute", "aexecute"],
    }

    storages = discover_storage_backends()
    if storages:
        class_map.update(
            {
                cls.__module__ + "." + cls.__name__: [
                    "create",
                    "read",
                    "upsert",
                    "drop",
                    "upgrade_schema",
                ]
                for cls in storages
            }
        )

    models = find_model_subclasses()

    if models:
        class_map.update(
            {
                # TODO: Support streaming
                cls.__module__ + "." + cls.__name__: ["invoke", "ainvoke"]
                for cls in models
            }
        )

    for cls_path, methods in class_map.items():
        mod_name, cls_name = cls_path.rsplit(".", 1)
        try:
            module = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError) as exc:
            _logger.debug("Agno autologging: failed to import %s – %s", cls_path, exc)
            continue

        for method_name in methods:
            try:
                original = getattr(cls, method_name)
                wrapper = (
                    patched_async_class_call
                    if inspect.iscoroutinefunction(original)
                    else patched_class_call
                )
                safe_patch(FLAVOR_NAME, cls, method_name, wrapper)
            except AttributeError as exc:
                _logger.debug(
                    "Agno autologging: cannot patch %s.%s – %s", cls_path, method_name, exc
                )
