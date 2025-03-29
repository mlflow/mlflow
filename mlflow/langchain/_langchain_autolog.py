import importlib
import inspect
import logging

from mlflow.langchain import FLAVOR_NAME
from mlflow.models.model import _MODEL_TRACKER
from mlflow.utils.autologging_utils import safe_patch

_logger = logging.getLogger(__name__)


def _patch_runnable_cls(cls):
    """
    For classes that are subclasses of Runnable, we patch the `invoke`, `batch`, `stream` and
    `ainvoke`, `abatch`, `astream` methods for autologging.

    Args:
        cls: The class to patch.
    """
    for func_name in ["invoke", "batch", "stream", "ainvoke", "abatch", "astream"]:
        if hasattr(cls, func_name):
            safe_patch(
                FLAVOR_NAME,
                cls,
                func_name,
                _patched_inference,
            )


def _patched_inference(original, self, *args, **kwargs):
    """
    A patched implementation of langchain models inference process which enables
    logging the traces.

    We patch inference functions for different models based on their usage.
    """
    _MODEL_TRACKER.set_active_model_id_for_identity(id(self))
    return original(self, *args, **kwargs)


def _inspect_module_and_patch_cls(module_name, inspected_modules, patched_classes):
    """
    Internal method to inspect the module and patch classes that are
    subclasses of Runnable for autologging.
    """
    from langchain_core.runnables import Runnable

    if module_name not in inspected_modules:
        inspected_modules.add(module_name)

        try:
            for _, obj in inspect.getmembers(importlib.import_module(module_name)):
                if inspect.ismodule(obj) and (
                    obj.__name__.startswith("langchain") or obj.__name__.startswith("langgraph")
                ):
                    _inspect_module_and_patch_cls(obj.__name__, inspected_modules, patched_classes)
                elif (
                    inspect.isclass(obj)
                    and obj.__name__ not in patched_classes
                    and issubclass(obj, Runnable)
                ):
                    _patch_runnable_cls(obj)
                    patched_classes.add(obj.__name__)
        except Exception as e:
            _logger.debug(f"Failed to patch module {module_name}. Error: {e}", exc_info=True)
