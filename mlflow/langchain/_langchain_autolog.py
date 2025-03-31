import importlib
import inspect
import logging

import mlflow
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
    # NB: we only create LoggedModels for invoke/ainvoke methods
    # Other methods (batch/stream) invocation order on sub components is not guaranteed
    if hasattr(cls, "invoke"):
        safe_patch(
            FLAVOR_NAME,
            cls,
            "invoke",
            _patched_invoke,
        )
    if hasattr(cls, "ainvoke"):
        safe_patch(
            FLAVOR_NAME,
            cls,
            "ainvoke",
            _patched_ainvoke,
        )
    for func_name in ["batch", "stream", "abatch", "astream"]:
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
    if model_id := _MODEL_TRACKER.get(id(self)):
        _MODEL_TRACKER.set_active_model_id(model_id)
    else:
        _MODEL_TRACKER.set_active_model_id(None)
    return original(self, *args, **kwargs)


def _patched_invoke(original, self, *args, **kwargs):
    """
    A patched implementation of langchain models invoke process which enables
    logging the traces.
    """
    if model_id := _MODEL_TRACKER.get(id(self)):
        _MODEL_TRACKER.set_active_model_id(model_id)
    # NB: this check ensures we don't create LoggedModels for internal components
    elif not _MODEL_TRACKER._is_active_model_id_set:
        logged_model = mlflow.create_logged_model(
            name=self.__class__.__name__,
        )
        _MODEL_TRACKER.set(id(self), logged_model.model_id)
        _MODEL_TRACKER.set_active_model_id(logged_model.model_id)
        _logger.debug(
            f"Created LoggedModel with model_id {logged_model.model_id} "
            f"for {self.__class__.__name__}"
        )
    else:
        _MODEL_TRACKER.set_active_model_id(None)
        # return directly in this case to avoid mutating _is_active_model_id_set
        return original(self, *args, **kwargs)

    _MODEL_TRACKER._is_active_model_id_set = True
    try:
        return original(self, *args, **kwargs)
    finally:
        _MODEL_TRACKER._is_active_model_id_set = False


async def _patched_ainvoke(original, self, *args, **kwargs):
    """
    A patched implementation of langchain models ainvoke process which enables
    logging the traces.
    """
    if model_id := _MODEL_TRACKER.get(id(self)):
        _MODEL_TRACKER.set_active_model_id(model_id)
    # NB: this check ensures we don't create LoggedModels for internal components
    elif not _MODEL_TRACKER._is_active_model_id_set:
        logged_model = mlflow.create_logged_model(
            name=self.__class__.__name__,
        )
        _MODEL_TRACKER.set(id(self), logged_model.model_id)
        _MODEL_TRACKER.set_active_model_id(logged_model.model_id)
        _logger.debug(
            f"Created LoggedModel with model_id {logged_model.model_id} "
            f"for {self.__class__.__name__}"
        )
    else:
        _MODEL_TRACKER.set_active_model_id(None)
        # return directly in this case to avoid mutating _is_active_model_id_set
        return await original(self, *args, **kwargs)

    _MODEL_TRACKER._is_active_model_id_set = True
    try:
        return await original(self, *args, **kwargs)
    finally:
        _MODEL_TRACKER._is_active_model_id_set = False


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
