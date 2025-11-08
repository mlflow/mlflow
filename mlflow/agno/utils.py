import importlib
import logging
import pkgutil

from agno.models.base import Model
from agno.storage.base import Storage

_logger = logging.getLogger(__name__)


def discover_storage_backends():
    # 1. Import all storage modules
    import agno.storage as pkg

    for _, modname, _ in pkgutil.iter_modules(pkg.__path__):
        try:
            importlib.import_module(f"{pkg.__name__}.{modname}")
        except ImportError as e:
            _logger.debug(f"Failed to import {modname}: {e}")
            continue

    # 2. Recursively collect subclasses
    def all_subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from all_subclasses(sub)

    return list(all_subclasses(Storage))


def find_model_subclasses():
    # 1. Import all Model modules
    import agno.models as pkg

    for _, modname, _ in pkgutil.iter_modules(pkg.__path__):
        try:
            importlib.import_module(f"{pkg.__name__}.{modname}")
        except ImportError as e:
            _logger.debug(f"Failed to import {modname}: {e}")
            continue

    # 2. Recursively collect subclasses
    def all_subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from all_subclasses(sub)

    models = list(all_subclasses(Model))
    # Sort so that more specific classes are patched before their bases
    models.sort(key=lambda c: len(c.__mro__), reverse=True)
    return models
