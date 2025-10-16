import importlib
import logging
import pkgutil

from agno.models.base import Model

_logger = logging.getLogger(__name__)


def discover_storage_backends():
    try:
        from agno.storage.base import Storage

        storage_package = "agno.storage"
        storage_bases = (Storage,)
    except ImportError:
        try:
            from agno.db.base import AsyncBaseDb, BaseDb

            storage_package = "agno.db"
            storage_bases = tuple(cls for cls in (BaseDb, AsyncBaseDb) if cls is not None)
            if not storage_bases:
                return []
        except ImportError:
            return []

    pkg = importlib.import_module(storage_package)

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

    discovered = []
    for base in storage_bases:
        for cls in all_subclasses(base):
            if cls not in discovered:
                discovered.append(cls)

    return discovered


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
