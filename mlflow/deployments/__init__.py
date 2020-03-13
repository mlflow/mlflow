from mlflow.deployments.base_plugin import BasePlugin
from mlflow.deployments import interface


# TODO: It's a good practise to avoid using ``list`` here

__all__ = ['BasePlugin', 'create', 'delete', 'update', 'list', 'describe']


def __dir__():
    return list(globals().keys()) + __all__


def __getattr__(name):
    """
    Lazy loader to avoid loading the plugins until a request
    to use functions from this file
    """
    if not interface.plugin_store.has_plugins_loaded:
        interface.plugin_store.register_entrypoints()
    try:
        return interface.__dict__[name]
    except KeyError:
        raise AttributeError(f"module {__name__} has no attribute {name}")
