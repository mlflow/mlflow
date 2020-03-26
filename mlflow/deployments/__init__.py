from mlflow.deployments.base_plugin import BasePlugin
from mlflow.deployments import interface


__all__ = ['create', 'delete', 'update', 'list', 'describe', 'BasePlugin']


def __dir__():
    return list(globals().keys()) + __all__


def __getattr__(name):
    """
    Lazy loader to avoid loading the plugins until a request is triggered which
    requires to use the plugin functions.
    The reasoning here is that the community developed plugin is something the MLFlow core doesn't
    have any control of and could be doing heavy lifting on the import. This heavy lifting
    could cause the whole mlflow application to slow down. And the problem escalates if there
    more than one such plugins installed.
    We can't do this in interface.py since the call to any functions would not reach __getattr__
    because those functions are already in the name scope and has precedence over __getattr__
    in the lookup. Even if we find a way to do it, on each __getattr__ call, we would end up
    doing the plugin registration, which seems not ideal
    """
    if not interface.plugin_store.has_plugins_loaded:
        interface.plugin_store.register_entrypoints()
    try:
        return interface.__dict__[name]
    except KeyError:
        raise AttributeError(f"module {__name__} has no attribute {name}")
